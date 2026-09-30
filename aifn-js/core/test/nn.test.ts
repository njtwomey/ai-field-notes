import { describe, expect, it } from 'vitest'
import { gradCheck, grad } from 'aifn/foundation/autodiff'
import { binaryCrossEntropyWithLogits, softmaxCrossEntropy } from 'aifn/learning/losses'
import { adam, inspect, sgd, training } from 'aifn/nn/training'
import { avgPool2d, conv1d, conv2d, elu, gelu, leakyRelu, maxPool2d, relu, silu } from 'aifn/nn/functional'
import {
  batchNorm,
  causalMask,
  Conv2d,
  dropout,
  Embedding,
  GruCell,
  layerNorm,
  Linear,
  LstmCell,
  Mlp,
  multiHeadAttention,
  MultiHeadAttention,
  RnnCell,
  Sequential,
  unroll,
  ActivationLayer,
  Flatten,
} from 'aifn/nn/layers'
import { TransformerBlock } from 'aifn-applied/neural/architectures'
import { countParams } from 'aifn/foundation/pytree'
import { heUniform, xavierUniform } from 'aifn/nn/init'
import { stream } from 'aifn/foundation/random'
import { add, mul, sum, take, tensor, toFlat, unwrap, type Tensor, type Value } from 'aifn/foundation/tensor'
import { extend, run, seek, trace } from 'aifn/foundation/trace'
import { fixture } from './fixtures'

type N = number[] | number[][] | number[][][] | number[][][][]
type Case = Record<string, N | number | number[]>
const F = fixture<Record<string, Case>>('nn')
const T = (x: unknown) => tensor(x as N)
const flat = (v: Value) => {
  const r = unwrap(v)
  return typeof r === 'number' ? [r] : toFlat(r)
}
const flatN = (x: unknown): number[] => (Array.isArray(x) ? (x as unknown[]).flatMap(flatN) : [x as number])

function close(actual: Value | number[], expected: unknown, tol = 1e-9) {
  const a = Array.isArray(actual) ? actual : flat(actual)
  const e = flatN(expected)
  expect(a.length).toBe(e.length)
  let worst = 0
  a.forEach((v, i) => (worst = Math.max(worst, Math.abs(v - e[i]) / Math.max(1, Math.abs(e[i])))))
  expect(worst).toBeLessThan(tol)
}

/** Σ w·y as a function to differentiate. */
const weighted = (w: unknown) => (y: Value) => sum(mul(y, T(w)))

describe('convolution', () => {
  for (const name of ['conv2d_plain', 'conv2d_strided', 'conv2d_dilated']) {
    it(`${name} matches torch, value and gradients`, () => {
      const c = F[name]
      const opts = {
        stride: c.stride as [number, number],
        padding: c.padding as [number, number],
        dilation: c.dilation as [number, number],
      }
      close(conv2d(T(c.x), T(c.k), opts), c.y)
      const f = (x: Value, k: Value) => weighted(c.w)(conv2d(x, k, opts))
      const [gx, gk] = grad(f, { argnums: [0, 1] })(T(c.x), T(c.k))
      close(gx, c.gx)
      close(gk, c.gk)
    })
  }

  it('conv1d matches torch', () => {
    const c = F.conv1d
    const opts = { stride: 2, padding: 1, dilation: 2 }
    close(conv1d(T(c.x), T(c.k), opts), c.y)
    const [gx, gk] = grad((x: Value, k: Value) => weighted(c.w)(conv1d(x, k, opts)), { argnums: [0, 1] })(
      T(c.x),
      T(c.k),
    )
    close(gx, c.gx)
    close(gk, c.gk)
  })

  it('second derivatives through conv2d agree with finite differences', () => {
    const s = stream('conv-second')
    const x = Conv2d(1, 1, 2).init(s).weight
    const img = tensor([
      [
        [0.3, -0.2, 0.5],
        [0.1, 0.7, -0.4],
        [0.2, 0.0, 0.6],
      ],
    ])
    // g(k) = ‖∂/∂x Σ conv(x, k)²‖², a function of the kernel through the input gradient.
    const g = (k: Value) => {
      const gx = grad((u: Value) => sum(mul(conv2d(u, k), conv2d(u, k))))(img as Value)
      return sum(mul(gx, gx))
    }
    expect(gradCheck(g, x as Value, { rtol: 1e-4 }).ok).toBe(true)
  })

  it('pooling matches torch', () => {
    const m = F.maxpool2d
    close(maxPool2d(T(m.x), 2, { stride: 2, padding: 1 }), m.y)
    close(grad((x: Value) => weighted(m.w)(maxPool2d(x, 2, { stride: 2, padding: 1 })))(T(m.x)), m.gx)
    const a = F.avgpool2d
    close(avgPool2d(T(a.x), 3, { stride: 2, padding: 1 }), a.y)
    close(grad((x: Value) => weighted(a.w)(avgPool2d(x, 3, { stride: 2, padding: 1 })))(T(a.x)), a.gx)
  })
})

describe('normalisation', () => {
  it('layer norm matches torch', () => {
    const c = F.layernorm
    close(layerNorm(T(c.x), T(c.gamma), T(c.beta)), c.y)
    const [gx, gg, gb] = grad((x: Value, g: Value, b: Value) => weighted(c.w)(layerNorm(x, g, b)), {
      argnums: [0, 1, 2],
    })(T(c.x), T(c.gamma), T(c.beta))
    close(gx, c.gx, 1e-8)
    close(gg, c.ggamma)
    close(gb, c.gbeta)
  })

  it('batch norm matches torch in training mode', () => {
    const c = F.batchnorm
    close(batchNorm(T(c.x), T(c.gamma), T(c.beta)), c.y)
    const [gx, gg] = grad((x: Value, g: Value) => weighted(c.w)(batchNorm(x, g, T(c.beta))), { argnums: [0, 1] })(
      T(c.x),
      T(c.gamma),
    )
    close(gx, c.gx, 1e-8)
    close(gg, c.ggamma)
  })
})

describe('attention', () => {
  it('multi-head causal self-attention matches torch', () => {
    const c = F.mha
    const p = {
      query: { weight: T(c.wq), bias: T(c.bq) },
      key: { weight: T(c.wk), bias: T(c.bk) },
      value: { weight: T(c.wv), bias: T(c.bv) },
      output: { weight: T(c.wo), bias: T(c.bo) },
    }
    const heads = c.heads as number
    const { output, weights } = multiHeadAttention(p, T(c.x), T(c.x), { heads, causal: true })
    close(output, c.y)
    close(weights, c.weights)
    const gx = grad((x: Value) => weighted(c.w)(multiHeadAttention(p, x, x, { heads, causal: true }).output))(T(c.x))
    close(gx, c.gx, 1e-8)
  })

  it('a causal mask hides the future', () => {
    expect(toFlat(causalMask(3))).toEqual([1, 0, 0, 1, 1, 0, 1, 1, 1])
  })

  it('a transformer block keeps its shape and has finite gradients', () => {
    const block = TransformerBlock(8, { heads: 2, causal: true, hidden: 16 })
    const params = block.init(stream('block'))
    const x = tensor(Array.from({ length: 4 }, (_, t) => Array.from({ length: 8 }, (_, j) => Math.sin(t + j))))
    expect((unwrap(block.apply(params, x)) as Tensor).shape).toEqual([4, 8])
    const report = gradCheck((p: typeof params) => sum(block.apply(p, x)), params, { rtol: 1e-4, atol: 1e-6 })
    expect(report.ok).toBe(true)
  })
})

describe('recurrent cells', () => {
  it('LSTM, GRU and RNN cells match torch', () => {
    const l = F.lstm
    const lp = { inputWeight: T(l.inputWeight), hiddenWeight: T(l.hiddenWeight), bias: T(l.bias) }
    const lstm = LstmCell(3, 4)
    const next = lstm.step(lp, T(l.x), { h: T(l.h), c: T(l.c) })
    close(next.h, l.h1)
    close(next.c!, l.c1)
    const [gx, gh, gc] = grad(
      (x: Value, h: Value, c: Value) => {
        const s = lstm.step(lp, x, { h, c })
        return weighted(l.w)(add(s.h, mul(0.5, s.c!)))
      },
      { argnums: [0, 1, 2] },
    )(T(l.x), T(l.h), T(l.c))
    close(gx, l.gx)
    close(gh, l.gh)
    close(gc, l.gc)

    const g = F.gru
    const gp = {
      inputWeight: T(g.inputWeight),
      hiddenWeight: T(g.hiddenWeight),
      bias: T(g.bias),
      hiddenBias: T(g.hiddenBias),
    }
    const gru = GruCell(3, 4)
    close(gru.step(gp, T(g.x), { h: T(g.h) }).h, g.h1)
    close(grad((h: Value) => weighted(g.w)(gru.step(gp, T(g.x), { h }).h))(T(g.h)), g.gh)

    const r = F.rnn
    const rp = { inputWeight: T(r.inputWeight), hiddenWeight: T(r.hiddenWeight), bias: T(r.bias) }
    close(RnnCell(3, 4).step(rp, T(r.x), { h: T(r.h) }).h, r.h1)
  })

  it('unroll stacks hidden states over time', () => {
    const cell = GruCell(2, 3)
    const p = cell.init(stream('gru'))
    const xs = tensor([[[1, 0]], [[0, 1]], [[1, 1]]])
    const u = unroll(cell, p, xs)
    expect((unwrap(u.outputs) as Tensor).shape).toEqual([3, 1, 3])
    expect(u.states).toHaveLength(4)
  })
})

describe('layers and activations', () => {
  it('activations match torch', () => {
    const a = F.activations
    const x = T(a.x)
    close(gelu(x), a.gelu, 1e-12)
    close(gelu(x, { approximate: 'tanh' }), a.geluTanh, 1e-12)
    close(silu(x), a.silu, 1e-12)
    close(elu(x), a.elu, 1e-12)
    close(leakyRelu(x), a.leakyRelu, 1e-12)
    expect(grad((v: Value) => relu(v))(-1)).toBe(0)
    expect(grad((v: Value) => relu(v))(2)).toBe(1)
  })

  it('layer gradients agree with finite differences', () => {
    const s = stream('gradcheck')
    const model = Sequential(Conv2d(1, 2, 2), ActivationLayer('tanh'), Flatten(), Linear(8, 1))
    const params = model.init(s)
    const x = tensor([
      [
        [
          [0.1, 0.5, -0.3],
          [0.8, -0.2, 0.4],
          [0.0, 0.3, 0.9],
        ],
      ],
    ])
    expect(gradCheck((p: typeof params) => sum(model.apply(p, x)), params).ok).toBe(true)
    const emb = Embedding(5, 3)
    const ep = emb.init(s)
    expect(gradCheck((p: typeof ep) => sum(mul(emb.apply(p, tensor([1, 3, 1])), 1.5)), ep).ok).toBe(true)
  })

  it('take adds gradients over repeated rows', () => {
    const table = tensor([
      [1, 2],
      [3, 4],
    ])
    expect(toFlat(grad((t: Value) => sum(take(t, [1, 1, 0])))(table) as Tensor)).toEqual([1, 1, 2, 2])
  })

  it('dropout keeps the expectation and is reproducible', () => {
    const x = tensor(Array.from({ length: 4000 }, () => 1))
    const y = flat(dropout(stream('d'), x, 0.3))
    expect(y.reduce((a, b) => a + b, 0) / y.length).toBeCloseTo(1, 1)
    expect(flat(dropout(stream('d'), x, 0.3))).toEqual(y)
  })

  it('initialisers have their variances', () => {
    const w = xavierUniform()(stream('x'), [200, 300], { fanIn: 200, fanOut: 300 })
    const v = toFlat(w).reduce((a, b) => a + b * b, 0) / (200 * 300)
    expect(v).toBeCloseTo(2 / 500, 4)
    const h = heUniform()(stream('h'), [400, 50], { fanIn: 400, fanOut: 50 })
    const vh = toFlat(h).reduce((a, b) => a + b * b, 0) / (400 * 50)
    expect(Math.abs(vh / (2 / 400) - 1)).toBeLessThan(0.05)
  })

  it('inspect gives activation gradients consistent with parameter gradients', () => {
    const model = Mlp([2, 4, 1], { activation: 'tanh' })
    const params = model.init(stream('inspect'))
    const x = tensor([[0.5, -1]])
    const r = inspect(model, params, x, (y) => sum(y))
    expect(Object.keys(r.activations)).toEqual(['0', '1', '2'])
    // The output's own gradient is 1; the last layer's input gradient is its weight row.
    expect(flat(r.activationGrads['2'] as Tensor)).toEqual([1])
    expect(countParams(params)).toBe(2 * 4 + 4 + 4 + 1)
    const mha = MultiHeadAttention(4, { heads: 2 })
    const mp = mha.init(stream('m'))
    const att = inspect(
      mha,
      mp,
      tensor([
        [1, 0, 0, 1],
        [0, 1, 1, 0],
      ]),
      (y) => sum(y),
    )
    expect(Object.keys(att.activationGrads)).toContain('weights')
  })
})

describe('training', () => {
  const xor = {
    x: tensor([
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ]),
    y: tensor([0, 1, 1, 0]),
  }
  const model = Mlp([2, 8, 1], { activation: 'tanh', init: xavierUniform() })
  const lossOf = (p: ReturnType<typeof model.init>, b: typeof xor) =>
    binaryCrossEntropyWithLogits(model.apply(p, b.x), tensor(toFlat(b.y).map((v) => [v])))

  it('reduces the loss on XOR with Adam and with SGD', () => {
    for (const optimizer of [adam({ lr: 0.05 }), sgd({ lr: 0.5, momentum: 0.9 })]) {
      const alg = training({ loss: lossOf, data: xor, optimizer })
      const start = alg.init({ params: model.init(stream('xor')) })
      const end = run(alg, { params: model.init(stream('xor')) }, 400)
      expect(end.loss).toBeLessThan(0.05)
      expect(end.loss).toBeLessThan(start.loss)
    }
  })

  it('follows the trace protocol with minibatches', () => {
    const data = {
      x: tensor(Array.from({ length: 12 }, (_, i) => [Math.cos(i), Math.sin(i)])),
      y: tensor(Array.from({ length: 12 }, (_, i) => [i % 2])),
    }
    const alg = training({
      loss: (p: ReturnType<typeof model.init>, b: typeof data) =>
        binaryCrossEntropyWithLogits(model.apply(p, b.x), b.y),
      data,
      batchSize: 4,
    })
    const opts = { params: model.init(stream('protocol')) }
    const a = trace(alg, opts, 20, { stream: stream(1), record: { loss: (s) => s.loss } })
    const b = trace(alg, opts, 20, { stream: stream(1), record: { loss: (s) => s.loss } })
    expect(toFlat(a.series.loss)).toEqual(toFlat(b.series.loss))
    expect(seek(alg, opts, 13, { stream: stream(1) }).loss).toBe(a.steps[13].loss)
    const shorter = trace(alg, opts, 10, { stream: stream(1), record: { loss: (s) => s.loss } })
    const longer = extend(shorter, alg, opts, 10)
    expect(toFlat(longer.series.loss)).toEqual(toFlat(a.series.loss))
    expect(a.steps[3].epoch).toBe(1)
    expect(Object.keys(a.steps[0].gradNorms)).toContain('[0].weight')
  })

  it('a softmax classifier trains on three classes', () => {
    const data = {
      x: tensor([
        [0, 0],
        [1, 0],
        [0, 1],
        [0.1, 0.1],
        [0.9, 0.1],
        [0.1, 0.9],
      ]),
      y: tensor([0, 1, 2, 0, 1, 2]),
    }
    const net = Linear(2, 3)
    const alg = training({
      loss: (p: ReturnType<typeof net.init>, b: typeof data) => softmaxCrossEntropy(net.apply(p, b.x), b.y),
      data,
      optimizer: adam({ lr: 0.1 }),
    })
    expect(run(alg, { params: net.init(stream('softmax')) }, 300).loss).toBeLessThan(0.2)
  })
})
