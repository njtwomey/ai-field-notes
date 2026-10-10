import { useMemo, useState } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Player,
  Plot,
  Points,
  Raster,
  Readout,
  setting,
  slider,
  useAxis,
  useComputed,
  useFigureState,
  Vectors,
  type Vector,
} from 'aifn-render'
import { grad, stopGradient } from 'aifn-compute/foundation/autodiff'
import {
  add,
  matmul,
  mul,
  square,
  sub,
  sum,
  tensor,
  toFlat,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { assignNearest } from 'aifn-compute/numerics/neighbours'

type P = readonly [number, number]

/** The starting codebook: three codes in two dimensions. */
const CODES: readonly P[] = [
  [0, 0],
  [1.5, 1.2],
  [-1, 1.4],
]
const SPAN = 3
const GRID = Array.from({ length: 61 }, (_, i) => -SPAN + (2 * SPAN * i) / 60)
const GRID_ROWS = GRID.flatMap((y) => GRID.map((x) => [x, y]))
/** Arrows show each term's negative gradient times this, so that they stay on the plot. */
const ARROW = 0.3

const nearest = (z: P, codes: readonly P[]) => {
  let best = 0
  codes.forEach((e, j) => {
    if ((e[0] - z[0]) ** 2 + (e[1] - z[1]) ** 2 < (codes[best][0] - z[0]) ** 2 + (codes[best][1] - z[1]) ** 2) best = j
  })
  return best
}
const sq = (a: P, b: P) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

type Step = {
  z: P
  codes: P[]
  k: number
  rec: number
  codebook: number
  commitment: number
  /** Each term's gradient: reconstruction and commitment on z_e, codebook on the chosen code. */
  gRec: P
  gCommit: P
  gCodebook: P
}

/**
 * Gradient descent on one encoder output z_e (a free vector here, standing in for the encoder) and the codebook. The
 * decoder is the identity, so the reconstruction term is ‖z_q − t‖² for a target t. The loss is written as in the
 * paper, with stop-gradients, and differentiated by autodiff, one term at a time so that each can be drawn.
 */
function trace(o: {
  z0: P
  t: P
  beta: number
  rec: boolean
  codebook: boolean
  commitment: boolean
  rate: number
  steps: number
}): Step[] {
  const target = tensor([...o.t])
  let z: P = o.z0
  let codes: P[] = CODES.map((e) => [...e] as unknown as P)
  const out: Step[] = []
  for (let s = 0; s <= o.steps; s++) {
    const k = nearest(z, codes)
    const pick = tensor(codes.map((_, j) => (j === k ? 1 : 0)))
    // The chosen code e_k as a function of the codebook, so that gradients reach the row that was picked.
    const chosen = (e: Value) => matmul(pick, e)
    const terms = {
      rec: (p: { z: Value; e: Value }) => {
        const zq = add(p.z, stopGradient(sub(chosen(p.e), p.z)))
        return sum(square(sub(zq, target)))
      },
      codebook: (p: { z: Value; e: Value }) => sum(square(sub(stopGradient(p.z), chosen(p.e)))),
      commitment: (p: { z: Value; e: Value }) => mul(o.beta, sum(square(sub(p.z, stopGradient(chosen(p.e)))))),
    }
    const at = { z: tensor([...z]), e: tensor(codes.map((e) => [...e])) }
    const g = (term: keyof typeof terms) => {
      const d = grad(terms[term])(at)
      return { z: toFlat(d.z as Tensor), e: toFlat(d.e as Tensor) }
    }
    const gr = g('rec')
    const gc = g('codebook')
    const gm = g('commitment')
    const ek = codes[k]
    out.push({
      z,
      codes,
      k,
      rec: sq(ek, o.t),
      codebook: sq(z, ek),
      commitment: o.beta * sq(z, ek),
      gRec: [gr.z[0], gr.z[1]],
      gCommit: [gm.z[0], gm.z[1]],
      gCodebook: [gc.e[2 * k], gc.e[2 * k + 1]],
    })
    const on = (flag: boolean, v: number) => (flag ? v : 0)
    z = [
      z[0] - o.rate * (on(o.rec, gr.z[0]) + on(o.commitment, gm.z[0])),
      z[1] - o.rate * (on(o.rec, gr.z[1]) + on(o.commitment, gm.z[1])),
    ]
    codes = codes.map((e, j) => [
      e[0] - o.rate * on(o.codebook, gc.e[2 * j]),
      e[1] - o.rate * on(o.codebook, gc.e[2 * j + 1]),
    ])
  }
  return out
}

export function VqLossForces() {
  const state = useFigureState({
    zx: slider(-SPAN, SPAN, 1.2, { step: 0.01, onChart: true }),
    zy: slider(-SPAN, SPAN, 0.3, { step: 0.01, onChart: true }),
    tx: slider(-SPAN, SPAN, -0.9, { step: 0.01, onChart: true }),
    ty: slider(-SPAN, SPAN, -1.1, { step: 0.01, onChart: true }),
    rec: setting(true, 'reconstruction'),
    codebook: setting(true, 'codebook loss'),
    commitment: setting(true, 'commitment loss'),
    beta: float(0.25, { ge: 0, le: 4, suggestions: [0, 0.1, 0.25, 1, 2], label: 'commitment weight β' }),
    rate: float(0.1, { ge: 0.01, le: 0.4, suggestions: [0.05, 0.1, 0.2], label: 'step size' }),
    steps: int(60, { ge: 10, le: 300, suggestions: [30, 60, 120], label: 'steps' }),
  })
  const { zx, zy, tx, ty, rec, codebook, commitment, beta, rate, steps } = state

  const run = useComputed(
    () => trace({ z0: [zx, zy], t: [tx, ty], beta, rec, codebook, commitment, rate, steps }),
    [zx, zy, tx, ty, beta, rec, codebook, commitment, rate, steps],
  )
  const path = run.value
  const key = `${zx}|${zy}|${tx}|${ty}|${beta}|${rec}|${codebook}|${commitment}|${rate}|${steps}`
  const [pos, setPos] = useState({ key, step: 0 })
  const step = pos.key === key ? Math.min(pos.step, path.length - 1) : 0
  const now = path[step]

  const cells = useMemo(() => {
    const labels = toFlat(assignNearest(GRID_ROWS, now.codes).labels)
    return GRID.map((_, r) => labels.slice(r * GRID.length, (r + 1) * GRID.length))
  }, [now.codes])

  const zPath = useMemo(() => {
    const upTo = path.slice(0, step + 1)
    return { x: upTo.map((p) => p.z[0]), y: upTo.map((p) => p.z[1]) }
  }, [path, step])
  const codePaths = useMemo(
    () =>
      CODES.map((_, j) => {
        const upTo = path.slice(0, step + 1)
        return { x: upTo.map((p) => p.codes[j][0]), y: upTo.map((p) => p.codes[j][1]) }
      }),
    [path, step],
  )

  const arrows = useMemo(() => {
    const out: Vector[] = []
    const ek = now.codes[now.k]
    const arrow = (from: P, g: P, label: string, slot?: number): Vector => ({
      from: [from[0], from[1]],
      to: [from[0] - ARROW * g[0], from[1] - ARROW * g[1]],
      label,
      // The commitment and codebook arrows point at each other, so their labels sit at the midpoints, not the tips.
      labelAt: 'middle',
      slot,
    })
    if (rec) out.push(arrow(now.z, now.gRec, 'straight-through'))
    if (commitment && beta > 0) out.push(arrow(now.z, now.gCommit, 'commitment'))
    if (codebook) out.push(arrow(ek, now.gCodebook, 'codebook', now.k))
    return out
  }, [now, rec, commitment, codebook, beta])

  const x = useAxis({ label: 'latent z₁', range: [-SPAN, SPAN] })
  const y = useAxis({ label: 'latent z₂', range: [-SPAN, SPAN], equal: x })

  return (
    <Figure
      title="The three loss terms on one encoder output"
      defaultSize="L"
      state={state}
      caption={`One encoder output z_e (black dot) and a codebook of three codes (coloured dots), in a two-dimensional latent space shaded by the code each point would snap to. The decoder is the identity, so the reconstruction term is ‖z_q − t‖² for the target t (black cross): the decoder wants the chosen code to sit at t. Every step of gradient descent moves z_e and the codes by the gradients of the terms that are switched on, each computed by autodiff from the loss written with stop-gradients. The arrows are each term's negative gradient, times ${ARROW}: the straight-through reconstruction gradient and the commitment pull act on z_e, the codebook pull on the chosen code. Lines trace the paths so far. Drag z_e's starting point and the target t. Try the terms one at a time. With the reconstruction term alone, the straight-through gradient 2(e_k − t) is the same at every step while z_e stays in one cell, so z_e moves in a straight line, crosses cells and never stops, and the codes never move: nothing but the codebook loss trains them. Adding the codebook loss makes the chosen code follow z_e to t. The commitment loss pulls z_e back towards its code: at β = 0, z_e runs ahead of the code and overshoots; at β = 2 the two move together, more slowly.`}
      controls={
        <Player
          value={step}
          onChange={(s) => setPos({ key, step: s })}
          count={path.length}
          label="step"
          format={(s) => `step ${s} of ${path.length - 1}`}
        />
      }
      readouts={
        <>
          <Readout label="chosen code k*" value={String(now.k + 1)} />
          <Readout label="‖z_e − e_k*‖²" value={formatNumber(now.codebook)} />
          <Readout label="reconstruction ‖z_q − t‖²" value={formatNumber(now.rec)} />
          <Readout label="codebook loss" value={formatNumber(now.codebook)} />
          <Readout label="commitment loss β‖z_e − e_k*‖²" value={formatNumber(now.commitment)} />
        </>
      }
    >
      <Plot x={x} y={y}>
        <Raster name="code cells" x={GRID} y={GRID} z={cells} scale="categorical" fillOpacity={0.14} boundary />
        {codePaths.map((p, j) => (
          <Curve key={j} name={`code ${j + 1} path`} x={p.x} y={p.y} slot={j} width={1.5} />
        ))}
        <Curve name="z_e path" x={zPath.x} y={zPath.y} emphasis width={1.5} />
        {now.codes.map((e, j) => (
          <Points key={j} name={`code ${j + 1}`} x={[e[0]]} y={[e[1]]} slot={j} size={11} live />
        ))}
        <Points name="target t" x={[tx]} y={[ty]} emphasis shape={4} size={12} />
        <Points name="z_e" x={[now.z[0]]} y={[now.z[1]]} emphasis size={9} live />
        <Vectors vectors={arrows} />
        <Handle {...state.handle(['zx', 'zy'], { label: 'z_e start' })} />
        <Handle {...state.handle(['tx', 'ty'], { label: 't' })} />
      </Plot>
    </Figure>
  )
}
