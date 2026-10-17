import { useMemo } from 'react'
import {
  Bars,
  choice,
  Figure,
  formatNumber,
  int,
  Plot,
  Points,
  Readout,
  useAxis,
  useFigureState,
  when,
} from 'aifn-render'
import { convolve, pad } from 'aifn-compute/foundation/convolution'
import { child, normals, permutation, stream } from 'aifn-compute/foundation/random'
import { matmul, mean, tensor, take, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { relu } from 'aifn-compute/nn/functional'
import { FixedHeight } from '../_shared/figure-helpers'

/** Signal length, as in the worked example. */
const N = 6
const WORKED = [1, 3, 0, 2, 0, 0]
const POSITIONS = Array.from({ length: N }, (_, i) => i)

type Action = 'shift' | 'reflect' | 'permute'
type KernelName = 'difference' | 'smoothing' | 'random'
/** Kernel taps a_0, a_1, ... and the tap that sits at the output position (its origin). */
const KERNELS: Record<KernelName, { taps: number[]; origin: number }> = {
  difference: { taps: [1, -1], origin: 0 },
  smoothing: { taps: [1, 2, 1], origin: 1 },
  random: { taps: [0.6, 1, -0.8], origin: 1 },
}

/** A fixed dense layer: a 6 × 6 matrix of seeded normals scaled by 1/√6. */
const DENSE = normals(child(stream('symmetry/dense'), 'w'), [N, N], 0, 1 / Math.sqrt(N))

/** Circular convolution y_i = Σ_u a_u x_{(i − u + origin) mod n}: wrap padding, then a valid convolution. */
function circularConvolution(x: number[], kernel: { taps: number[]; origin: number }): number[] {
  const k = kernel.taps.length
  const padded = pad(tensor(x), [k - 1 - kernel.origin, kernel.origin], 'wrap')
  return Array.from(toFlat(convolve(padded as Tensor, kernel.taps, { mode: 'valid' })))
}

/** The index map of the group element: (g·x)_i = x_{source[i]}. */
function sourceIndices(action: Action, amount: number, seed: number): number[] {
  if (action === 'shift') return POSITIONS.map((i) => (((i - amount) % N) + N) % N)
  if (action === 'reflect') return POSITIONS.map((i) => (N - i) % N)
  return Array.from(toFlat(permutation(stream(`symmetry/permutation/${seed}`), N)))
}

const act = (src: number[], x: number[]) => Array.from(toFlat(take(tensor(x), tensor(src)) as Tensor))
const dense = (x: number[]) => Array.from(toFlat(matmul(DENSE, tensor(x)) as Tensor))
const rectify = (x: number[]) => Array.from(toFlat(relu(tensor(x)) as Tensor))
/** The permutation-equivariant layer y = x − mean(x)·1 (a = 1, b = −1 in a·x + b·mean(x)·1). */
const centre = (x: number[]) => {
  const m = mean(tensor(x)) as number
  return x.map((v) => v - m)
}
const total = (x: number[]) => x.reduce((s, v) => s + v, 0)
const gap = (a: number[], b: number[]) => Math.max(...a.map((v, i) => Math.abs(v - b[i])))

/**
 * Which layers commute with a transformation of a length-6 cyclic signal: for each layer, the output on the
 * transformed input against the transformed output on the original input.
 */
export function EquivarianceCheck() {
  const state = useFigureState({
    action: choice(
      [
        { value: 'shift', label: 'cyclic shift' },
        { value: 'reflect', label: 'reflection i → −i' },
        { value: 'permute', label: 'random permutation' },
      ] as const,
      'shift',
      { label: 'transformation g' },
    ),
    amount: int(1, { ge: 0, le: N - 1, label: 'shift (positions)', when: when('action', 'shift') }),
    perm: int(1, { ge: 1, le: 999, label: 'permutation seed', when: when('action', 'permute') }),
    kernel: choice(
      [
        { value: 'difference', label: 'difference (1, −1)' },
        { value: 'smoothing', label: 'symmetric (1, 2, 1)' },
        { value: 'random', label: 'asymmetric (0.6, 1, −0.8)' },
      ] as const,
      'difference',
      { label: 'convolution kernel' },
    ),
    signal: int(0, { ge: 0, le: 999, label: 'signal (0 = worked example, else a seed)' }),
  })
  const { action, amount, perm, kernel, signal } = state

  const r = useMemo(() => {
    const x = signal === 0 ? WORKED : Array.from(toFlat(normals(stream(`symmetry/signal/${signal}`), [N])))
    const src = sourceIndices(action, amount, perm)
    const gx = act(src, x)
    const conv = (v: number[]) => circularConvolution(v, KERNELS[kernel])
    const layers = [
      { name: 'circular convolution', f: conv, slot: 0 },
      { name: 'dense layer', f: dense, slot: 1 },
      { name: 'ReLU after the convolution', f: (v: number[]) => rectify(conv(v)), slot: 0 },
      { name: 'x − mean(x) (Deep Sets layer)', f: centre, slot: 2 },
    ].map((l) => {
      const onTransformed = l.f(gx)
      const transformedOutput = act(src, l.f(x))
      return { ...l, onTransformed, transformedOutput, error: gap(onTransformed, transformedOutput) }
    })
    const invariant = [
      { name: 'sum ∘ ReLU ∘ conv', f: (v: number[]) => total(rectify(conv(v))) },
      { name: 'sum ∘ dense', f: (v: number[]) => total(dense(v)) },
    ].map((l) => ({ name: l.name, before: l.f(x), after: l.f(gx) }))
    return { x, gx, src, layers, invariant }
  }, [action, amount, perm, kernel, signal])

  const px = useAxis({ label: 'position i', range: [-0.6, N - 0.4], nice: false, integer: true })
  const yIn = useAxis({ label: 'value', hold: 'union' })
  const yConv = useAxis({ label: 'output', hold: 'union' })
  const yDense = useAxis({ label: 'output', hold: 'union' })
  const yRelu = useAxis({ label: 'output', hold: 'union' })
  const yCentre = useAxis({ label: 'output', hold: 'union' })
  const ix = useAxis({ label: '', range: [-0.6, 1.6], nice: false, integer: true })
  const iy = useAxis({ label: 'readout', hold: 'union' })
  const yAxes = [yConv, yDense, yRelu, yCentre]

  return (
    <Figure
      title="Which layers commute with the transformation"
      state={state}
      defaultSize="L"
      hoverReadout={false}
      caption={`A cyclic signal of length ${N} goes through four layers. For each layer, the bars are f(g·x), the output on the transformed input, and the dots are g·f(x), the transformed output on the original input. Bars on dots means f commutes with g. Circular convolution commutes with every cyclic shift, with reflection only for the symmetric kernel, and with a general permutation for no kernel. The dense layer commutes with none. A ReLU after the convolution inherits whatever the convolution has. The layer x − mean(x) commutes with every permutation, so with shifts and reflections too. The bottom-right panel sums the outputs over positions: the sum after an equivariant layer does not change under g, and the sum after the dense layer does. Change the transformation, the shift, the kernel or the signal.`}
      readouts={
        <>
          {r.layers.map((l) => (
            <Readout key={l.name} label={`${l.name}: max |f(g·x) − g·f(x)|`} value={formatNumber(l.error)} />
          ))}
          <Readout label="g as an index map: (g·x)ᵢ = x at" value={r.src.join(', ')} />
        </>
      }
    >
      <FixedHeight height={170}>
        <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
          <Plot x={px} y={yIn} title="input">
            <Bars name="g·x" x={POSITIONS} y={r.gx} muted />
            <Points name="x" x={POSITIONS} y={r.x} emphasis size={7} />
          </Plot>
          {r.layers.map((l, k) => (
            <Plot key={l.name} x={px} y={yAxes[k]} title={l.name}>
              <Bars name="f(g·x)" x={POSITIONS} y={l.onTransformed} slot={l.slot} />
              <Points name="g·f(x)" x={POSITIONS} y={l.transformedOutput} emphasis size={7} />
            </Plot>
          ))}
          <Plot x={ix} y={iy} title="sum: ReLU∘conv (left), dense (right)">
            <Bars name="on x" x={[-0.2, 0.8]} y={r.invariant.map((v) => v.before)} width={0.35} muted />
            <Bars name="on g·x" x={[0.2, 1.2]} y={r.invariant.map((v) => v.after)} width={0.35} slot={0} />
          </Plot>
        </div>
      </FixedHeight>
    </Figure>
  )
}
