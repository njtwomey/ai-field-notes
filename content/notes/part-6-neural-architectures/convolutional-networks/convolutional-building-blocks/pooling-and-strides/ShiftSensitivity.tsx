import { useMemo } from 'react'
import {
  Bars,
  choice,
  Curve,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { avgPool1d, conv1d, maxPool1d } from 'aifn-compute/nn/functional'
import { tensor, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { FixedHeight } from '../_shared/FixedHeight'

const L = 48
const SHIFTS = 16

/** A one-pixel spike, a two-pixel bar and a patch of fine texture, shifted right by t. */
const base = (n: number) =>
  (n === 10 ? 1 : 0) + (n >= 18 && n <= 19 ? 1 : 0) + (n >= 28 && n <= 35 ? 0.5 + 0.5 * Math.cos(Math.PI * n * 0.9) : 0)
const signal = (t: number) => Array.from({ length: L }, (_, n) => base(n - t))

const BLUR = tensor([[[0.25, 0.5, 0.25]]])
const PICK = tensor([[[1]]])

type Method = 'subsample' | 'max' | 'average' | 'blur'
const METHODS: { value: Method; label: string }[] = [
  { value: 'subsample', label: 'keep every 2nd sample (stride-2 1 × 1)' },
  { value: 'max', label: 'max pool 2, stride 2' },
  { value: 'average', label: 'average pool 2, stride 2' },
  { value: 'blur', label: 'max (stride 1), blur [1, 2, 1]/4, then keep every 2nd' },
]

/** Each downsampler as engine calls on a [1, L] signal; every one halves the length. */
const downsample: Record<Method, (x: Value) => Value> = {
  subsample: (x) => conv1d(x, PICK, { stride: 2 }),
  max: (x) => maxPool1d(x, 2),
  average: (x) => avgPool1d(x, 2),
  blur: (x) => conv1d(maxPool1d(x, 2, { stride: 1 }), BLUR, { stride: 2, padding: 1 }),
}

const run = (m: Method, t: number) => toFlat(downsample[m](tensor([signal(t)])) as Tensor)
const energy = (y: number[]) => y.reduce((a, v) => a + v * v, 0)

/**
 * Four ways to halve a signal, applied to the same signal shifted by t positions. A shift-equivariant downsampler
 * gives the same output, moved, for every even shift, and an output of nearly constant energy for every shift.
 */
export function ShiftSensitivity() {
  const state = useFigureState({
    method: choice<Method>(METHODS, 'max', { label: 'downsampler' }),
    shift: int(0, { ge: 0, le: SHIFTS - 1, label: 'input shift t', onChart: true }),
  })
  const t = state.shift

  const sweeps = useMemo(
    () =>
      METHODS.map(({ value }) => {
        const e = Array.from({ length: SHIFTS }, (_, s) => energy(run(value, s)))
        const lo = Math.min(...e)
        const hi = Math.max(...e)
        return { method: value, e, spread: (hi - lo) / ((hi + lo) / 2) }
      }),
    [],
  )
  const x = signal(t)
  const y = run(state.method, t)
  // Output m summarises input positions 2m and 2m + 1, so draw it at 2m + 0.5.
  const outX = y.map((_, m) => 2 * m + 0.5)
  const current = sweeps.find((s) => s.method === state.method)!

  const xAxis = useAxis({ label: 'position (input samples)', range: [-0.5, L - 0.5] })
  const yAxis = useAxis({ label: 'value', range: [-0.05, 1.1] })
  const sAxis = useAxis({ label: 'input shift t', range: [-0.5, SHIFTS - 0.5], integer: true })
  const eAxis = useAxis({ label: 'output energy Σ y²', hold: 'union' })
  const shifts = Array.from({ length: SHIFTS }, (_, s) => s)
  return (
    <Figure
      title="Downsampling and shifts"
      state={state}
      caption="Top: the input, a one-sample spike, a two-sample bar and a patch of fine texture, shifted right by t (bars), and the downsampled output at half the rate (points, drawn between the two input samples each output summarises). Bottom: the energy of the output for every shift, one curve per downsampler. Drag the guide line in the lower chart to shift the input. Keeping every second sample loses the spike at odd shifts, and max pooling changes the bar's width with the shift's parity: both energies swing between two values. Blurring before subsampling, as in anti-aliased pooling, keeps the energy almost constant."
      readouts={
        <>
          <Readout label="shift" value={t} />
          <Readout label="output energy" value={formatNumber(energy(y))} />
          {sweeps.map((s) => (
            <Readout
              key={s.method}
              label={`energy swing, ${METHODS.find((m) => m.value === s.method)!.label}`}
              value={`${formatNumber(100 * s.spread)} %`}
            />
          ))}
        </>
      }
    >
      <FixedHeight height={200}>
        <Plot x={xAxis} y={yAxis} ariaLabel="Shifted input and its downsampled output">
          <Bars name="input" x={x.map((_, n) => n)} y={x} width={0.8} muted />
          <Points name="output" x={outX} y={y} emphasis />
        </Plot>
      </FixedHeight>
      <FixedHeight height={200}>
        <div className="mt-3">
          <Plot x={sAxis} y={eAxis} ariaLabel="Output energy against shift">
            {sweeps.map((s, k) => (
              <Curve
                key={s.method}
                name={METHODS[k].label}
                x={shifts}
                y={s.e}
                slot={k}
                showPoints
                width={s.method === current.method ? 3 : 1.5}
              />
            ))}
            <Handle {...state.handle('shift', { label: 'shift t' })} />
          </Plot>
        </div>
      </FixedHeight>
    </Figure>
  )
}
