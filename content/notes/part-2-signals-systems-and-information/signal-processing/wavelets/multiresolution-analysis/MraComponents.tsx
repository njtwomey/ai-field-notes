import { useMemo } from 'react'
import {
  choice,
  Figure,
  formatNumber,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { FILTERS, wavedec, waverec, type Family } from '../_shared/wavelets'

type Shape = 'steps' | 'chirp'

const N = 512
const T = Array.from({ length: N }, (_, n) => n / N)

function makeSignal(shape: Shape): Float64Array {
  const x = new Float64Array(N)
  for (let n = 0; n < N; n++) {
    const t = T[n]
    x[n] =
      shape === 'steps'
        ? (t > 0.25 ? 1 : 0) - (t > 0.6 ? 1.5 : 0) + 0.8 * t + 0.25 * Math.sin(2 * Math.PI * 24 * t)
        : Math.sin(2 * Math.PI * (2 * t + 30 * t * t)) * (1 - 0.5 * t) + 0.5 * Math.cos(2 * Math.PI * 1.5 * t)
  }
  return x
}

/**
 * A J-level orthogonal wavelet decomposition splits a signal into a coarse approximation A_J and details D_1 … D_J,
 * each projected back to full length. They sum exactly to the signal, and their energies add.
 */
export function MraComponents() {
  const state = useFigureState({
    shape: choice<Shape>(
      [
        { value: 'steps', label: 'steps and a tone' },
        { value: 'chirp', label: 'chirp' },
      ],
      'steps',
      { label: 'signal' },
    ),
    family: choice<Family>(
      [
        { value: 'haar', label: 'Haar' },
        { value: 'db2', label: 'db2' },
        { value: 'db4', label: 'db4' },
      ],
      'db4',
      { label: 'wavelet' },
    ),
    levels: int(4, { min: 1, max: 5, step: 1, label: 'levels J', format: (v) => String(v) }),
  })

  const r = useMemo(() => {
    const x = makeSignal(state.shape)
    const h = FILTERS[state.family]
    const { approx, details } = wavedec(x, h, state.levels)
    const zero = (v: Float64Array) => new Float64Array(v.length)
    // Each component: reconstruct with every other coefficient set zeroed.
    const a = waverec(approx, details.map(zero), h)
    const ds = details.map((_, j) =>
      waverec(
        zero(approx),
        details.map((d, i) => (i === j ? d : zero(d))),
        h,
      ),
    )
    const energy = (v: ArrayLike<number>) => Array.from(v).reduce((s, u) => s + u * u, 0)
    const total = energy(x)
    const coeffEnergy = [energy(approx), ...details.map(energy)]
    const sum = Float64Array.from(x, (_, n) => a[n] + ds.reduce((s, d) => s + d[n], 0))
    const error = Math.max(...Array.from(x, (v, n) => Math.abs(v - sum[n])))
    return { x, a, ds, total, coeffEnergy, error }
  }, [state.shape, state.family, state.levels])

  // Stack the components with vertical offsets so each has its own band.
  const gap = 2.2
  const J = state.levels
  const series: SeriesSpec[] = [
    { name: 'signal', type: 'line', x: T, y: Array.from(r.x), slot: 0 },
    { name: `A${J}`, type: 'line', x: T, y: Array.from(r.a, (v) => v - gap), slot: 1 },
    ...r.ds
      .map((d, j) => ({ d, j }))
      .reverse()
      .map(({ d, j }, i): SeriesSpec => ({
        name: `D${j + 1}`,
        type: 'line',
        x: T,
        y: Array.from(d, (v) => v - gap * (i + 2)),
        slot: 2 + (J - 1 - j),
      })),
  ]

  const xAxis = useAxis({ label: 't', range: [0, 1] })
  const yAxis = useAxis({ label: 'components (offset)', hold: 'union' })
  return (
    <Figure
      title="A signal as approximation plus details"
      state={state}
      caption="Top: the signal. Below, offset for legibility: the level-J approximation A_J, then the details D_J (coarsest) down to D_1 (finest), each reconstructed to full length with every other coefficient zeroed. The components add up to the signal exactly, and, because the transform is orthogonal, the coefficient energies add up to the signal's energy. Steps and kinks put detail energy at every level near the discontinuity; smooth parts go to the approximation."

      readouts={
        <>
          <Readout
            label="energy share A, then D_J … D_1"
            value={[r.coeffEnergy[0], ...r.coeffEnergy.slice(1).reverse()]
              .map((e) => `${formatNumber((100 * e) / r.total)}%`)
              .join(', ')}
          />
          <Readout label="max reconstruction error" value={r.error.toExponential(1)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={420}>
        {seriesLayers(series)}
      </Plot>
    </Figure>
  )
}
