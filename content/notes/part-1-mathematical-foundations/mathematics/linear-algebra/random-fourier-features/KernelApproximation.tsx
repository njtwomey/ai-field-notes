import { useMemo } from 'react'
import {
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { normal, stream, uniform } from 'aifn-compute/foundation/random'

const GRID = toFlat(linspace(-4, 4, 201))
/** Capped because each draw costs D × 201 cosines, recomputed while D is dragged. */
const MAX_DRAWS = 20

/** k(0, x) for the Gaussian kernel against its estimate z(0)ᵀz(x) from D random Fourier features. */
export function KernelApproximation() {
  const state = useFigureState({
    logD: float(1.5, {
      min: 0,
      max: 3.5,
      step: 0.1,
      label: 'number of features D (log scale)',
      points_per_decade: 2,
      logTransform: 'value-is-log',
      format: (v) => String(Math.round(10 ** v)),
    }),
    width: float(1, { min: 0.3, max: 2, step: 0.05, label: 'kernel width ℓ' }),
    count: int(5, { min: 1, max: MAX_DRAWS, step: 1, label: 'draws', format: (v) => String(v) }),
    seed: int(1, { min: 1, max: 20, step: 1, label: 'seed' }),
  })
  const D = Math.round(10 ** state.logD)

  const r = useMemo(() => {
    const exact = GRID.map((x) => Math.exp(-(x * x) / (2 * state.width ** 2)))
    // Draw k uses its own stream, so raising the count adds curves without changing the earlier ones.
    const approxes = Array.from({ length: state.count }, (_, k) => {
      const g = stream(state.seed * 1000 + k)
      // Frequencies from the kernel's spectral density, N(0, 1/ℓ²), and uniform phases.
      const w = Array.from({ length: D }, () => normal(g) / state.width)
      const b = Array.from({ length: D }, () => 2 * Math.PI * uniform(g))
      const scale = 2 / D
      const z0 = b.map((bi) => Math.cos(bi))
      return GRID.map((x) => scale * w.reduce((s, wi, i) => s + z0[i] * Math.cos(wi * x + b[i]), 0))
    })
    const maxErr =
      approxes.reduce((acc, a) => acc + Math.max(...a.map((v, i) => Math.abs(v - exact[i]))), 0) / approxes.length
    const many = approxes.length > 1
    const series: SeriesSpec[] = [
      ...approxes.map((y): SeriesSpec => ({
        name: many ? 'random-feature estimates z(0)ᵀz(x)' : 'random-feature estimate z(0)ᵀz(x)',
        type: 'line',
        x: GRID,
        y,
        slot: 1,
        thin: many,
      })),
      { name: 'exact k(0, x)', type: 'line', x: GRID, y: exact, slot: 0 },
    ]
    return { series, maxErr }
  }, [D, state.width, state.seed, state.count])

  const xAxis = useAxis({ label: 'x', range: [-4, 4] })
  const yAxis = useAxis({ label: 'k(0, x)', range: [-0.6, 1.4] })
  return (
    <Figure
      title="Approximating a Gaussian kernel with random features"
      state={state}
      caption="The exact curve is the Gaussian kernel k(0, x). Each light curve is one estimate z(0)ᵀz(x), built from its own independent draw of D random cosine features; the draws slider sets how many. The estimates scatter evenly about the exact curve because each is unbiased, and their spread shrinks like 1/√D: a tenfold increase in D cuts the error by about a factor of three. Change the seed to see a different set of draws."

      readouts={
        <>
          <Readout label="largest error on the plot (mean over draws)" value={formatNumber(r.maxErr)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={320}>
        {seriesLayers(r.series)}
      </Plot>
    </Figure>
  )
}
