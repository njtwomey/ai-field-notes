import { useMemo } from 'react'
import {
  Area,
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { normalCdf, normalPdf, normalQuantile } from 'aifn-compute/numerics/special'

type Dist = 'gaussian' | 'lognormal'

const X_MIN = 0
const X_MAX = 45
const Y_GRID = toFlat(linspace(X_MIN, X_MAX, 901))
const Q_GRID = toFlat(linspace(2, 26, 121))
const DY = Y_GRID[1] - Y_GRID[0]

/** Two outcome distributions with median 10: Gaussian with sd 2, and log-normal with log-scale sd 0.4 (right-skewed). */
const DISTS: Record<Dist, { pdf: (y: number) => number; cdf: (y: number) => number; quantile: (p: number) => number }> =
  {
    gaussian: {
      pdf: (y) => normalPdf((y - 10) / 2) / 2,
      cdf: (y) => normalCdf((y - 10) / 2),
      quantile: (p) => 10 + 2 * normalQuantile(p),
    },
    lognormal: {
      pdf: (y) => (y <= 0 ? 0 : normalPdf((Math.log(y) - Math.log(10)) / 0.4) / (0.4 * y)),
      cdf: (y) => (y <= 0 ? 0 : normalCdf((Math.log(y) - Math.log(10)) / 0.4)),
      quantile: (p) => 10 * Math.exp(0.4 * normalQuantile(p)),
    },
  }

const pinball = (u: number, tau: number) => (u >= 0 ? tau * u : (tau - 1) * u)

/** Expected pinball loss E[rho_tau(Y - q)], by numerical integration of a density given on Y_GRID. */
function expectedLoss(density: number[], tau: number, q: number) {
  let s = 0
  for (let i = 0; i < Y_GRID.length; i++) s += pinball(Y_GRID[i] - q, tau) * density[i]
  return s * DY
}

export function PinballExplorer() {
  const state = useFigureState({
    tau: slider(0.05, 0.95, 0.9, { step: 0.05, label: 'quantile level τ', format: (v) => v.toFixed(2) }),
    q: float(10, { min: 2, max: 26, step: 0.1, label: 'forecast q', format: (v) => v.toFixed(1) }),
    dist: choice<Dist>(
      [
        { value: 'gaussian', label: 'Gaussian' },
        { value: 'lognormal', label: 'log-normal' },
      ],
      'gaussian',
      { label: 'outcome distribution' },
    ),
  })

  const d = DISTS[state.dist]
  const density = useMemo(() => Y_GRID.map((y) => DISTS[state.dist].pdf(y)), [state.dist])
  const curve = useMemo(() => Q_GRID.map((qq) => expectedLoss(density, state.tau, qq)), [density, state.tau])
  const best = d.quantile(state.tau)
  const lossAtQ = expectedLoss(density, state.tau, state.q)
  const lossAtBest = expectedLoss(density, state.tau, best)

  const lossSeries = useMemo(() => {
    const qStar = DISTS[state.dist].quantile(state.tau)
    return [
      { name: 'expected pinball loss', x: Q_GRID, y: curve, slot: 0 },
      {
        name: 'minimum: the τ-quantile',
        x: [qStar],
        y: [expectedLoss(density, state.tau, qStar)],
        emphasis: true,
      },
    ] as const
  }, [curve, density, state.dist, state.tau])
  const densitySeries = useMemo(() => {
    const below = Y_GRID.filter((y) => y <= state.q)
    return [
      { name: 'density of Y', x: Y_GRID, y: density, slot: 1 },
      { name: 'P(Y ≤ q)', x: below, y: below.map((y) => DISTS[state.dist].pdf(y)), slot: 1 },
    ] as const
  }, [density, state.q, state.dist])

  const xAxis = useAxis({ label: 'forecast q', range: [2, 26] })
  const yAxis = useAxis({ label: 'expected loss', range: [0, undefined], hold: 'union' })
  const xAxis2 = useAxis({ label: 'y', range: [2, 26] })
  const yAxis2 = useAxis({ label: 'density', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="The pinball loss is minimised at the quantile"
      state={state}
      caption="The outcome Y has the distribution in the lower panel (both choices have median 10). The upper panel shows the expected pinball loss of a forecast q at level τ. Drag the vertical line or use the slider to move q. The loss is lowest where the shaded probability below q equals τ, that is, at the τ-quantile of Y. For the skewed distribution the quantiles above the median lie further out than below it."

      readouts={
        <>
          <Readout label="P(Y ≤ q)" value={formatNumber(d.cdf(state.q))} />
          <Readout label="expected loss at q" value={formatNumber(lossAtQ)} />
          <Readout label="τ-quantile" value={formatNumber(best)} />
          <Readout label="expected loss at the quantile" value={formatNumber(lossAtBest)} />
        </>
      }
    >
      <div className="space-y-4">
        <Plot x={xAxis} y={yAxis} height={220}>
          <Curve {...lossSeries[0]} />
          <Points {...lossSeries[1]} />
          <Handle {...state.handle('q', { label: 'q' })} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={160}>
          <Curve {...densitySeries[0]} />
          <Area {...densitySeries[1]} />
        </Plot>
      </div>
    </Figure>
  )
}
