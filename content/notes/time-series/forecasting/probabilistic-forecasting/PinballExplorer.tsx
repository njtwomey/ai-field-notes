import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from '@/components/viz'
import { linspace } from '@/lib/math'
import { normalCdf, normalPdf, normalQuantile } from '@/lib/math/special'

type Dist = 'gaussian' | 'lognormal'

const X_MIN = 0
const X_MAX = 45
const Y_GRID = linspace(X_MIN, X_MAX, 901)
const Q_GRID = linspace(2, 26, 121)
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
  const tau = useParam(0.9, { min: 0.05, max: 0.95, step: 0.05 })
  const q = useParam(10, { min: 2, max: 26, step: 0.1 })
  const [dist, setDist] = useState<Dist>('gaussian')

  const d = DISTS[dist]
  const density = useMemo(() => Y_GRID.map((y) => DISTS[dist].pdf(y)), [dist])
  const curve = useMemo(() => Q_GRID.map((qq) => expectedLoss(density, tau.value, qq)), [density, tau.value])
  const best = d.quantile(tau.value)
  const lossAtQ = expectedLoss(density, tau.value, q.value)
  const lossAtBest = expectedLoss(density, tau.value, best)

  const lossSeries = useMemo((): XYSeries[] => {
    const qStar = DISTS[dist].quantile(tau.value)
    return [
      { name: 'expected pinball loss', type: 'line', x: Q_GRID, y: curve, slot: 0 },
      {
        name: 'minimum: the τ-quantile',
        type: 'scatter',
        x: [qStar],
        y: [expectedLoss(density, tau.value, qStar)],
        emphasis: true,
      },
    ]
  }, [curve, density, dist, tau.value])
  const densitySeries = useMemo((): XYSeries[] => {
    const below = Y_GRID.filter((y) => y <= q.value)
    return [
      { name: 'density of Y', type: 'line', x: Y_GRID, y: density, slot: 1 },
      { name: 'P(Y ≤ q)', type: 'line', x: below, y: below.map((y) => DISTS[dist].pdf(y)), slot: 1, area: true },
    ]
  }, [density, q.value, dist])

  return (
    <Interactive
      title="The pinball loss is minimised at the quantile"
      caption="The outcome Y has the distribution in the lower panel (both choices have median 10). The upper panel shows the expected pinball loss of a forecast q at level τ. Drag the vertical line or use the slider to move q. The loss is lowest where the shaded probability below q equals τ, that is, at the τ-quantile of Y. For the skewed distribution the quantiles above the median lie further out than below it."
      controls={
        <>
          <ParamSlider label="quantile level τ" param={tau} format={(v) => v.toFixed(2)} />
          <ParamSlider label="forecast q" param={q} format={(v) => v.toFixed(1)} />
          <ParamChoice
            label="outcome distribution"
            value={dist}
            onChange={setDist}
            options={[
              { value: 'gaussian', label: 'Gaussian' },
              { value: 'lognormal', label: 'log-normal' },
            ]}
          />
        </>
      }
      readout={
        <>
          <Readout label="P(Y ≤ q)" value={formatNumber(d.cdf(q.value))} />
          <Readout label="expected loss at q" value={formatNumber(lossAtQ)} />
          <Readout label="τ-quantile" value={formatNumber(best)} />
          <Readout label="expected loss at the quantile" value={formatNumber(lossAtBest)} />
        </>
      }
    >
      <div className="space-y-4">
        <XYChart
          series={lossSeries}
          xLabel="forecast q"
          yLabel="expected loss"
          xRange={[2, 26]}
          yRange={[0, undefined]}
          height={220}
          handles={[{ kind: 'x', at: q.value, onDrag: q.set, label: 'q' }]}
        />
        <XYChart
          series={densitySeries}
          xLabel="y"
          yLabel="density"
          xRange={[2, 26]}
          yRange={[0, undefined]}
          height={160}
        />
      </div>
    </Interactive>
  )
}
