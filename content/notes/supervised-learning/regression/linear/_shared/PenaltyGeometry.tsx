import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import { boundary, contour, penalty, quadratic, solve, type Vec } from './geometry'

const RANGE: [number, number] = [-2.5, 2.5]
const clamp = (v: number) => Math.min(Math.max(v, RANGE[0] + 0.1), RANGE[1] - 0.1)

const line = (name: string, points: Vec[], extra: Partial<XYSeries> = {}): XYSeries => ({
  name,
  type: 'line',
  x: points.map((p) => p[0]),
  y: points.map((p) => p[1]),
  ...extra,
})

/**
 * Where least-squares contours first touch a penalty ball, for two coefficients. In `compare` mode the ball is the
 * lasso diamond or the ridge disc; in `elastic` mode an l1-ratio slider morphs the elastic-net ball between them.
 */
export function PenaltyGeometry({ mode }: { mode: 'compare' | 'elastic' }) {
  const [centre, setCentre] = useState<Vec>([2.2, 1.0])
  const correlation = useParam(0.7, { min: -0.9, max: 0.9, step: 0.05 })
  const ratio = useParam(0.5, { min: 0, max: 1, step: 0.05 })
  const [shape, setShape] = useState<'l1' | 'l2'>('l1')
  const r = mode === 'elastic' ? ratio.value : shape === 'l1' ? 1 : 0
  const rho = correlation.value

  const result = useMemo(() => {
    const { w, inside } = solve(centre, rho, r)
    const q = quadratic(w, centre, rho)
    const series: XYSeries[] = []
    if (mode === 'elastic') {
      series.push(line('ℓ₁ diamond', boundary(1, 8), { muted: true, dashed: true }))
      series.push(line('ℓ₂ disc', boundary(0, 240), { muted: true, dashed: true }))
    }
    series.push(
      line(mode === 'elastic' ? 'elastic-net ball' : r === 1 ? 'ℓ₁ ball (lasso)' : 'ℓ₂ ball (ridge)', boundary(r), {
        slot: 0,
      }),
    )
    if (!inside) {
      series.push(line('contour touching the ball', contour(centre, rho, q), { slot: 1 }))
      series.push(line('wider contour', contour(centre, rho, q * 2.25), { slot: 1, dashed: true }))
    }
    series.push({ name: 'solution', type: 'scatter', x: [w[0]], y: [w[1]], emphasis: true })
    const zero = !inside && (w[0] === 0 || w[1] === 0)
    return { w, inside, series, zero }
  }, [centre, rho, r, mode])

  const handles: Handle[] = [
    { kind: 'point', at: centre, label: 'OLS', onDrag: ([x, y]) => setCentre([clamp(x), clamp(y)]) },
  ]
  const { w } = result
  const l1 = Math.abs(w[0]) + Math.abs(w[1])
  const l2sq = w[0] * w[0] + w[1] * w[1]

  return (
    <Interactive
      title={mode === 'elastic' ? 'The elastic-net ball' : 'Why the lasso sets coefficients to zero'}
      caption={
        mode === 'elastic'
          ? 'The ball r‖w‖₁ + (1 − r)‖w‖₂² ≤ 1 for two standardised features with correlation ρ. At r = 1 it is the lasso diamond and at r = 0 the ridge disc; in between it keeps corners on the axes but its edges curve outwards. Drag the OLS point along the direction w₁ = −w₂, the direction in which it is least certain when the features are correlated. The lasso solution runs along the diamond’s edge into a corner and drops a feature; the elastic-net solution moves far less and keeps weight on both.'
          : 'The round handle is the least-squares solution; the ellipses around it are contours of the squared error, stretched when the features are correlated. The penalised solution is the point where the smallest contour touches the ball of allowed coefficients. The ℓ₁ ball is a diamond with corners on the axes, and a whole region of OLS positions first touches it at a corner, where one coefficient is exactly zero. The ℓ₂ disc has no corners, so its contact point almost never lies on an axis.'
      }
      controls={
        <>
          {mode === 'elastic' ? (
            <ParamSlider label="l1 ratio r" param={ratio} />
          ) : (
            <ParamChoice
              label="penalty"
              value={shape}
              onChange={setShape}
              options={[
                { value: 'l1', label: 'lasso ℓ₁' },
                { value: 'l2', label: 'ridge ℓ₂' },
              ]}
            />
          )}
          <ParamSlider label="feature correlation ρ" param={correlation} />
        </>
      }
      readout={
        <>
          <Readout label="solution w" value={`(${formatNumber(w[0])}, ${formatNumber(w[1])})`} />
          <Readout label="‖w‖₁" value={formatNumber(l1)} />
          <Readout label="‖w‖₂²" value={formatNumber(l2sq)} />
          <Readout label="penalty r‖w‖₁ + (1 − r)‖w‖₂²" value={formatNumber(penalty(w, r))} />
          <Readout
            label="zero coefficient"
            value={
              result.inside ? 'no (OLS inside the ball)' : result.zero ? (w[0] === 0 ? 'w₁ = 0' : 'w₂ = 0') : 'none'
            }
          />
        </>
      }
    >
      <div className="mx-auto w-full max-w-lg">
        <XYChart
          series={result.series}
          xRange={RANGE}
          yRange={RANGE}
          equalAspect
          handles={handles}
          xLabel="w₁"
          yLabel="w₂"
        />
      </div>
    </Interactive>
  )
}
