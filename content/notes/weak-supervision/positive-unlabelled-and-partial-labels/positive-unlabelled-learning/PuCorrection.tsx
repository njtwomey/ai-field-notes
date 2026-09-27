import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'
import { linspace } from '@/lib/math'
import { normalPdf } from '@/lib/math/special'

const X_RANGE: [number, number] = [-5, 5]
const GRID = linspace(-5, 5, 401)

/** First x on the grid where a curve crosses 1/2 from below, or null if it never does. */
function crossing(ys: number[]): number | null {
  for (let i = 1; i < ys.length; i++) if (ys[i - 1] < 0.5 && ys[i] >= 0.5) return GRID[i]
  return null
}

/**
 * Elkan and Noto's correction in one dimension. Positives are N(Δ/2, 1), negatives N(−Δ/2, 1). A fraction c of the
 * positives is labelled at random (SCAR). A classifier of labelled against unlabelled learns g(x) = c·f(x); dividing by
 * c recovers f. The estimate e₁ = E[g(x) | s = 1] equals c only when the classes do not overlap.
 */
export function PuCorrection() {
  const prior = useParam(0.4, { min: 0.1, max: 0.9, step: 0.05 })
  const c = useParam(0.3, { min: 0.05, max: 1, step: 0.05 })
  const sep = useParam(3, { min: 0.5, max: 6, step: 0.25 })
  const pi = prior.value
  const cv = c.value
  const d = sep.value

  const r = useMemo(() => {
    const pPos = GRID.map((x) => normalPdf(x - d / 2))
    const pNeg = GRID.map((x) => normalPdf(x + d / 2))
    const f = GRID.map((_, i) => (pi * pPos[i]) / (pi * pPos[i] + (1 - pi) * pNeg[i]))
    const g = f.map((v) => cv * v)
    // e₁ = E[g(x) | s = 1]; under SCAR the labelled points are distributed as the positives.
    const dx = GRID[1] - GRID[0]
    const e1 = GRID.reduce((s, _, i) => s + g[i] * pPos[i] * dx, 0)
    const corrected = g.map((v) => Math.min(1, v / e1))
    return { f, g, corrected, e1 }
  }, [pi, cv, d])

  const series: XYSeries[] = [
    { name: 'P(y = 1 | x), true', type: 'line', x: GRID, y: r.f, slot: 0 },
    { name: 'g(x) = P(s = 1 | x), labelled vs unlabelled', type: 'line', x: GRID, y: r.g, slot: 1 },
    { name: 'g(x) / e₁, Elkan–Noto estimate', type: 'line', x: GRID, y: r.corrected, slot: 2, dashed: true },
    { name: 'threshold 1/2', type: 'line', x: [...X_RANGE], y: [0.5, 0.5], muted: true, dashed: true },
  ]
  const bayes = crossing(r.f)
  const naive = crossing(r.g)
  const fixed = crossing(r.corrected)
  const fmt = (v: number | null) => (v === null ? 'none' : formatNumber(v))

  return (
    <Interactive
      title="Correcting a labelled-versus-unlabelled classifier"
      caption="Positives are N(Δ/2, 1) and negatives N(−Δ/2, 1), with class prior π. Only a fraction c of the positives carries a label, chosen at random. A classifier trained to separate labelled from unlabelled points learns g(x) = c·P(y = 1 | x): the same shape as the true posterior, scaled down by c. Thresholding g at one half misses most positives, or all of them when c < 1/2. Dividing by c restores the posterior. Elkan and Noto estimate c by e₁, the average of g over labelled points; e₁ equals c only when the classes do not overlap, so with overlap it is too small and the corrected curve overshoots."
      controls={
        <>
          <ParamSlider label="class prior π" param={prior} />
          <ParamSlider label="label frequency c" param={c} />
          <ParamSlider label="class separation Δ" param={sep} />
        </>
      }
      readout={
        <>
          <Readout label="e₁ (estimate of c)" value={formatNumber(r.e1)} />
          <Readout label="true boundary" value={fmt(bayes)} />
          <Readout label="naive boundary, g = 1/2" value={fmt(naive)} />
          <Readout label="corrected boundary" value={fmt(fixed)} />
        </>
      }
    >
      <XYChart series={series} xLabel="x" yLabel="probability" xRange={X_RANGE} yRange={[0, 1.02]} />
    </Interactive>
  )
}
