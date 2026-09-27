import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam } from '@/components/viz'
import { linspace } from '@/lib/math'
import { rocExample } from '../../_shared/rocExample'

/**
 * The same classifier in precision-recall space and in precision-recall-gain space, at a chosen prevalence π. The
 * rates come from the shared ROC example; precision is recomputed for π. The F1-optimal point and the F1 isometric
 * through it are drawn in both: a hyperbola in PR space, a straight line of slope −1 in PRG space.
 */
export function PrgCurves() {
  const pi = useParam(0.2, { min: 0.02, max: 0.8, step: 0.01 })
  const data = useMemo(() => rocExample(), [])

  const r = useMemo(() => {
    const p = pi.value
    const pts = data.curve
      .filter(([, tpr]) => tpr > 0)
      .map(([fpr, tpr]) => {
        const prec = (p * tpr) / (p * tpr + (1 - p) * fpr)
        const f1 = (2 * prec * tpr) / (prec + tpr)
        return {
          rec: tpr,
          prec,
          f1,
          recG: (tpr - p) / ((1 - p) * tpr),
          precG: (prec - p) / ((1 - p) * prec),
        }
      })
    const best = pts.reduce((a, b) => (b.f1 > a.f1 ? b : a))
    const inGain = pts.filter((q) => q.recG >= 0 && q.precG >= 0)
    const fg = (best.f1 - p) / ((1 - p) * best.f1)
    // F1 isometric in PR space: P = F R / (2R − F), for R > F/2.
    const rs = linspace(best.f1 / 2 + 1e-3, 1, 120)
    const iso = { x: rs, y: rs.map((x) => (best.f1 * x) / (2 * x - best.f1)) }
    return { pts, best, inGain, fg, iso }
  }, [data, pi.value])

  return (
    <Interactive
      title="Precision-recall and precision-recall-gain curves"
      caption="One classifier's thresholds, drawn in PR space (left) and PRG space (right) at prevalence π. The ringed point has the highest F1. The line through it is its F1 isometric: a hyperbola in PR space, but in PRG space a straight line of slope −1, so the F1-optimal point is where a line of slope −1 touches the curve. Change π: the PR baseline (precision = π) moves, while in PRG space the always-positive classifier stays at (1, 0)."
      controls={<ParamSlider label="prevalence π" param={pi} />}
      readout={
        <>
          <Readout label="best F1" value={formatNumber(r.best.f1)} />
          <Readout label="its F1 gain" value={formatNumber(r.fg)} />
          <Readout label="(recall gain + precision gain) / 2" value={formatNumber((r.best.recG + r.best.precG) / 2)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <XYChart
          height={300}
          xLabel="recall"
          yLabel="precision"
          xRange={[0, 1]}
          yRange={[0, 1]}
          series={[
            {
              name: 'baseline precision = π',
              type: 'line',
              x: [0, 1],
              y: [pi.value, pi.value],
              dashed: true,
              muted: true,
            },
            { name: 'PR curve', type: 'line', x: r.pts.map((q) => q.rec), y: r.pts.map((q) => q.prec), slot: 0 },
            { name: 'F1 isometric', type: 'line', x: r.iso.x, y: r.iso.y, slot: 1, dashed: true },
            { name: 'best F1', type: 'scatter', x: [r.best.rec], y: [r.best.prec], emphasis: true },
          ]}
        />
        <XYChart
          height={300}
          xLabel="recall gain"
          yLabel="precision gain"
          xRange={[0, 1]}
          yRange={[0, 1]}
          series={[
            { name: 'baseline F1 = π', type: 'line', x: [0, 1], y: [1, 0], dashed: true, muted: true },
            {
              name: 'PRG curve',
              type: 'line',
              x: r.inGain.map((q) => q.recG),
              y: r.inGain.map((q) => q.precG),
              slot: 0,
            },
            {
              name: 'F1 isometric',
              type: 'line',
              x: [Math.max(0, 2 * r.fg - 1), Math.min(1, 2 * r.fg)],
              y: [Math.min(1, 2 * r.fg - Math.max(0, 2 * r.fg - 1)), 2 * r.fg - Math.min(1, 2 * r.fg)],
              slot: 1,
              dashed: true,
            },
            { name: 'best F1', type: 'scatter', x: [r.best.recG], y: [r.best.precG], emphasis: true },
          ]}
        />
      </div>
    </Interactive>
  )
}
