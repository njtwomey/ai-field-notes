import { useMemo } from 'react'
import { Curve, Figure, formatNumber, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { rocExample } from '../../_shared/rocExample'
import { linspace, toFlat } from 'aifn/foundation/tensor'

/**
 * The same classifier in precision-recall space and in precision-recall-gain space, at a chosen prevalence π. The
 * rates come from the shared ROC example; precision is recomputed for π. The F1-optimal point and the F1 isometric
 * through it are drawn in both: a hyperbola in PR space, a straight line of slope −1 in PRG space.
 */
export function PrgCurves() {
  const state = useFigureState({
    pi: slider(0.02, 0.8, 0.2, { step: 0.01, label: 'prevalence π' }),
  })
  const data = useMemo(() => rocExample(), [])

  const r = useMemo(() => {
    const p = state.pi
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
    const rs = toFlat(linspace(best.f1 / 2 + 1e-3, 1, 120))
    const iso = { x: rs, y: rs.map((x) => (best.f1 * x) / (2 * x - best.f1)) }
    return { pts, best, inGain, fg, iso }
  }, [data, state.pi])

  const xAxis = useAxis({ label: 'recall', range: [0, 1] })
  const yAxis = useAxis({ label: 'precision', range: [0, 1] })
  const xAxis2 = useAxis({ label: 'recall gain', range: [0, 1] })
  const yAxis2 = useAxis({ label: 'precision gain', range: [0, 1] })
  return (
    <Figure
      title="Precision-recall and precision-recall-gain curves"
      state={state}
      caption="One classifier's thresholds, drawn in PR space (left) and PRG space (right) at prevalence π. The ringed point has the highest F1. The line through it is its F1 isometric: a hyperbola in PR space, but in PRG space a straight line of slope −1, so the F1-optimal point is where a line of slope −1 touches the curve. Change π: the PR baseline (precision = π) moves, while in PRG space the always-positive classifier stays at (1, 0)."

      readouts={
        <>
          <Readout label="best F1" value={formatNumber(r.best.f1)} />
          <Readout label="its F1 gain" value={formatNumber(r.fg)} />
          <Readout label="(recall gain + precision gain) / 2" value={formatNumber((r.best.recG + r.best.precG) / 2)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={300}>
          <Curve name="baseline precision = π" x={[0, 1]} y={[state.pi, state.pi]} dashed muted />
          <Curve name="PR curve" x={r.pts.map((q) => q.rec)} y={r.pts.map((q) => q.prec)} slot={0} />
          <Curve name="F1 isometric" x={r.iso.x} y={r.iso.y} slot={1} dashed />
          <Points name="best F1" x={[r.best.rec]} y={[r.best.prec]} emphasis />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={300}>
          <Curve name="baseline F1 = π" x={[0, 1]} y={[1, 0]} dashed muted />
          <Curve name="PRG curve" x={r.inGain.map((q) => q.recG)} y={r.inGain.map((q) => q.precG)} slot={0} />
          <Curve
            name="F1 isometric"
            x={[Math.max(0, 2 * r.fg - 1), Math.min(1, 2 * r.fg)]}
            y={[Math.min(1, 2 * r.fg - Math.max(0, 2 * r.fg - 1)), 2 * r.fg - Math.min(1, 2 * r.fg)]}
            slot={1}
            dashed
          />
          <Points name="best F1" x={[r.best.recG]} y={[r.best.precG]} emphasis />
        </Plot>
      </div>
    </Figure>
  )
}
