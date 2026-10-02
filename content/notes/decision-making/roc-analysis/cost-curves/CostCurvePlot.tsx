import { useMemo } from 'react'
import {
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type Segment,
} from 'aifn-render'
import { linspace } from '@/lib/math'
import { rocExample } from '../../_shared/rocExample'

const PC = linspace(0, 1, 201)

/**
 * Drummond and Holte's cost curves for the thresholds of one scoring classifier. Each ROC point (FPR, TPR) becomes the
 * line NE = FNR·PC + FPR·(1 − PC) from (0, FPR) to (1, FNR). The lower envelope of the lines is the classifier's cost
 * curve, and corresponds to the ROC convex hull.
 */
export function CostCurvePlot() {
  const pc = useParam(0.5, { min: 0, max: 1, step: 0.005 })
  const data = useMemo(() => rocExample(), [])

  const lines = data.curve.map(([fpr, tpr]) => ({ fpr, fnr: 1 - tpr }))
  const segments: Segment[] = lines.map((l) => ({ from: [0, l.fpr], to: [1, l.fnr] }))
  const envelope = useMemo(() => PC.map((x) => Math.min(...lines.map((l) => l.fnr * x + l.fpr * (1 - x)))), [lines])
  const auc = useMemo(
    () => envelope.reduce((acc, v, i) => (i ? acc + ((v + envelope[i - 1]) / 2) * (PC[i] - PC[i - 1]) : 0), 0),
    [envelope],
  )

  const best = lines.reduce((a, b) =>
    b.fnr * pc.value + b.fpr * (1 - pc.value) < a.fnr * pc.value + a.fpr * (1 - pc.value) ? b : a,
  )
  const cost = best.fnr * pc.value + best.fpr * (1 - pc.value)
  const handles: Handle[] = [{ kind: 'x', at: pc.value, label: 'operating point PC(+)', onDrag: (x) => pc.set(x) }]

  return (
    <Interactive
      title="Cost curves of one scoring classifier"
      caption="Each thin line is one threshold of the classifier from the ROC convex hull figure: its normalised expected cost as the probability cost PC(+) runs from 0 to 1. The thick line is their lower envelope, the best achievable cost at each operating point; it is the dual of the ROC convex hull. The dashed lines are the trivial classifiers. Drag the vertical line to an operating point to read off the best threshold and its cost."
      controls={<ParamSlider label="probability cost PC(+)" param={pc} />}
      readout={
        <>
          <Readout
            label="best threshold (FPR, TPR)"
            value={`(${formatNumber(best.fpr)}, ${formatNumber(1 - best.fnr)})`}
          />
          <Readout label="normalised expected cost" value={formatNumber(cost)} />
          <Readout label="trivial classifier" value={formatNumber(Math.min(pc.value, 1 - pc.value))} />
          <Readout label="area under the envelope" value={formatNumber(auc)} />
        </>
      }
    >
      <XYChart
        height={340}
        xLabel="probability cost PC(+)"
        yLabel="normalised expected cost"
        xRange={[0, 1]}
        yRange={[0, 1]}
        handles={handles}
        segments={segments}
        series={[
          { name: 'always negative', type: 'line', x: [0, 1], y: [0, 1], dashed: true, muted: true },
          { name: 'always positive', type: 'line', x: [0, 1], y: [1, 0], dashed: true, muted: true },
          { name: 'best threshold here', type: 'line', x: [0, 1], y: [best.fpr, best.fnr], slot: 1 },
          { name: 'lower envelope', type: 'line', x: PC, y: envelope, emphasis: true },
        ]}
      />
    </Interactive>
  )
}
