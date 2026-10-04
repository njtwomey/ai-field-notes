import { useMemo } from 'react'
import {
  Curve,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Readout,
  type Segment,
  Segments,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { rocExample } from '../../_shared/rocExample'
import { linspace, toFlat } from 'aifn/foundation/tensor'

const PC = toFlat(linspace(0, 1, 201))

/**
 * Drummond and Holte's cost curves for the thresholds of one scoring classifier. Each ROC point (FPR, TPR) becomes the
 * line NE = FNR·PC + FPR·(1 − PC) from (0, FPR) to (1, FNR). The lower envelope of the lines is the classifier's cost
 * curve, and corresponds to the ROC convex hull.
 */
export function CostCurvePlot() {
  const state = useFigureState({
    pc: slider(0, 1, 0.5, { step: 0.005, label: 'probability cost PC(+)' }),
  })
  const data = useMemo(() => rocExample(), [])

  const lines = data.curve.map(([fpr, tpr]) => ({ fpr, fnr: 1 - tpr }))
  const segments: Segment[] = lines.map((l) => ({ from: [0, l.fpr], to: [1, l.fnr] }))
  const envelope = useMemo(() => PC.map((x) => Math.min(...lines.map((l) => l.fnr * x + l.fpr * (1 - x)))), [lines])
  const auc = useMemo(
    () => envelope.reduce((acc, v, i) => (i ? acc + ((v + envelope[i - 1]) / 2) * (PC[i] - PC[i - 1]) : 0), 0),
    [envelope],
  )

  const best = lines.reduce((a, b) =>
    b.fnr * state.pc + b.fpr * (1 - state.pc) < a.fnr * state.pc + a.fpr * (1 - state.pc) ? b : a,
  )
  const cost = best.fnr * state.pc + best.fpr * (1 - state.pc)

  const xAxis = useAxis({ label: 'probability cost PC(+)', range: [0, 1] })
  const yAxis = useAxis({ label: 'normalised expected cost', range: [0, 1] })
  return (
    <Figure
      title="Cost curves of one scoring classifier"
      state={state}
      caption="Each thin line is one threshold of the classifier from the ROC convex hull figure: its normalised expected cost as the probability cost PC(+) runs from 0 to 1. The thick line is their lower envelope, the best achievable cost at each operating point; it is the dual of the ROC convex hull. The dashed lines are the trivial classifiers. Drag the vertical line to an operating point to read off the best threshold and its cost."

      readouts={
        <>
          <Readout
            label="best threshold (FPR, TPR)"
            value={`(${formatNumber(best.fpr)}, ${formatNumber(1 - best.fnr)})`}
          />
          <Readout label="normalised expected cost" value={formatNumber(cost)} />
          <Readout label="trivial classifier" value={formatNumber(Math.min(state.pc, 1 - state.pc))} />
          <Readout label="area under the envelope" value={formatNumber(auc)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={340}>
        <Curve name="always negative" x={[0, 1]} y={[0, 1]} dashed muted />
        <Curve name="always positive" x={[0, 1]} y={[1, 0]} dashed muted />
        <Curve name="best threshold here" x={[0, 1]} y={[best.fpr, best.fnr]} slot={1} />
        <Curve name="lower envelope" x={PC} y={envelope} emphasis />
        <Segments segments={segments} />
        <Handle {...state.handle('pc', { label: 'operating point PC(+)' })} />
      </Plot>
    </Figure>
  )
}
