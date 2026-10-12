import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function VideoAnnotationVisualizer() {
  const state = useFigureState({
    annotationMode: choice<'interpolation_cost' | 'tiou_localization' | 'hota_tracking'>(
      [
        {
          value: 'interpolation_cost',
          label: 'Keyframe Interpolation: Human Label Budget vs Boundary Drift (Caba 2015)',
        },
        {
          value: 'tiou_localization',
          label: 'Temporal Action Localization: Boundary Jitter vs tIoU Thresholds (Gu 2018)',
        },
        {
          value: 'hota_tracking',
          label: 'Object Tracking Evaluation: HOTA Detection vs Association Balance (Luiten 2021)',
        },
      ],
      'interpolation_cost',
      { label: 'video annotation perspective' },
    ),
    param1: slider(2, 30, 10, {
      step: 2,
      label: 'keyframe interval / temporal boundary jitter / identity switches',
    }),
    param2: slider(1, 10, 5, {
      step: 1,
      label: 'non-linear curvature / proposal duration / recall parameter',
    }),
  })

  const { annotationMode, param1, param2 } = state

  const results = useMemo(() => {
    if (annotationMode === 'interpolation_cost') {
      // Keyframe intervals from 2 to 30 frames
      const intervals = [2, 4, 6, 8, 10, 12, 16, 20, 24, 30]
      // Manual annotator time cost (% of full dense annotation)
      const annotatorCostPct = intervals.map((k) => (1 / k) * 100)
      // Mean bounding box IoU quality degradation due to linear interpolation
      // non-linear trajectory curvature parameter
      const curveFactor = param2 * 0.15
      const interpolationIoU = intervals.map((k) => {
        const drop = Math.pow(k / 10, 1.8) * 8.5 * curveFactor
        return Math.max(25, 96 - drop)
      })

      const currK = param1
      const currCost = (1 / currK) * 100
      const currDrop = Math.pow(currK / 10, 1.8) * 8.5 * curveFactor
      const currIoU = Math.max(25, 96 - currDrop)

      return {
        x: intervals,
        y: interpolationIoU,
        baseline: annotatorCostPct,
        stat1: `Keyframe Interval: Delta K = ${currK} frames`,
        stat2: `Labeling Effort: ${currCost.toFixed(1)}% of dense cost`,
        stat3: `Interpolated IoU: ${currIoU.toFixed(1)}% precision`,
        warning:
          currK > 16
            ? 'Sparse keyframes slash human labeling hours by >90% but produce significant trajectory drift during rapid non-linear turns'
            : 'Dense keyframe pacing preserves sub-pixel bounding box fidelity for fast kinematic actions at moderate annotation cost',
        axisXLabel: 'Annotated Keyframe Step Interval Delta K (Frames)',
        axisYLabel: 'Interpolated Bounding Box IoU (%) / Relative Labeling Cost (%)',
        xRange: [1, 31] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    } else if (annotationMode === 'tiou_localization') {
      // Temporal IoU threshold sweep from 0.3 to 0.95
      const thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
      // Ground truth action duration = 10.0 seconds
      // Jitter offset in start/end time = param1 * 0.1 seconds (0.2s to 3.0s)
      const jitterSec = param1 * 0.1
      // tIoU = Intersection / Union = (10 - jitter) / (10 + jitter)
      const actualTIoU = Math.max(0, (10 - jitterSec) / (10 + jitterSec))
      // Average Precision across thresholds
      const mApScores = thresholds.map((th) => (actualTIoU >= th ? Math.min(95, 95 * Math.pow(1 - th, 0.2)) : 0))

      const currTh = 0.5 + (param2 / 10) * 0.45 // 0.55 to 0.95
      const isAccepted = actualTIoU >= currTh

      return {
        x: thresholds,
        y: mApScores,
        baseline: thresholds.map(() => actualTIoU * 100),
        stat1: `Boundary Jitter: ±${jitterSec.toFixed(1)}s on 10s action`,
        stat2: `Empirical tIoU: ${(actualTIoU * 100).toFixed(1)}%`,
        stat3: `Eval Threshold (${currTh.toFixed(2)}): ${isAccepted ? 'HIT (Accepted)' : 'MISS (Rejected)'}`,
        warning:
          actualTIoU < 0.7
            ? 'Action boundary ambiguities degrade high-threshold tIoU mAP@0.75; human annotators struggle to agree on exact start/end milliseconds'
            : 'Sharp temporal action boundaries satisfy strict ActivityNet and AVA localization evaluation protocols',
        axisXLabel: 'Temporal Intersection-over-Union (tIoU) Evaluation Cutoff Threshold',
        axisYLabel: 'Localization Average Precision mAP (%) / Empirical tIoU (%)',
        xRange: [0.28, 0.98] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    } else {
      // Identity switches (IDSW) from 1 to 20
      const idswCounts = [1, 2, 4, 6, 8, 10, 12, 14, 16, 20]
      // Detection Accuracy DetA fixed around 85%
      const detA = 85
      // Association Accuracy AssA decays as IDSW increases
      const assAScores = idswCounts.map((id) => Math.max(15, 88 - Math.pow(id, 1.2) * 2.8))
      // HOTA = sqrt(DetA * AssA)
      const hotaScores = assAScores.map((ass) => Math.sqrt(detA * ass))
      // Legacy MOTA = DetScore - (IDSW / GT) * penalty -> drops steeply
      const motaScores = idswCounts.map((id) => Math.max(0, 82 - id * 3.8))

      const currIdsw = param1
      const currAssA = Math.max(15, 88 - Math.pow(currIdsw, 1.2) * 2.8)
      const currHota = Math.sqrt(detA * currAssA)
      const currMota = Math.max(0, 82 - currIdsw * 3.8)

      return {
        x: idswCounts,
        y: hotaScores,
        baseline: motaScores,
        stat1: `Identity Switches (IDSW): ${currIdsw}`,
        stat2: `HOTA Metric: ${currHota.toFixed(1)}%`,
        stat3: `Legacy MOTA: ${currMota.toFixed(1)}% (AssA: ${currAssA.toFixed(1)}%)`,
        warning:
          currIdsw > 10
            ? 'High identity fragmentation severely degrades association accuracy AssA and HOTA; tracker repeatedly assigns new IDs to occluded objects'
            : 'Stable object tracklets maintain identity permanence across transient visual occlusions',
        axisXLabel: 'Track Identity Switches (IDSW Count Across Sequence)',
        axisYLabel: 'Tracking Benchmark Metric Score (%)',
        xRange: [0.5, 21] as [number, number],
        yRange: [0, 95] as [number, number],
      }
    }
  }, [annotationMode, param1, param2])

  const axisX = useAxis({ label: results.axisXLabel, range: results.xRange })
  const axisY = useAxis({ label: results.axisYLabel, range: results.yRange })

  return (
    <Figure
      title="Video Annotation: Keyframe Interpolation, Temporal IoU, and HOTA Tracking"
      state={state}
      hoverReadout={false}
      readouts={
        <>
          <Readout label="Annotation Parameter" value={results.stat1} />
          <Readout label="Primary Accuracy Metric" value={results.stat2} />
          <Readout label="Secondary Metric" value={results.stat3} />
          <Readout label="Labeling Diagnostic" value={results.warning} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Annotation Precision and Evaluation Response">
        <Curve
          name={
            annotationMode === 'interpolation_cost'
              ? 'Human Annotation Cost Relative to Dense (%)'
              : annotationMode === 'tiou_localization'
                ? 'Empirical Proposal tIoU Coverage (%)'
                : 'Legacy CLEAR MOTA Benchmark (%)'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            annotationMode === 'interpolation_cost'
              ? 'Interpolated Trajectory Box IoU Precision (%)'
              : annotationMode === 'tiou_localization'
                ? 'Localization Average Precision mAP (%)'
                : 'Higher Order Tracking Accuracy (HOTA) (%)'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Benchmark Data Points" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
