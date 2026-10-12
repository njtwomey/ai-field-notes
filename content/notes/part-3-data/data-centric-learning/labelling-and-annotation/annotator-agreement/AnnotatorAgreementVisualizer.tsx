import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function AnnotatorAgreementVisualizer() {
  const state = useFigureState({
    metricLens: choice<'kappa_paradox' | 'iou_overlap' | 'human_ceiling'>(
      [
        { value: 'kappa_paradox', label: 'Kappa Prevalence Paradox (High Agreement, Low Kappa)' },
        { value: 'iou_overlap', label: 'Bounding Box & Span Overlap (IoU Thresholds)' },
        { value: 'human_ceiling', label: 'Agreement as Model Performance Ceiling' },
      ],
      'kappa_paradox',
      { label: 'agreement evaluation lens' },
    ),
    prevalence: slider(50, 98, 85, {
      step: 2,
      label: 'majority class prevalence (%)',
    }),
    boundaryJitter: slider(0.0, 0.45, 0.15, {
      step: 0.05,
      label: 'annotator boundary jitter fraction',
    }),
  })

  const { metricLens, prevalence, boundaryJitter } = state

  const results = useMemo(() => {
    if (metricLens === 'kappa_paradox') {
      // Sweep prevalence from 50% to 98%
      const pGrid = [50, 60, 70, 80, 85, 90, 95, 98]
      // Assume a constant high observed agreement p_o = 92%
      // Expected agreement under marginal symmetry: p_e = p^2 + (1-p)^2
      const obsCurve = pGrid.map(() => 92.0)
      const kappaCurve = pGrid.map((pVal) => {
        const p = pVal / 100
        const pe = p * p + (1 - p) * (1 - p)
        const po = 0.92
        const k = (po - pe) / (1 - pe)
        return Math.max(0, k * 100)
      })

      const curP = prevalence / 100
      const curPe = curP * curP + (1 - curP) * (1 - curP)
      const curKappa = Math.max(0, (0.92 - curPe) / (1 - curPe))

      return {
        x: pGrid,
        y: kappaCurve,
        baseline: obsCurve,
        readout1: '92.0% observed agreement',
        readout2: `κ = ${curKappa.toFixed(2)} (Prevalence = ${prevalence}%)`,
        warning:
          curKappa < 0.4
            ? 'PARADOX ALERT: High agreement (92%) masked by extreme prevalence'
            : 'Kappa within expected bounds',
      }
    } else if (metricLens === 'iou_overlap') {
      // IoU across jitter levels 0.0 to 0.45
      const jitters = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
      // 1D/2D overlap: (1 - jitter)^2 / (1 + 2*jitter - jitter^2)
      const ious = jitters.map((j) => {
        const inter = Math.pow(1 - j, 2)
        const union = 1 + 2 * j
        return (inter / union) * 100
      })
      const thresh75 = jitters.map(() => 75.0)

      const curInter = Math.pow(1 - boundaryJitter, 2)
      const curUnion = 1 + 2 * boundaryJitter
      const curIoU = (curInter / curUnion) * 100

      return {
        x: jitters.map((j) => j * 100),
        y: ious,
        baseline: thresh75,
        readout1: `Mean IoU: ${curIoU.toFixed(1)}%`,
        readout2: curIoU >= 75 ? 'Passes Strict COCO IoU ≥ 0.75' : 'Fails Strict IoU; passes relaxed 0.50',
        warning:
          curIoU < 60
            ? 'High span boundary disagreement; annotators select inconsistent token offsets'
            : 'Reproducible bounding boundaries',
      }
    } else {
      // Model test accuracy vs human agreement (Human agreement 60% to 95%)
      const humAgrees = [60, 65, 70, 75, 80, 85, 90, 95]
      // Ceiling line y = x
      const ceiling = humAgrees.map((h) => h)
      // Realistic model accuracy tops out slightly below human ceiling
      const modelMax = humAgrees.map((h) => Math.min(h - 2, h * 0.95))

      return {
        x: humAgrees,
        y: modelMax,
        baseline: ceiling,
        readout1: 'Human Consensus = 85.0%',
        readout2: 'Max Plausible Model Accuracy = 83.0%',
        warning: 'Demanding >85% model accuracy against single-rater labels overfits human noise',
      }
    }
  }, [metricLens, prevalence, boundaryJitter])

  const axisX = useAxis({
    label:
      metricLens === 'kappa_paradox'
        ? 'Majority Class Prevalence (%)'
        : metricLens === 'iou_overlap'
          ? 'Boundary Spatial / Token Jitter (%)'
          : 'Inter-Annotator Consensus Rate (%)',
    range: metricLens === 'kappa_paradox' ? [48, 100] : metricLens === 'iou_overlap' ? [0, 48] : [58, 98],
  })

  const axisY = useAxis({
    label:
      metricLens === 'kappa_paradox'
        ? 'Cohen’s Kappa Metric Scale (κ × 100)'
        : metricLens === 'iou_overlap'
          ? 'Intersection over Union (IoU %)'
          : 'Model Validation Test Accuracy (%)',
    range: [0, 102],
  })

  return (
    <Figure
      title="Annotator Agreement & Reliability Explorer"
      state={state}
      hoverReadout={false}
      caption="Evaluating inter-rater agreement: Feinstein's prevalence paradox where high observed agreement produces paradoxically low kappa, spatial/span IoU degradation, and human agreement as an empirical model ceiling."
      readouts={
        <>
          <Readout label="Agreement Metric" value={results.readout1} />
          <Readout label="Reliability Index" value={results.readout2} />
          <Readout
            label="Methodological Warning"
            value={results.warning}
            color={metricLens === 'kappa_paradox' && prevalence > 90 ? 'var(--amber-600)' : 'var(--blue-600)'}
          />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Agreement Diagnostic Profile">
        <Curve name="Observed Metric / Model Trajectory" x={results.x} y={results.y} slot={0} width={2.5} />
        <Curve
          name="Baseline / Theoretical Ceiling Line"
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Points
          name="Operating Diagnostic State"
          x={[metricLens === 'kappa_paradox' ? prevalence : metricLens === 'iou_overlap' ? boundaryJitter * 100 : 85]}
          y={[metricLens === 'kappa_paradox' ? Number(results.readout2.split(' ')[2]) * 100 || 50 : 80]}
          slot={1}
          size={8}
        />
      </Plot>
    </Figure>
  )
}
