import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function SelectiveLabelsVisualizer() {
  const state = useFigureState({
    viewMode: choice<'approval_censoring' | 'reject_inference'>(
      [
        { value: 'approval_censoring', label: 'Selective Labels: Observed Approved vs Unobserved Rejects' },
        { value: 'reject_inference', label: 'Exploration Slice: Randomized Passes Reveal Counterfactual Truth' },
      ],
      'approval_censoring',
      { label: 'selective labels view' },
    ),
    approvalCutoff: slider(520, 720, 620, {
      step: 10,
      label: 'underwriting score approval threshold tau (applicants below tau rejected)',
    }),
    explorationRate: slider(0.0, 0.15, 0.05, {
      step: 0.01,
      label: 'randomized exploration pass rate on rejected pool (epsilon)',
    }),
  })

  const results = useMemo(() => {
    const scores: number[] = []
    const trueDefaultProb: number[] = []
    const observedDefaultProb: number[] = []

    const tau = state.approvalCutoff
    const eps = state.explorationRate

    let totalApplicants = 0
    let totalApproved = 0
    let approvedDefaults = 0
    let rejectedDefaults = 0
    let totalRejected = 0
    let explorationCount = 0

    for (let s = 450; s <= 800; s += 10) {
      scores.push(s)
      // Logistic ground truth default rate
      const logit = -0.018 * (s - 580)
      const pDef = 1.0 / (1.0 + Math.exp(-logit))
      trueDefaultProb.push(pDef * 100.0)

      // Assume applicant density roughly normal centered at 600
      const weight = Math.exp(-0.5 * Math.pow((s - 600) / 70, 2))
      totalApplicants += weight

      if (s >= tau) {
        // Deterministically approved
        observedDefaultProb.push(pDef * 100.0)
        totalApproved += weight
        approvedDefaults += weight * pDef
      } else {
        totalRejected += weight
        rejectedDefaults += weight * pDef

        if (state.viewMode === 'reject_inference' && eps > 0) {
          // Explored slice reveals default probability with slight noise
          observedDefaultProb.push(pDef * 100.0)
          explorationCount += weight * eps
        } else {
          // Unobserved: zero data available
          observedDefaultProb.push(0.0)
        }
      }
    }

    const approvalRatePct = (totalApproved / totalApplicants) * 100.0
    const approvedDefaultRate = totalApproved > 0 ? (approvedDefaults / totalApproved) * 100.0 : 0
    const rejectedDefaultRate = totalRejected > 0 ? (rejectedDefaults / totalRejected) * 100.0 : 0

    const cutoffX = [tau, tau]
    const cutoffY = [0, 85]

    return {
      scores,
      trueDefaultProb,
      observedDefaultProb,
      cutoffX,
      cutoffY,
      approvalRatePct,
      approvedDefaultRate,
      rejectedDefaultRate,
      explorationCount: Math.round(explorationCount * 100),
    }
  }, [state.approvalCutoff, state.explorationRate, state.viewMode])

  const axisX = useAxis({
    label: 'Credit Score / Underwriting Feature X (Cutoff tau)',
    range: [440, 810],
  })

  const axisY = useAxis({
    label: 'Default Rate (%)',
    range: [0, 90],
  })

  return (
    <Figure
      title="The Selective Labels Dilemma: Missing Outcomes on Rejected Applications"
      state={state}
      hoverReadout={false}
      caption="Under selective labels, outcome Y (loan default) is observed only for applicants who passed the approval threshold tau. Training exclusively on approved cases blinds the model to the unobserved counterfactual default rate of rejected applicants, leading to severe evaluation and policy skew."
      readouts={
        <>
          <Readout label="Portfolio Approval Rate" value={`${results.approvalRatePct.toFixed(1)}%`} />
          <Readout label="Observed Approved Defaults" value={`${results.approvedDefaultRate.toFixed(1)}%`} />
          <Readout label="Latent Rejected Defaults" value={`${results.rejectedDefaultRate.toFixed(1)}% (Unobserved)`} />
          <Readout
            label="Exploration Benefit"
            value={
              state.viewMode === 'reject_inference'
                ? `Epsilon pass identifies unobservables`
                : 'Zero label visibility below tau'
            }
          />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Ground Truth Default Curve vs Observed Label Support">
        <Curve
          name="Latent Population Truth P(Default | X)"
          x={results.scores}
          y={results.trueDefaultProb}
          slot={0}
          width={2.5}
        />
        <Curve
          name="Approval Threshold Cutoff (tau)"
          x={results.cutoffX}
          y={results.cutoffY}
          slot={2}
          dashed
          width={2}
        />
        {state.viewMode === 'approval_censoring' ? (
          <Points
            name="Observed Training Labels"
            x={results.scores.filter((s) => s >= state.approvalCutoff)}
            y={results.observedDefaultProb.filter((_, idx) => results.scores[idx] >= state.approvalCutoff)}
            slot={0}
            size={7}
          />
        ) : (
          <Points
            name="Exploration-Assisted Labels"
            x={results.scores}
            y={results.observedDefaultProb}
            slot={1}
            size={6}
          />
        )}
      </Plot>
    </Figure>
  )
}
