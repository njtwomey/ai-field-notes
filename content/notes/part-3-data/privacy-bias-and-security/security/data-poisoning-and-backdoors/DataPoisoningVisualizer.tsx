import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function DataPoisoningVisualizer() {
  const state = useFigureState({
    attackType: choice<'backdoor' | 'clean_label' | 'label_flip'>(
      [
        { value: 'backdoor', label: 'BadNets Backdoor Trigger (Trojan token/patch: Gu et al. 2019)' },
        {
          value: 'clean_label',
          label: 'Clean-Label Feature Collision (Correct labels, shifted features: Shafahi et al. 2018)',
        },
        { value: 'label_flip', label: 'Untargeted Label Flipping (Degrades global model convergence)' },
      ],
      'backdoor',
      { label: 'attack mechanism' },
    ),
    poisonBudget: slider(0.1, 5.0, 1.0, {
      step: 0.1,
      label: 'poisoning budget (% of total training dataset)',
    }),
  })

  const results = useMemo(() => {
    // Generate curves across poison budget p in [0.1, 5.0]%
    const pGrid: number[] = []
    const asrGrid: number[] = []
    const cleanAccGrid: number[] = []

    for (let p = 0.1; p <= 5.0; p += 0.2) {
      pGrid.push(p)
      let asr = 0
      let cleanAcc = 95.0

      if (state.attackType === 'backdoor') {
        // Backdoors achieve near 100% ASR rapidly while leaving clean accuracy intact
        asr = 100.0 / (1.0 + Math.exp(-2.5 * (p - 0.7)))
        cleanAcc = Math.max(90, 95.2 - 0.3 * p)
      } else if (state.attackType === 'clean_label') {
        // Clean-label requires higher budget to distort decision boundary around target
        asr = 100.0 / (1.0 + Math.exp(-1.5 * (p - 1.8)))
        cleanAcc = Math.max(89, 94.8 - 0.5 * p)
      } else {
        // Random label flipping degrades clean accuracy while having zero targeted ASR
        asr = 10.0 // Baseline random chance
        cleanAcc = Math.max(50, 95.0 - 6.5 * p)
      }

      asrGrid.push(asr)
      cleanAccGrid.push(cleanAcc)
    }

    const p = state.poisonBudget
    let currentAsr = 0
    let currentCleanAcc = 95.0
    let stealthStatus = ''

    if (state.attackType === 'backdoor') {
      currentAsr = 100.0 / (1.0 + Math.exp(-2.5 * (p - 0.7)))
      currentCleanAcc = Math.max(90, 95.2 - 0.3 * p)
      stealthStatus =
        currentCleanAcc >= 94.0 ? 'Extremely Stealthy (Invisible on Clean Validation)' : 'Moderate Anomaly'
    } else if (state.attackType === 'clean_label') {
      currentAsr = 100.0 / (1.0 + Math.exp(-1.5 * (p - 1.8)))
      currentCleanAcc = Math.max(89, 94.8 - 0.5 * p)
      stealthStatus = 'Stealthy (Labels Validated by Human Auditors)'
    } else {
      currentAsr = 10.0
      currentCleanAcc = Math.max(50, 95.0 - 6.5 * p)
      stealthStatus = 'Obvious (Readily Caught by Validation Metrics)'
    }

    return {
      pGrid,
      asrGrid,
      cleanAccGrid,
      currentP: p,
      currentAsr,
      currentCleanAcc,
      stealthStatus,
    }
  }, [state.attackType, state.poisonBudget])

  const axisX = useAxis({
    label: 'adversarial poisoning fraction (% of training corpus)',
    range: [0, 5],
  })
  const axisY = useAxis({
    label: 'performance metric (%)',
    range: [0, 105],
  })

  return (
    <Figure
      title="Data poisoning regimes: targeted backdoor triggers versus indiscriminate degradation"
      state={state}
      hoverReadout={false}
      caption="Comparing attack dynamics across data poisoning paradigms. In backdoor trojan attacks (BadNets: Gu et al. 2019), poisoning less than 1% of training data yields an Attack Success Rate (ASR) exceeding 90% whenever the trigger is present, while clean validation accuracy remains virtually unaffected (>94%). Clean-label feature collisions (Shafahi et al. 2018) evade human inspection by keeping labels correct. In contrast, indiscriminate label flipping causes immediate degradation on standard validation splits, exposing the attack."
      readouts={
        <>
          <Readout label="attack success rate (ASR)" value={`${results.currentAsr.toFixed(1)}%`} />
          <Readout label="clean validation accuracy" value={`${results.currentCleanAcc.toFixed(1)}%`} />
          <Readout label="auditing detectability" value={results.stealthStatus} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Attack success rate and clean accuracy vs poison budget">
        <Curve
          name="Attack Success Rate (ASR on triggered inputs)"
          x={results.pGrid}
          y={results.asrGrid}
          slot={1}
          width={2.5}
        />
        <Curve name="Clean Validation Accuracy" x={results.pGrid} y={results.cleanAccGrid} slot={0} width={2} dashed />
        <Points name="current ASR operating point" x={[results.currentP]} y={[results.currentAsr]} slot={1} size={6} />
      </Plot>
    </Figure>
  )
}
