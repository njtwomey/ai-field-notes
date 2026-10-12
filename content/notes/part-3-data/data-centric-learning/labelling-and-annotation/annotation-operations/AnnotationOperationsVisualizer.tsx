import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function AnnotationOperationsVisualizer() {
  const state = useFigureState({
    opsLens: choice<'gold_monitor' | 'fatigue_drift' | 'funnel'>(
      [
        { value: 'gold_monitor', label: 'Gold Trap Monitoring & Rater Quarantining' },
        { value: 'fatigue_drift', label: 'Longitudinal Fatigue Drift (Speed vs Error Surge)' },
        { value: 'funnel', label: 'Multi-Tier Escalation Funnel & Unit Cost' },
      ],
      'gold_monitor',
      { label: 'operational monitoring lens' },
    ),
    goldCutoff: slider(60, 90, 80, {
      step: 5,
      label: 'minimum gold accuracy cutoff (%)',
    }),
    quarantineAction: choice<'auto' | 'none'>(
      [
        { value: 'auto', label: 'Automated Quarantining & Throttling Enabled' },
        { value: 'none', label: 'No Automatic Filtering (Polluted Dataset)' },
      ],
      'auto',
      { label: 'quarantine enforcement' },
    ),
  })

  const { opsLens, goldCutoff, quarantineAction } = state

  const results = useMemo(() => {
    if (opsLens === 'gold_monitor') {
      // 5 annotators: A1, A2, A3, A4, A5
      const raters = [1, 2, 3, 4, 5]
      const goldAccuracies = [96.0, 92.0, 74.0, 48.0, 88.0]
      const cutoffLine = raters.map(() => goldCutoff)

      const passing = goldAccuracies.filter((a) => a >= goldCutoff).length
      const quarantined = 5 - passing

      return {
        x: raters,
        y: goldAccuracies,
        baseline: cutoffLine,
        stat1: `${passing} raters approved, ${quarantined} quarantined`,
        stat2:
          quarantineAction === 'auto'
            ? 'Annotator 4 (Spammer) & Annotator 3 (Fatigued) revoked'
            : 'All noisy labels leak into training set',
        alert: quarantined > 0 ? 'Quality gate triggered: Low-performing raters isolated' : 'All raters pass threshold',
      }
    } else if (opsLens === 'fatigue_drift') {
      // 8 time intervals across a 4-hour session (every 30 mins)
      const periods = [1, 2, 3, 4, 5, 6, 7, 8]
      // Task speed drops from 40s to 12s (rushing)
      // Error rate surges from 4% to 26%
      const errorRates = [4.0, 4.5, 5.0, 6.2, 9.5, 14.0, 21.0, 27.5]
      const baselineTolerance = periods.map(() => 8.0)

      return {
        x: periods,
        y: errorRates,
        baseline: baselineTolerance,
        stat1: 'Hour 1-2: 4.8% error rate (Nominal)',
        stat2: 'Hour 3-4: 24.2% error rate (Fatigue spike)',
        alert: 'Recommend enforcing 10-minute rest breaks after 90 minutes of continuous labeling',
      }
    } else {
      // Escalation funnel: 4 stages
      // 1: Ingested Batch (10,000 tasks, $0 cost)
      // 2: Tier 1 Crowd Consensus (8,200 accepted at $0.15/ea)
      // 3: Tier 2 Senior Adjudication (1,800 escalated at $1.20/ea)
      // 4: Golden Trap Re-checks (500 audits at $0.40/ea)
      const stages = [1, 2, 3, 4]
      const volumePct = [100.0, 82.0, 18.0, 5.0]
      const targetLine = stages.map(() => 15.0)

      return {
        x: stages,
        y: volumePct,
        baseline: targetLine,
        stat1: 'Total Batch Cost: $3,590 for 10,000 items',
        stat2: 'Effective Unit Cost: $0.36 per gold-standard label',
        alert: 'Escalation routing preserves budget by directing expensive experts only to ambiguous ties',
      }
    }
  }, [opsLens, goldCutoff, quarantineAction])

  const axisX = useAxis({
    label:
      opsLens === 'gold_monitor'
        ? 'Annotator Identity (A1: Senior, A2: Good, A3: Fatigued, A4: Spammer, A5: Average)'
        : opsLens === 'fatigue_drift'
          ? '30-Minute Shift Intervals (Intervals 1-4: Fresh, 5-8: Fatigued)'
          : 'Escalation Pipeline Stage (1: Raw, 2: Tier 1 Consensus, 3: Adjudication, 4: Audit)',
    range: opsLens === 'gold_monitor' ? [0.5, 5.5] : opsLens === 'fatigue_drift' ? [0.5, 8.5] : [0.5, 4.5],
  })

  const axisY = useAxis({
    label:
      opsLens === 'gold_monitor'
        ? 'Rolling Accuracy on Gold Trap Questions (%)'
        : opsLens === 'fatigue_drift'
          ? 'Empirical Annotation Error Rate (%)'
          : 'Pipeline Volume Share (%)',
    range: [0, 105],
  })

  return (
    <Figure
      title="Annotation Operations & Quality Control Explorer"
      state={state}
      hoverReadout={false}
      caption="Operational management of labeling workflows: Tracking rater fidelity via hidden gold questions, diagnosing longitudinal session fatigue surges, and optimizing multi-tier adjudication funnels."
      readouts={
        <>
          <Readout label="Operational Status" value={results.stat1} />
          <Readout label="Quality Diagnostic" value={results.stat2} />
          <Readout
            label="Policy Recommendation"
            value={results.alert}
            color={opsLens === 'gold_monitor' && goldCutoff > 85 ? 'var(--amber-600)' : 'var(--blue-600)'}
          />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Annotation Operations Metric Profile">
        <Curve name="Threshold / Baseline Tolerance" x={results.x} y={results.baseline} slot={2} dashed width={1.8} />
        {opsLens === 'fatigue_drift' ? (
          <Curve name="Error Trajectory Over Time" x={results.x} y={results.y} slot={0} width={2.5} />
        ) : (
          <Points name="Observed Metric Performance" x={results.x} y={results.y} slot={0} size={8} />
        )}
      </Plot>
    </Figure>
  )
}
