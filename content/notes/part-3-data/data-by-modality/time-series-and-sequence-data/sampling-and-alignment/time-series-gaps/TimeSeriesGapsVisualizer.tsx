import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function TimeSeriesGapsVisualizer() {
  const state = useFigureState({
    visualizerMode: choice<'imputation_strategies' | 'grud_decay' | 'lookahead_leakage'>(
      [
        { value: 'imputation_strategies', label: 'Imputation Profiles: Forward Fill vs Bounded Staleness' },
        { value: 'grud_decay', label: 'GRU-D Temporal Decay: Elapsed Time Delta vs Mean Reversion (Che 2018)' },
        { value: 'lookahead_leakage', label: 'Look-Ahead Bias: Bidirectional Spline vs Causal Evaluation Gap' },
      ],
      'imputation_strategies',
      { label: 'gap processing paradigm' },
    ),
    gapDuration: slider(2, 10, 5, {
      step: 1,
      label: 'missing gap length delta_t',
    }),
    stalenessThreshold: slider(1, 8, 3, {
      step: 1,
      label: 'max allowed staleness tau_max',
    }),
  })

  const { visualizerMode, gapDuration, stalenessThreshold } = state

  const results = useMemo(() => {
    if (visualizerMode === 'imputation_strategies') {
      // Time axis from t = 0 to 14
      const timeSteps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
      // Gap starts at t = 3, lasts gapDuration steps
      // Pre-gap value at t=3 is 70, post-gap value at t=3+gapDuration is 40
      const gapStart = 3
      const gapEnd = Math.min(13, gapStart + gapDuration)

      // Forward Fill (LOCF): stays at 70 throughout the gap
      const locf = timeSteps.map((t) => {
        if (t <= gapStart) return 70 - t * 2
        if (t < gapEnd) return 70 - gapStart * 2
        return 40 - (t - gapEnd) * 1.5
      })

      // Bounded staleness: after stalenessThreshold steps into gap, decays to global mean (50)
      const bounded = timeSteps.map((t) => {
        if (t <= gapStart) return 70 - t * 2
        const elapsed = t - gapStart
        if (t < gapEnd) {
          if (elapsed <= stalenessThreshold) return 70 - gapStart * 2
          return 50 // reverted to empirical baseline / flagged invalid
        }
        return 40 - (t - gapEnd) * 1.5
      })

      const currElapsed = gapDuration
      const isStale = currElapsed > stalenessThreshold

      return {
        x: timeSteps,
        y: bounded,
        baseline: locf,
        stat1: `Gap duration: ${gapDuration} steps (t=${gapStart} to ${gapEnd})`,
        stat2: `Max staleness threshold: ${stalenessThreshold} steps`,
        stat3: isStale ? 'Stale observation expired' : 'Within staleness limit',
        warning: isStale
          ? `Gap exceeds staleness threshold by ${currElapsed - stalenessThreshold} steps: forward-filled values revert to empirical baseline`
          : 'Values forward-filled safely within permitted freshness window',
        axisXLabel: 'Timeline Timestamp (Steps t)',
        axisYLabel: 'Imputed Sensor Signal Value',
        xRange: [-0.5, 14.5] as [number, number],
        yRange: [25, 80] as [number, number],
      }
    } else if (visualizerMode === 'grud_decay') {
      // Elapsed steps delta_t from 0 to 12
      const deltas = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12]
      // Decay factor gamma = exp(-0.35 * delta)
      const gamma = deltas.map((d) => Math.exp(-0.35 * d) * 100)
      // Mean reversion weight (1 - gamma)
      const meanReversion = deltas.map((d) => (1 - Math.exp(-0.35 * d)) * 100)

      const currDelta = gapDuration
      const currGamma = Math.exp(-0.35 * currDelta) * 100
      const currReversion = (1 - Math.exp(-0.35 * currDelta)) * 100

      return {
        x: deltas,
        y: gamma,
        baseline: meanReversion,
        stat1: `Elapsed time delta: ${currDelta} steps`,
        stat2: `Last-obs weight gamma: ${currGamma.toFixed(1)}%`,
        stat3: `Global-mean weight: ${currReversion.toFixed(1)}%`,
        warning:
          currGamma < 20
            ? 'Long gap: Imputation memory has faded; GRU-D state relies almost entirely on global empirical mean'
            : 'Short gap: Recent measurement carries dominant weight in continuous hidden state update',
        axisXLabel: 'Elapsed Inactivity Duration (Steps delta_t)',
        axisYLabel: 'Weight Contribution to State Update (%)',
        xRange: [-0.5, 12.5] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    } else {
      // Look-ahead leakage: Artificial offline test score improvement from acausal interpolation
      const gaps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      // Causal forward fill test error (MAE): error grows with gap length
      const causalError = gaps.map((g) => 2.5 + g * 1.8)
      // Acausal bidirectional interpolation error: artificially low because it cheats with future anchor
      const acausalError = gaps.map((g) => 2.5 + g * 0.4)

      const currCausal = 2.5 + gapDuration * 1.8
      const currAcausal = 2.5 + gapDuration * 0.4
      const leakageGap = currCausal - currAcausal

      return {
        x: gaps,
        y: causalError,
        baseline: acausalError,
        stat1: `Gap length: ${gapDuration} steps`,
        stat2: `Causal online MAE: ${currCausal.toFixed(1)}`,
        stat3: `Acausal leaked gap: -${leakageGap.toFixed(1)} MAE (biased)`,

        warning:
          'CRITICAL LEAKAGE: Centred spline or backward fill peeks at post-gap future measurements, creating phantom test accuracy',
        axisXLabel: 'Missing Series Gap Duration (Steps)',
        axisYLabel: 'Forecast Evaluation Error (MAE)',
        xRange: [0.5, 10.5] as [number, number],
        yRange: [0, Math.max(...causalError) * 1.1] as [number, number],
      }
    }
  }, [visualizerMode, gapDuration, stalenessThreshold])

  const axisX = useAxis({
    label: results.axisXLabel,
    range: results.xRange,
  })

  const axisY = useAxis({
    label: results.axisYLabel,
    range: results.yRange,
  })

  return (
    <Figure
      title="Gaps in Time Series: Imputation Dynamics, Temporal Decay, and Look-Ahead Leakage"
      state={state}
      hoverReadout={false}
      caption="Evaluating missingness in temporal series: Forward fill bounded by maximum staleness, continuous decay toward population baselines in GRU-D, and the catastrophic validation gap induced by acausal interpolation."
      readouts={
        <>
          <Readout label="Temporal Span" value={results.stat1} />
          <Readout label="Freshness Policy" value={results.stat2} />
          <Readout
            label={
              visualizerMode === 'imputation_strategies'
                ? 'Freshness Status'
                : visualizerMode === 'grud_decay'
                  ? 'Mean Reversion'
                  : 'Evaluation Gap'
            }
            value={results.stat3}
            color={visualizerMode === 'lookahead_leakage' ? 'var(--red-600)' : 'var(--blue-600)'}
          />
          <Readout label="Pipeline Directive" value={results.warning} color="var(--amber-600)" />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Gap Imputation Profile">
        <Curve
          name={
            visualizerMode === 'imputation_strategies'
              ? 'Unbounded LOCF (Indefinite Carry-Forward)'
              : visualizerMode === 'grud_decay'
                ? 'Global Empirical Baseline Weight (1 - gamma)'
                : 'Acausal Bidirectional Interpolation (Look-Ahead Leakage)'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            visualizerMode === 'imputation_strategies'
              ? 'Bounded Staleness Imputation Trajectory'
              : visualizerMode === 'grud_decay'
                ? 'Last-Observation Carry Weight gamma'
                : 'Causal Online Forecast Error (Realistic)'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Discrete Observations" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
