import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function TargetExplorationVisualizer() {
  const state = useFigureState({
    probeMode: choice<'distribution' | 'temporal' | 'screening'>(
      [
        { value: 'distribution', label: 'Distribution & Skew (Zero-Inflation & Long Tails)' },
        { value: 'temporal', label: 'Temporal Stability (Drift, Seasonality, Policy Shifting)' },
        { value: 'screening', label: 'Single-Feature Screening (Predictive Power vs Leakage)' },
      ],
      'distribution',
      { label: 'diagnostic lens' },
    ),
    zeroInflation: slider(0, 70, 30, {
      step: 5,
      label: 'zero-inflation % (point mass at y=0)',
    }),
    skewness: slider(0.4, 2.5, 1.4, {
      step: 0.1,
      label: 'continuous right-tail skewness',
    }),
    policyShift: slider(-25, 25, 12, {
      step: 2,
      label: 'week-6 policy step jump (% pts)',
    }),
    leakStrength: slider(0.0, 1.0, 0.7, {
      step: 0.1,
      label: 'post-outcome leakage in feature 3',
    }),
  })

  const { probeMode, zeroInflation, skewness, policyShift, leakStrength } = state

  const results = useMemo(() => {
    if (probeMode === 'distribution') {
      const xVals: number[] = []
      const yVals: number[] = []
      const zeroFrac = zeroInflation / 100

      // Spike at 0
      xVals.push(0)
      yVals.push(zeroFrac * 70)

      for (let x = 0.2; x <= 10; x += 0.2) {
        xVals.push(x)
        const logx = Math.log(x)
        const dens =
          (1 - zeroFrac) *
          (1 / (x * skewness * Math.sqrt(2 * Math.PI))) *
          Math.exp(-Math.pow(logx - 0.8, 2) / (2 * skewness * skewness))
        yVals.push(dens * 50)
      }

      return {
        x: xVals,
        y: yVals,
        f3Auc: 50.0,
        threshold: xVals.map(() => 0),
      }
    } else if (probeMode === 'temporal') {
      const weeks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
      const baseRate = 18.0
      const rates = weeks.map((w) => {
        const season = 4.0 * Math.sin((w / 12) * 2 * Math.PI * 2)
        const shift = w >= 6 ? policyShift : 0
        return Math.max(1, Math.min(60, baseRate + season + shift))
      })

      return {
        x: weeks,
        y: rates,
        f3Auc: 50.0,
        threshold: weeks.map(() => baseRate),
      }
    } else {
      // screening
      const featureIdxs = [1, 2, 3, 4, 5]
      const f3Auc = 52.0 + 46.0 * leakStrength
      const aucs = [58.0, 69.0, f3Auc, 54.0, 50.0]
      const thresh = featureIdxs.map(() => 85.0)

      return {
        x: featureIdxs,
        y: aucs,
        f3Auc,
        threshold: thresh,
      }
    }
  }, [probeMode, zeroInflation, skewness, policyShift, leakStrength])

  const axisX = useAxis({
    label:
      probeMode === 'distribution'
        ? 'Target Variable Value (y)'
        : probeMode === 'temporal'
          ? 'Time Period (Calendar Weeks)'
          : 'Feature Index (1: Tenure, 2: Activity, 3: Suspect Flag, 4: Device, 5: Noise)',
    range: probeMode === 'distribution' ? [-0.5, 10.5] : probeMode === 'temporal' ? [0.5, 12.5] : [0.5, 5.5],
  })

  const axisY = useAxis({
    label:
      probeMode === 'distribution'
        ? 'Density / Frequency (%)'
        : probeMode === 'temporal'
          ? 'Positive Class Rate (%)'
          : 'Univariate ROC-AUC (%)',
    range: probeMode === 'distribution' ? [0, 80] : probeMode === 'temporal' ? [0, 50] : [45, 102],
  })

  return (
    <Figure
      title="Target Variable Diagnostic Explorer"
      state={state}
      hoverReadout={false}
      caption="Interactive exploration of target characteristics: zero-inflation & skewness transformations, temporal base-rate regime shifts, and single-feature leakage screening."
      readouts={
        <>
          {probeMode === 'distribution' && (
            <>
              <Readout label="Zero Spike Mass" value={`${zeroInflation}%`} />
              <Readout
                label="Hurdle Recommendation"
                value={zeroInflation > 20 ? 'Two-stage hurdle or Tweedie loss' : 'Standard regression suffices'}
              />
              <Readout label="Tail Skewness" value={`${skewness.toFixed(1)}`} />
            </>
          )}
          {probeMode === 'temporal' && (
            <>
              <Readout label="Policy Step Jump" value={`${policyShift > 0 ? '+' : ''}${policyShift}% pts`} />
              <Readout
                label="Stationarity Status"
                value={
                  Math.abs(policyShift) > 10 ? 'Non-stationary regime shift (use time split)' : 'Stationary trajectory'
                }
              />
            </>
          )}
          {probeMode === 'screening' && (
            <>
              <Readout label="Feature 3 Univariate AUC" value={`${results.f3Auc.toFixed(1)}%`} />
              <Readout
                label="Leakage Status"
                value={results.f3Auc > 85 ? 'CRITICAL LEAKAGE DETECTED (Feature 3)' : 'Plausible predictor'}
              />
            </>
          )}
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Target Diagnostics Profile">
        <Curve name="Observed Target Pattern" x={results.x} y={results.y} slot={0} width={2.5} />
        <Points
          name="Data Points"
          x={results.x}
          y={results.y}
          slot={results.f3Auc > 85 && probeMode === 'screening' ? 1 : 0}
          size={7}
        />
        {probeMode === 'screening' && (
          <Curve name="Leakage Threshold (85% AUC)" x={results.x} y={results.threshold} slot={2} dashed width={2} />
        )}
      </Plot>
    </Figure>
  )
}
