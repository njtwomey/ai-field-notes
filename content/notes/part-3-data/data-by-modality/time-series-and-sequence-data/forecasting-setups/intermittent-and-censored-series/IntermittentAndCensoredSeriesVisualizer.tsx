import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function IntermittentAndCensoredSeriesVisualizer() {
  const state = useFigureState({
    visualizerMode: choice<'croston_sba' | 'censored_stockout' | 'zero_inflation_tradeoff'>(
      [
        { value: 'croston_sba', label: 'Intermittent Demand: Croston vs SBA Bias Correction (Syntetos 2005)' },
        { value: 'censored_stockout', label: 'Stock-Out Censoring: Latent Demand vs Observed Sales (Tobin 1958)' },
        { value: 'zero_inflation_tradeoff', label: 'Zero-Inflation: Zero Mass % vs Continuous Forecast Distortion' },
      ],
      'croston_sba',
      { label: 'demand modeling regime' },
    ),
    intermittencyRate: slider(40, 90, 70, {
      step: 5,
      label: 'sparsity fraction (% zero days)',
    }),
    stockoutSeverity: slider(1, 8, 4, {
      step: 1,
      label: 'stock-out supply bottleneck level',
    }),
  })

  const { visualizerMode, intermittencyRate, stockoutSeverity } = state

  const results = useMemo(() => {
    if (visualizerMode === 'croston_sba') {
      // Timeline of 12 periods
      const periods = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
      // Average inter-arrival interval p = 1 / (1 - intermittencyRate / 100)
      const avgInterval = 1 / (1 - intermittencyRate / 100)
      const smoothingAlpha = 0.2
      // Croston estimate: rate = size / interval
      // SBA estimate: rate = (1 - alpha/2) * (size / interval)
      const baseDemandSize = 15
      const crostonRate = periods.map(() => baseDemandSize / avgInterval)
      const sbaRate = periods.map(() => (1 - smoothingAlpha / 2) * (baseDemandSize / avgInterval))

      const currCroston = baseDemandSize / avgInterval
      const currSBA = (1 - smoothingAlpha / 2) * (baseDemandSize / avgInterval)
      const biasPercent = ((currCroston - currSBA) / currSBA) * 100

      return {
        x: periods,
        y: crostonRate,
        baseline: sbaRate,
        stat1: `Zero-demand periods: ${intermittencyRate}%`,
        stat2: `Mean inter-arrival: ${avgInterval.toFixed(1)} periods`,
        stat3: `Croston upward bias: +${biasPercent.toFixed(1)}%`,
        warning:
          'Standard Croston method suffers systematic positive bias; the Syntetos-Boylan Approximation (SBA) corrects for interval smoothing variance',
        axisXLabel: 'Forecast Evaluation Period (Time Steps)',
        axisYLabel: 'Estimated Demand Rate (Units / Period)',
        xRange: [0.5, 12.5] as [number, number],
        yRange: [0, Math.max(...crostonRate) * 1.5] as [number, number],
      }
    } else if (visualizerMode === 'censored_stockout') {
      // 10 observation periods showing stock-outs
      const periods = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      // True latent customer demand: fluctuating around 20 units
      const latentDemand = [18, 22, 25, 30, 28, 26, 20, 24, 27, 22]
      // Available inventory capped by stockoutSeverity
      const inventoryCap = Math.max(8, 26 - stockoutSeverity * 2.5)
      // Observed sales = min(Demand, Inventory)
      const observedSales = latentDemand.map((d) => Math.min(d, inventoryCap))

      const totalLatent = latentDemand.reduce((a, b) => a + b, 0)
      const totalObserved = observedSales.reduce((a, b) => a + b, 0)
      const lostDemand = totalLatent - totalObserved

      return {
        x: periods,
        y: latentDemand,
        baseline: observedSales,
        stat1: `Inventory cap: ${inventoryCap.toFixed(0)} units`,
        stat2: `Observed sales: ${totalObserved} units`,
        stat3: `Unobserved lost demand: ${lostDemand} units`,
        warning:
          lostDemand > 20
            ? 'Severe stock-out truncation: Observed sales cap at available inventory; models trained on raw sales underestimate future demand'
            : 'Mild stock-out censoring: Latent demand mostly fulfilled by available inventory',
        axisXLabel: 'Sales Observation Time Period',
        axisYLabel: 'Volume (Units)',
        xRange: [0.5, 10.5] as [number, number],
        yRange: [0, 35] as [number, number],
      }
    } else {
      // Zero inflation rate vs MSE forecast distortion
      const zeroPercents = [20, 30, 40, 50, 60, 70, 80, 90]
      // Under standard MSE loss, the model predicts the unconditional mean E[Y] = (1 - p) * mu_pos
      // While true positives are around 20, the MSE prediction drops toward 2
      const unconditionalMean = zeroPercents.map((zp) => (1 - zp / 100) * 20)
      const truePositiveMean = zeroPercents.map(() => 20)

      const currUnconditional = (1 - intermittencyRate / 100) * 20

      return {
        x: zeroPercents,
        y: unconditionalMean,
        baseline: truePositiveMean,
        stat1: `Zero fraction: ${intermittencyRate}%`,
        stat2: `Positive transaction mean: 20 units`,
        stat3: `MSE prediction: ${currUnconditional.toFixed(1)} units`,
        warning:
          'MSE loss predicts a fractional trickle on zero-heavy series (e.g. 2 units every day), failing both stock replenishment and zero probability',
        axisXLabel: 'Zero Demand Frequency (% Null Observations)',
        axisYLabel: 'Unconditional Predicted Volume (Units)',
        xRange: [15, 95] as [number, number],
        yRange: [0, 25] as [number, number],
      }
    }
  }, [visualizerMode, intermittencyRate, stockoutSeverity])

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
      title="Intermittent & Censored Demand: Croston Decompositions, Stock-Out Truncation, and Hurdle Dynamics"
      state={state}
      hoverReadout={false}
      caption="Dynamics of intermittent and censored time series: The Syntetos-Boylan correction on Croston demand estimates, right-censored sales truncation under inventory stock-outs, and zero-inflation forecast distortion."
      readouts={
        <>
          <Readout label="Demand Regime" value={results.stat1} />
          <Readout label="Observed Metric" value={results.stat2} />
          <Readout
            label={
              visualizerMode === 'croston_sba'
                ? 'Croston Bias'
                : visualizerMode === 'censored_stockout'
                  ? 'Lost Sales'
                  : 'MSE Output'
            }
            value={results.stat3}
            color={visualizerMode === 'censored_stockout' ? 'var(--red-600)' : 'var(--blue-600)'}
          />
          <Readout label="Modeling Directive" value={results.warning} color="var(--amber-600)" />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Intermittent Demand Profile">
        <Curve
          name={
            visualizerMode === 'croston_sba'
              ? 'Syntetos-Boylan Corrected Rate (SBA)'
              : visualizerMode === 'censored_stockout'
                ? 'Observed Truncated Sales min(Demand, Inventory)'
                : 'Conditional Positive Transaction Mean'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            visualizerMode === 'croston_sba'
              ? 'Standard Croston Forecast (Positively Biased)'
              : visualizerMode === 'censored_stockout'
                ? 'True Latent Customer Demand'
                : 'Unconditional MSE Prediction'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Period Measurements" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
