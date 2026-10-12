import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function RetailDemandVisualizer() {
  const state = useFigureState({
    stockoutDay: slider(3, 10, 6, {
      step: 1,
      label: 'day stock-out begins (inventory depletion)',
    }),
    replenishmentDelay: slider(1, 6, 3, {
      step: 1,
      label: 'replenishment lead time duration (days out of stock)',
    }),
    modelComparison: choice<'observed_vs_true' | 'forecast_distortion'>(
      [
        { value: 'observed_vs_true', label: 'True Demand vs Censored Observed Sales' },
        { value: 'forecast_distortion', label: 'Naive Sales Forecast vs Tobit Imputation' },
      ],
      'observed_vs_true',
      { label: 'forecasting perspective' },
    ),
  })

  const results = useMemo(() => {
    const days = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    // Baseline underlying latent demand pattern (e.g. weekly seasonality + noise)
    const baseDemand = [42, 48, 55, 60, 75, 80, 50, 44, 47, 52, 68, 78, 85, 52]

    const stockStart = state.stockoutDay
    const stockEnd = stockStart + state.replenishmentDelay

    const sales: number[] = []
    const naiveForecast: number[] = []
    const tobitImputed: number[] = []

    let totalTrueDemand = 0
    let totalSales = 0

    days.forEach((d, idx) => {
      const td = baseDemand[idx]
      totalTrueDemand += td

      // If within stockout period, sales are capped or zero
      let s = td
      if (d === stockStart) {
        s = Math.min(td, 15) // partial inventory on depletion day
      } else if (d > stockStart && d < stockEnd) {
        s = 0 // zero stock
      }
      sales.push(s)
      totalSales += s

      // Naive moving average / regression trained on observed sales severely drops
      const naivePred = d >= stockStart ? Math.round(s * 0.75 + 10) : td
      naiveForecast.push(naivePred)

      // Tobit / censored EM model accounts for zero-stock indicator I=0
      tobitImputed.push(d >= stockStart && d < stockEnd ? td : s)
    })

    const lostSales = totalTrueDemand - totalSales
    const lostPct = (lostSales / totalTrueDemand) * 100.0

    return {
      days,
      baseDemand,
      sales,
      naiveForecast,
      tobitImputed,
      totalTrueDemand,
      totalSales,
      lostSales,
      lostPct,
    }
  }, [state.stockoutDay, state.replenishmentDelay])

  const axisX = useAxis({
    label: 'Timeline (Days)',
    range: [0.5, 14.5],
  })

  const axisY = useAxis({
    label: 'Units / Day',
    range: [0, 100],
  })

  return (
    <Figure
      title="Censored Demand and Stock-Out Distortion in Retail Telemetry"
      state={state}
      hoverReadout={false}
      caption="When inventory is exhausted, observed sales fall to zero while true consumer demand remains latent. Training forecasting models directly on raw sales leads to cyclical stock-outs: the model underforecasts demand, prompting insufficient reorders."
      readouts={
        <>
          <Readout label="Latent True Demand" value={`${results.totalTrueDemand} units`} />
          <Readout label="Observed Point-of-Sale" value={`${results.totalSales} units`} />
          <Readout
            label="Censored / Lost Volume"
            value={`${results.lostSales} units (${results.lostPct.toFixed(1)}%)`}
          />
          <Readout
            label="Naive Model Bias"
            value={results.lostPct > 15 ? 'Critical downward bias' : 'Mild attenuation'}
          />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Retail Sales vs Latent Demand">
        {state.modelComparison === 'observed_vs_true' ? (
          <>
            <Curve name="Latent True Demand D(t)" x={results.days} y={results.baseDemand} slot={0} width={2.5} />
            <Points name="Observed Sales S(t) = min(D, I)" x={results.days} y={results.sales} slot={1} size={7} />
          </>
        ) : (
          <>
            <Curve name="Tobit / Imputed Demand" x={results.days} y={results.tobitImputed} slot={0} width={2} />
            <Curve name="Naive Sales Forecast" x={results.days} y={results.naiveForecast} slot={2} dashed width={2} />
            <Points name="Observed Sales S(t)" x={results.days} y={results.sales} slot={1} size={6} />
          </>
        )}
      </Plot>
    </Figure>
  )
}
