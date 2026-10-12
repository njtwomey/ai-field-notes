import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function DataAcquisitionVisualizer() {
  const state = useFigureState({
    acquisitionSource: choice<'scrape' | 'synthetic' | 'expert' | 'hybrid'>(
      [
        { value: 'scrape', label: 'Uncurated Web Scrape (0.0001 USD/sample, high noise)' },
        { value: 'synthetic', label: 'Synthetic Generation (0.005 USD/sample, moderate diversity)' },
        { value: 'expert', label: 'Expert Human Labeling (2.00 USD/sample, gold quality)' },
        { value: 'hybrid', label: 'Hybrid Active Pipeline (Bulk scrape pretrain + Expert core-set)' },
      ],
      'hybrid',
      { label: 'acquisition channel' },
    ),
    budgetUsd: slider(500, 20000, 5000, {
      step: 500,
      label: 'data acquisition budget (USD)',
    }),
  })

  const results = useMemo(() => {
    const budget = state.budgetUsd
    let unitCost = 0.0001
    let sampleVolume = 0
    let testErrorPct = 0

    if (state.acquisitionSource === 'scrape') {
      unitCost = 0.0001
      sampleVolume = budget / unitCost
      // High noise floor (18%), slow power law alpha=0.18
      testErrorPct = 18.0 + 35.0 * Math.pow(sampleVolume / 1e5, -0.18)
    } else if (state.acquisitionSource === 'synthetic') {
      unitCost = 0.005
      sampleVolume = budget / unitCost
      // Medium floor (12%), alpha=0.30
      testErrorPct = 12.0 + 30.0 * Math.pow(sampleVolume / 1e4, -0.3)
    } else if (state.acquisitionSource === 'expert') {
      unitCost = 2.0
      sampleVolume = budget / unitCost
      // Low floor (4%), high efficiency alpha=0.45
      testErrorPct = 4.0 + 40.0 * Math.pow(sampleVolume / 100, -0.45)
    } else {
      // Hybrid: 80% budget to bulk pretraining, 20% to expert targeted active labeling
      sampleVolume = (budget * 0.8) / 0.0001 + (budget * 0.2) / 2.0
      testErrorPct = 4.5 + 28.0 * Math.pow(budget / 1000, -0.42)
    }

    testErrorPct = Math.max(3.5, Math.min(60.0, testErrorPct))

    // Sweep across budget [500 .. 20000] for plot curves
    const budgetGrid: number[] = []
    const scrapeCurve: number[] = []
    const syntheticCurve: number[] = []
    const expertCurve: number[] = []
    const hybridCurve: number[] = []

    for (let b = 500; b <= 20000; b += 500) {
      budgetGrid.push(b / 1000.0) // k USD
      const nScrape = b / 0.0001
      const nSynth = b / 0.005
      const nExp = b / 2.0

      scrapeCurve.push(Math.max(18.0, 18.0 + 35.0 * Math.pow(nScrape / 1e5, -0.18)))
      syntheticCurve.push(Math.max(12.0, 12.0 + 30.0 * Math.pow(nSynth / 1e4, -0.3)))
      expertCurve.push(Math.max(4.0, 4.0 + 40.0 * Math.pow(nExp / 100, -0.45)))
      hybridCurve.push(Math.max(4.5, 4.5 + 28.0 * Math.pow(b / 1000, -0.42)))
    }

    return {
      budget,
      sampleVolume: Math.round(sampleVolume),
      testErrorPct,
      budgetGrid,
      scrapeCurve,
      syntheticCurve,
      expertCurve,
      hybridCurve,
      currentY: testErrorPct,
    }
  }, [state.acquisitionSource, state.budgetUsd])

  const axisX = useAxis({
    label: 'Acquisition Budget (k USD)',
    range: [0.5, 20.5],
  })

  const axisY = useAxis({
    label: 'Model Test Error (%)',
    range: [0, 55],
  })

  return (
    <Figure
      title="The Economics of Data Acquisition: Cost vs Marginal Model Utility"
      state={state}
      hoverReadout={false}
      caption="Cheap uncurated web scrapes provide massive sample counts but hit a performance ceiling due to noise. Pure expert labeling is too costly to scale alone. A hybrid active learning strategy achieves the optimal Pareto frontier across all budget tiers."
      readouts={
        <>
          <Readout label="Acquisition Budget" value={`${(results.budget / 1000).toFixed(1)}k USD`} />
          <Readout label="Purchased Volume (N)" value={`${results.sampleVolume.toLocaleString()} samples`} />
          <Readout label="Achieved Test Error" value={`${results.testErrorPct.toFixed(1)}%`} />
          <Readout
            label="Pareto Efficiency"
            value={state.acquisitionSource === 'hybrid' ? 'Optimal Pareto Frontier' : 'Suboptimal Single-Source'}
          />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Marginal Accuracy Returns per Acquisition Dollar">
        <Curve name="Hybrid Active Pipeline" x={results.budgetGrid} y={results.hybridCurve} slot={0} width={2.5} />
        <Curve name="Pure Expert Human Labels" x={results.budgetGrid} y={results.expertCurve} slot={1} width={2} />
        <Curve
          name="Synthetic Simulators"
          x={results.budgetGrid}
          y={results.syntheticCurve}
          slot={2}
          dashed
          width={1.8}
        />
        <Points
          name="Current Budget Allocation"
          x={[results.budget / 1000.0]}
          y={[results.currentY]}
          slot={state.acquisitionSource === 'hybrid' ? 0 : 1}
          size={8}
        />
      </Plot>
    </Figure>
  )
}
