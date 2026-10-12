import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function FraudSecurityDataVisualizer() {
  const state = useFigureState({
    analysisType: choice<'base_rate' | 'delayed_chargeback'>(
      [
        {
          value: 'base_rate',
          label: 'Base-Rate Fallacy (Extreme class rarity vs investigator precision: Sommer & Paxson 2010)',
        },
        {
          value: 'delayed_chargeback',
          label: 'Chargeback Latency (Delayed ground-truth labels: Dal Pozzolo et al. 2015)',
        },
      ],
      'base_rate',
      { label: 'fraud data challenge' },
    ),
    sliderParam: slider(1, 30, 5, {
      step: 1,
      label: 'false positive rate (0.1% increments) / observation horizon (days)',
    }),
  })

  const results = useMemo(() => {
    // Generate trade-off curves across parameter in [1, 30]
    const xGrid: number[] = []
    const yGrid: number[] = []

    for (let x = 1; x <= 30; x += 1) {
      xGrid.push(x)
      let val = 0
      if (state.analysisType === 'base_rate') {
        // True positive rate TPR = 0.90, base rate p = 0.001 (1 in 1000)
        // False positive rate fpr = x * 0.001
        const p = 0.001
        const tpr = 0.9
        const fpr = x * 0.001
        // Precision = (tpr * p) / (tpr * p + fpr * (1 - p)) * 100%
        val = ((tpr * p) / (tpr * p + fpr * (1.0 - p))) * 100.0
      } else {
        // Cumulative chargeback arrival percentage: 1 - exp(-0.08 * x)
        val = 100.0 * (1.0 - Math.exp(-0.08 * x * 2.5))
      }
      yGrid.push(val)
    }

    const currentX = state.sliderParam
    let readoutLabelA = ''
    let readoutValA = ''
    let readoutLabelB = ''
    let readoutValB = ''
    let operationalState = ''

    if (state.analysisType === 'base_rate') {
      const p = 0.001
      const tpr = 0.9
      const fpr = currentX * 0.001
      const precision = ((tpr * p) / (tpr * p + fpr * (1.0 - p))) * 100.0
      readoutLabelA = 'effective investigator precision'
      readoutValA = `${precision.toFixed(1)}% (${(100 - precision).toFixed(1)}% false alarms)`
      readoutLabelB = 'classifier false positive rate'
      readoutValB = `${(fpr * 100).toFixed(2)}%`
      operationalState =
        precision < 20.0 ? 'Investigator Fatigue (Overwhelming false alarm queue)' : 'Viable Triaging Queue'
    } else {
      const completion = 100.0 * (1.0 - Math.exp(-0.08 * currentX * 2.5))
      readoutLabelA = 'chargeback labels confirmed'
      readoutValA = `${completion.toFixed(1)}%`
      readoutLabelB = 'observation latency window'
      readoutValB = `${currentX * 2.5} days elapsed`
      operationalState =
        completion < 80.0 ? 'Label Lag: Recent transaction records suffer high false negative rate' : 'Matured Cohort'
    }

    return {
      xGrid,
      yGrid,
      currentX,
      currentY: yGrid[currentX - 1],
      readoutLabelA,
      readoutValA,
      readoutLabelB,
      readoutValB,
      operationalState,
    }
  }, [state.analysisType, state.sliderParam])

  const axisX = useAxis({
    label:
      state.analysisType === 'base_rate'
        ? 'classifier false positive rate (x 0.1%)'
        : 'elapsed chargeback observation horizon (days)',
    range: [1, 30],
  })
  const axisY = useAxis({
    label:
      state.analysisType === 'base_rate' ? 'investigator precision / PPV (%)' : 'confirmed fraud labels received (%)',
    range: [0, 100],
  })

  return (
    <Figure
      title="Fraud and security data: base-rate fallacy and delayed chargeback feedback"
      state={state}
      hoverReadout={false}
      caption="Evaluating data dynamics in fraud and intrusion detection (Sommer & Paxson 2010, Dal Pozzolo et al. 2015). Because genuine fraud is extremely rare (e.g. 1 in 1,000 transactions), the base-rate fallacy dictates that even a highly specific classifier (FPR = 0.5%) produces an alert queue where over 80% of flagged events are false positives. Furthermore, confirmed fraud labels take 30 to 90 days to materialize via banking chargebacks, forcing models to train on severely delayed feedback."
      readouts={
        <>
          <Readout label={results.readoutLabelA} value={results.readoutValA} />
          <Readout label={results.readoutLabelB} value={results.readoutValB} />
          <Readout label="operational impact" value={results.operationalState} />
        </>
      }
    >
      <Plot
        x={axisX}
        y={axisY}
        title={
          state.analysisType === 'base_rate'
            ? 'Investigator precision vs false positive rate'
            : 'Cumulative fraud label arrival vs elapsed days'
        }
      >
        <Curve
          name="security metric curve"
          x={results.xGrid}
          y={results.yGrid}
          slot={state.analysisType === 'base_rate' ? 1 : 0}
          width={2.5}
        />
        <Points
          name="current operational operating point"
          x={[results.currentX]}
          y={[results.currentY]}
          slot={state.analysisType === 'base_rate' ? 1 : 0}
          size={6}
        />
      </Plot>
    </Figure>
  )
}
