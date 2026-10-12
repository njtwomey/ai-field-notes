import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function AdvertisingConversionVisualizer() {
  const state = useFigureState({
    modelingFocus: choice<'delayed_feedback' | 'negative_downsampling'>(
      [
        { value: 'delayed_feedback', label: 'Delayed Feedback Window (Attribution truncation: Chapelle 2014)' },
        {
          value: 'negative_downsampling',
          label: 'Negative Downsampling Calibration (He et al. 2014, McMahan et al. 2013)',
        },
      ],
      'delayed_feedback',
      { label: 'ad data challenge' },
    ),
    parameter: slider(1, 30, 7, {
      step: 1,
      label: 'attribution window length (days) / downsample rate divisor',
    }),
  })

  const results = useMemo(() => {
    // Generate trade-off curves across day horizon d in [1, 30]
    const xGrid: number[] = []
    const yGrid: number[] = []

    for (let d = 1; d <= 30; d += 1) {
      xGrid.push(d)
      let val = 0
      if (state.modelingFocus === 'delayed_feedback') {
        // Cumulative conversion percentage 1 - exp(-0.18 * d)
        val = 100.0 * (1.0 - Math.exp(-0.18 * d))
      } else {
        // Uncalibrated p' for a true conversion rate p = 0.01 with sampling rate w = 1 / d
        const pTrue = 0.01
        const w = 1.0 / d
        const pBiased = (pTrue * w) / (pTrue * w + (1.0 - pTrue) * w * w)
        val = pBiased * 100.0
      }
      yGrid.push(val)
    }

    const currentParam = state.parameter
    let capturedConversionPct = 0
    let falseNegativeRate = 0
    let calibratedStatus = ''

    if (state.modelingFocus === 'delayed_feedback') {
      capturedConversionPct = 100.0 * (1.0 - Math.exp(-0.18 * currentParam))
      falseNegativeRate = 100.0 - capturedConversionPct
      calibratedStatus = `Window captures ${capturedConversionPct.toFixed(1)}% of all delayed conversions`
    } else {
      const pTrue = 0.01
      const w = 1.0 / currentParam
      capturedConversionPct = (1.0 - w) * 100.0 // Negative data reduction percentage
      falseNegativeRate = (pTrue / (pTrue + (1.0 - pTrue) / w)) * 100.0
      calibratedStatus = `Calibration correction: p = p' / (p' + (1 - p') / ${w.toFixed(2)})`
    }

    return {
      xGrid,
      yGrid,
      currentX: currentParam,
      currentY: yGrid[currentParam - 1],
      capturedConversionPct,
      falseNegativeRate,
      calibratedStatus,
    }
  }, [state.modelingFocus, state.parameter])

  const axisX = useAxis({
    label:
      state.modelingFocus === 'delayed_feedback' ? 'attribution window duration (days)' : 'downsampling factor (1 / w)',
    range: [1, 30],
  })
  const axisY = useAxis({
    label:
      state.modelingFocus === 'delayed_feedback'
        ? 'cumulative conversions captured (%)'
        : 'calibrated probability distortion (%)',
    range: [0, 105],
  })

  return (
    <Figure
      title="Advertising and conversion data: delayed feedback horizons and negative downsampling"
      state={state}
      hoverReadout={false}
      caption="Evaluating conversion dynamics in ad ranking pipelines. Conversions arrive with substantial delays (hours to weeks) after ad impressions. Truncating the attribution window to 7 days captures only ~72% of true conversions, introducing false negative bias on recent clicks (Chapelle 2014). For massive negative impression logs, downsampling negatives by rate w accelerates training by 10-30x, requiring exact calibration correction p = p' / (p' + (1 - p')/w) to restore unbiased probabilities (He et al. 2014, McMahan et al. 2013)."
      readouts={
        <>
          <Readout label="conversion capture rate" value={`${results.capturedConversionPct.toFixed(1)}%`} />
          <Readout label="censored false negative bias" value={`${results.falseNegativeRate.toFixed(1)}% uncaptured`} />
          <Readout label="calibration contract" value={results.calibratedStatus} />
        </>
      }
    >
      <Plot
        x={axisX}
        y={axisY}
        title={
          state.modelingFocus === 'delayed_feedback'
            ? 'Conversion capture rate vs attribution window'
            : 'Probability distortion vs downsampling rate'
        }
      >
        <Curve name="response curve" x={results.xGrid} y={results.yGrid} slot={0} width={2.5} />
        <Points name="current operating point" x={[results.currentX]} y={[results.currentY]} slot={0} size={6} />
      </Plot>
    </Figure>
  )
}
