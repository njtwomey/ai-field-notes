import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function FinancialDataVisualizer() {
  const state = useFigureState({
    analysisType: choice<'bars' | 'survivorship'>(
      [
        { value: 'bars', label: 'Sampling Bars: Time vs Dollar Bars (López de Prado 2018)' },
        { value: 'survivorship', label: 'Survivorship Bias in Backtests (Survivors-only vs Point-in-Time universe)' },
      ],
      'bars',
      { label: 'financial data regime' },
    ),
    sliderParam: slider(1, 10, 5, {
      step: 1,
      label: 'information threshold scale / historical backtest horizon (years)',
    }),
  })

  const results = useMemo(() => {
    // Generate trade-off curves across horizon or threshold t in [1, 10]
    const xGrid: number[] = []
    const yGrid: number[] = []

    for (let t = 1; t <= 10; t += 1) {
      xGrid.push(t)
      let val = 0
      if (state.analysisType === 'bars') {
        // Return kurtosis (excess kurtosis / fat tails) drops from 12.0 down toward ~3.2 under dollar bars
        val = 12.0 * Math.exp(-0.25 * t) + 3.1
      } else {
        // Compounded equity spread: Biased index out-performs true survivorship-free index by ~3.5% per year
        const annualSurvSpread = 1.035
        val = (Math.pow(annualSurvSpread, t) - 1.0) * 100.0
      }
      yGrid.push(val)
    }

    const currentT = state.sliderParam
    let readoutLabelA = ''
    let readoutValA = ''
    let readoutLabelB = ''
    let readoutValB = ''
    let statusText = ''

    if (state.analysisType === 'bars') {
      const kurtosis = 12.0 * Math.exp(-0.25 * currentT) + 3.1
      readoutLabelA = 'return distribution kurtosis'
      readoutValA = `${kurtosis.toFixed(2)} (Gaussian = 3.0)`
      readoutLabelB = 'sampling regime'
      readoutValB = 'Dollar Bars (Fixed transaction volume per bar)'
      statusText = 'Restores near-normal returns and variance stationarity (Cont 2001)'
    } else {
      const spread = (Math.pow(1.035, currentT) - 1.0) * 100.0
      readoutLabelA = 'cumulative return inflation'
      readoutValA = `+${spread.toFixed(1)}% artificial gain`
      readoutLabelB = 'annualized survivorship alpha'
      readoutValB = '~350 bps per year'
      statusText = 'Fatal flaw: Bankrupt and delisted firms silently dropped from historical universe'
    }

    return {
      xGrid,
      yGrid,
      currentX: currentT,
      currentY: yGrid[currentT - 1],
      readoutLabelA,
      readoutValA,
      readoutLabelB,
      readoutValB,
      statusText,
    }
  }, [state.analysisType, state.sliderParam])

  const axisX = useAxis({
    label: state.analysisType === 'bars' ? 'information sampling threshold scale' : 'backtest time horizon (years)',
    range: [1, 10],
  })
  const axisY = useAxis({
    label:
      state.analysisType === 'bars'
        ? 'empirical return distribution kurtosis'
        : 'cumulative backtest return overstatement (%)',
    range: [0, state.analysisType === 'bars' ? 14 : 50],
  })

  return (
    <Figure
      title="Financial data: information-driven bars, return kurtosis, and survivorship bias"
      state={state}
      hoverReadout={false}
      caption="Evaluating quantitative financial data engineering principles (López de Prado 2018, Cont 2001). Traditional fixed-time bars (e.g. 5-minute bars) oversample quiet market periods and undersample volatile market openings, causing severe excess kurtosis and heteroskedasticity. Dollar bars sample fixed increments of transacted market value, recovering near-Gaussian returns. In backtesting, filtering on modern index constituents without point-in-time delisting history creates survivorship bias, artificially inflating annualized returns by 300-400 basis points."
      readouts={
        <>
          <Readout label={results.readoutLabelA} value={results.readoutValA} />
          <Readout label={results.readoutLabelB} value={results.readoutValB} />
          <Readout label="quantitative audit status" value={results.statusText} />
        </>
      }
    >
      <Plot
        x={axisX}
        y={axisY}
        title={
          state.analysisType === 'bars'
            ? 'Return kurtosis vs information bar sampling threshold'
            : 'Cumulative return overstatement vs backtest horizon'
        }
      >
        <Curve
          name="financial quantitative curve"
          x={results.xGrid}
          y={results.yGrid}
          slot={state.analysisType === 'survivorship' ? 1 : 0}
          width={2.5}
        />
        <Points
          name="current quantitative configuration"
          x={[results.currentX]}
          y={[results.currentY]}
          slot={state.analysisType === 'survivorship' ? 1 : 0}
          size={6}
        />
      </Plot>
    </Figure>
  )
}
