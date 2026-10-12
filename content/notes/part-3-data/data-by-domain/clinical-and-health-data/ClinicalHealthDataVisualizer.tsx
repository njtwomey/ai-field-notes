import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function ClinicalHealthDataVisualizer() {
  const state = useFigureState({
    clinicalChallenge: choice<'informative_sampling' | 'immortal_time'>(
      [
        {
          value: 'informative_sampling',
          label: 'Informative Sampling (Test frequency carries diagnostic signal: Agniel 2018)',
        },
        { value: 'immortal_time', label: 'Immortal Time Bias (Cohort index misalignment inflates treatment survival)' },
      ],
      'informative_sampling',
      { label: 'EHR epidemiological challenge' },
    ),
    testFrequency: slider(1, 20, 6, {
      step: 1,
      label: 'clinical observation frequency (lab tests ordered per 24h / delay days)',
    }),
  })

  const results = useMemo(() => {
    // Generate risk curves across observation frequency or delay in [1, 20]
    const xGrid: number[] = []
    const yGrid: number[] = []

    for (let f = 1; f <= 20; f += 1) {
      xGrid.push(f)
      let val = 0
      if (state.clinicalChallenge === 'informative_sampling') {
        // Acuity/mortality risk surges with lab test frequency: P(ICU transfer) ~ 1 - exp(-0.14 * f)
        val = 100.0 * (1.0 - Math.exp(-0.14 * f))
      } else {
        // Apparent hazard ratio under immortal time bias (true HR = 1.0, apparent HR drops falsely)
        val = 1.0 / (1.0 + 0.12 * f)
      }
      yGrid.push(val)
    }

    const currentF = state.testFrequency
    let readoutA = ''
    let readoutB = ''
    let statusMsg = ''

    if (state.clinicalChallenge === 'informative_sampling') {
      const risk = 100.0 * (1.0 - Math.exp(-0.14 * currentF))
      readoutA = `${risk.toFixed(1)}% ICU transfer probability`
      readoutB = `${currentF} lab orders / 24h`
      statusMsg = 'High Acuity: Measurement timing itself acts as dominant diagnostic feature'
    } else {
      const biasedHr = 1.0 / (1.0 + 0.12 * currentF)
      readoutA = `HR = ${biasedHr.toFixed(2)} (False 50%+ protective effect)`
      readoutB = `${currentF} days immortal window`
      statusMsg = 'Severe Bias: Landmark analysis or time-dependent Cox model required'
    }

    return {
      xGrid,
      yGrid,
      currentX: currentF,
      currentY: yGrid[currentF - 1],
      readoutA,
      readoutB,
      statusMsg,
    }
  }, [state.clinicalChallenge, state.testFrequency])

  const axisX = useAxis({
    label:
      state.clinicalChallenge === 'informative_sampling'
        ? 'lab tests ordered per 24 hours'
        : 'exposure definition delay (days)',
    range: [1, 20],
  })
  const axisY = useAxis({
    label:
      state.clinicalChallenge === 'informative_sampling'
        ? 'predicted adverse outcome risk (%)'
        : 'apparent hazard ratio (true HR = 1.0)',
    range: [0, state.clinicalChallenge === 'informative_sampling' ? 100 : 1.1],
  })

  return (
    <Figure
      title="Clinical and health data: informative observation processes and immortal time bias"
      state={state}
      hoverReadout={false}
      caption="Evaluating epidemiological dynamics in electronic health records (EHR) (Johnson et al. 2016, Rajkomar et al. 2018). In clinical data, lab tests are ordered selectively when patients deteriorate: test frequency alone strongly predicts acuity (informative observation). In cohort studies, misaligning the study index date with treatment initiation creates an artificial 'immortal window' where treated patients cannot experience the endpoint event, falsely exaggerating treatment efficacy."
      readouts={
        <>
          <Readout label="outcome estimate" value={results.readoutA} />
          <Readout label="exposure scale" value={results.readoutB} />
          <Readout label="clinical epidemiological implication" value={results.statusMsg} />
        </>
      }
    >
      <Plot
        x={axisX}
        y={axisY}
        title={
          state.clinicalChallenge === 'informative_sampling'
            ? 'Acuity risk vs clinical observation intensity'
            : 'Spurious hazard ratio vs immortal delay window'
        }
      >
        <Curve
          name="epidemiological curve"
          x={results.xGrid}
          y={results.yGrid}
          slot={state.clinicalChallenge === 'immortal_time' ? 1 : 0}
          width={2.5}
        />
        <Points
          name="current clinical cohort operating point"
          x={[results.currentX]}
          y={[results.currentY]}
          slot={state.clinicalChallenge === 'immortal_time' ? 1 : 0}
          size={6}
        />
      </Plot>
    </Figure>
  )
}
