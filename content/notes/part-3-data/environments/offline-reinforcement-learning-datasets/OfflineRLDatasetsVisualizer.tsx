import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { normal, stream } from 'aifn-compute/foundation/random'

export function OfflineRLDatasetsVisualizer() {
  const state = useFigureState({
    datasetQuality: choice<'random' | 'medium' | 'expert'>(
      [
        { value: 'medium', label: 'Medium quality dataset (suboptimal demonstration band)' },
        { value: 'random', label: 'Random exploration dataset (broad coverage, poor returns)' },
        { value: 'expert', label: 'Expert demonstration dataset (narrow optimal corridor)' },
      ],
      'medium',
      { label: 'offline dataset regime' },
    ),
    estimator: choice<'standard_q' | 'cql'>(
      [
        { value: 'cql', label: 'Conservative Q-Learning (CQL: penalizes OOD action values)' },
        { value: 'standard_q', label: 'Standard Q-learning (suffers OOD extrapolation spikes)' },
      ],
      'cql',
      { label: 'offline algorithm' },
    ),
    conservatismAlpha: slider(0.5, 5.0, 2.0, {
      step: 0.5,
      label: 'CQL conservative penalty weight alpha',
    }),
  })

  const results = useMemo(() => {
    const rng = stream(49)
    const N = 80
    const dataS: number[] = []
    const dataA: number[] = []

    // Optimal action manifold: a*(s) = sin(s)
    for (let i = 0; i < N; i++) {
      const s = -2.5 + (5.0 * i) / (N - 1)
      let a = 0

      if (state.datasetQuality === 'random') {
        a = normal(rng, 0, 1.2)
      } else if (state.datasetQuality === 'medium') {
        // Suboptimal behavior: flat action or offset
        a = 0.4 * Math.sin(s) + normal(rng, 0.3, 0.25)
      } else {
        // Expert: tight around optimal sin(s)
        a = Math.sin(s) + normal(rng, 0, 0.08)
      }

      dataS.push(s)
      dataA.push(Math.max(-2.2, Math.min(2.2, a)))
    }

    // Evaluate learned policy across state domain [-2.5, 2.5]
    const evalS: number[] = []
    const policyA: number[] = []
    const optimalA: number[] = []

    let maxOodSpike = 0
    let expectedReturn = 0

    for (let s = -2.5; s <= 2.55; s += 0.1) {
      evalS.push(s)
      const opt = Math.sin(s)
      optimalA.push(opt)

      let pol = 0
      if (state.estimator === 'standard_q') {
        if (state.datasetQuality === 'expert') {
          // At edge of narrow support, standard Q exploits spurious extrapolation spikes
          if (Math.abs(s) > 1.2) {
            pol = Math.sign(s) * 2.1 // Out of distribution hallucination!
            maxOodSpike = 14.5
          } else {
            pol = opt
          }
        } else if (state.datasetQuality === 'medium') {
          pol = 0.4 * Math.sin(s) + 0.3 + (Math.abs(s) > 1.5 ? 1.8 : 0)
          maxOodSpike = 8.2
        } else {
          pol = normal(rng, 0, 0.8)
          maxOodSpike = 11.0
        }
      } else {
        // CQL penalizes OOD values: stays strictly within behavioral support
        if (state.datasetQuality === 'expert') {
          pol = opt // Expert policy successfully extracted
          maxOodSpike = 0.4 / state.conservatismAlpha
        } else if (state.datasetQuality === 'medium') {
          pol = 0.4 * Math.sin(s) + 0.25
          maxOodSpike = 0.8 / state.conservatismAlpha
        } else {
          pol = 0.1 * Math.sin(s)
          maxOodSpike = 1.2 / state.conservatismAlpha
        }
      }

      policyA.push(Math.max(-2.2, Math.min(2.2, pol)))
    }

    let coveragePct = 65
    if (state.datasetQuality === 'random') {
      coveragePct = 92
      expectedReturn = state.estimator === 'cql' ? 42.0 : 18.0
    } else if (state.datasetQuality === 'medium') {
      coveragePct = 55
      expectedReturn = state.estimator === 'cql' ? 71.5 : 44.0
    } else {
      coveragePct = 25
      expectedReturn = state.estimator === 'cql' ? 96.0 : 52.0
    }

    return {
      dataS,
      dataA,
      evalS,
      policyA,
      optimalA,
      coveragePct,
      expectedReturn,
      maxOodSpike,
    }
  }, [state.datasetQuality, state.estimator, state.conservatismAlpha])

  const xAxis = useAxis({
    label: 'continuous state s',
    range: [-2.8, 2.8],
  })
  const yAxis = useAxis({
    label: 'continuous action a',
    range: [-2.5, 2.5],
  })

  return (
    <Figure
      title="Offline reinforcement learning: dataset behavioral coverage, OOD extrapolation, and conservatism"
      state={state}
      hoverReadout={false}
      caption="Evaluating offline reinforcement learning datasets. The offline dataset D provides logged transitions without environment interactions. In standard Q-learning, the policy queries actions outside the behavioral support of the dataset, where function approximators hallucinate erroneous high values, causing policy collapse upon deployment. Conservative Q-Learning (CQL) introduces a value penalty that anchors Q-values to the dataset support, guaranteeing reliable policy extraction."
      readouts={
        <>
          <Readout label="behavioral coverage" value={`${results.coveragePct}% state-action support`} />
          <Readout
            label="OOD extrapolation error"
            value={
              state.estimator === 'standard_q'
                ? `Severe (+${results.maxOodSpike.toFixed(1)} Q-error)`
                : `Bounded (+${results.maxOodSpike.toFixed(2)} Q-error)`
            }
          />
          <Readout label="expected deployment return" value={`${results.expectedReturn.toFixed(1)} / 100`} />
          <Readout
            label="policy extraction safety"
            value={state.estimator === 'cql' ? 'Conservative & Grounded' : 'Catastrophic OOD Hallucination'}
          />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} title="Logged offline dataset transitions and learned policy manifold">
        <Points
          name="logged dataset transitions (state, action) in D"
          x={results.dataS}
          y={results.dataA}
          slot={2}
          size={4}
        />
        <Curve
          name="ground-truth optimal policy a*(s)"
          x={results.evalS}
          y={results.optimalA}
          slot={1}
          dashed
          width={1.5}
        />
        <Curve name="learned policy action manifold" x={results.evalS} y={results.policyA} slot={0} width={2.5} />
      </Plot>
    </Figure>
  )
}
