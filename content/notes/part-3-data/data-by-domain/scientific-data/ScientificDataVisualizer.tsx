import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function ScientificDataVisualizer() {
  const state = useFigureState({
    modelType: choice<'unconstrained_mlp' | 'physics_constrained'>(
      [
        { value: 'unconstrained_mlp', label: 'Unconstrained Black-Box Neural Network' },
        { value: 'physics_constrained', label: 'Physics-Informed / Conservation-Enforced Model' },
      ],
      'unconstrained_mlp',
      { label: 'inductive bias' },
    ),
    noiseSigma: slider(0.0, 0.25, 0.08, {
      step: 0.01,
      label: 'measurement uncertainty (sensor noise sigma)',
    }),
    extrapolationTime: slider(8, 16, 12, {
      step: 1,
      label: 'evaluation horizon (training observed only for t <= 6)',
    }),
  })

  const results = useMemo(() => {
    const tGrid: number[] = []
    const trueTrajectory: number[] = []
    const predTrajectory: number[] = []

    const omega = 1.0
    const maxT = state.extrapolationTime

    for (let t = 0; t <= maxT; t += 0.2) {
      tGrid.push(t)
      const trueVal = Math.cos(omega * t)
      trueTrajectory.push(trueVal)

      let predVal = 0
      if (state.modelType === 'physics_constrained') {
        // Enforces Hamiltonian energy conservation: perfectly periodic with minor amplitude noise
        const dampingParam = 0.002
        predVal = Math.exp(-dampingParam * t) * Math.cos(omega * t)
      } else {
        // Unconstrained: fits t <= 6 well, but drifts/damps or explodes for t > 6
        if (t <= 6) {
          predVal = Math.cos(omega * t) + 0.05 * Math.sin(3 * omega * t)
        } else {
          const deltaT = t - 6
          // Spurious non-physical damping and frequency shift
          predVal = Math.exp(-0.25 * deltaT) * Math.cos((omega + 0.15 * deltaT) * t)
        }
      }
      predTrajectory.push(predVal)
    }

    // Sparse observed training measurements up to t=6
    const trainT = [0.5, 1.2, 2.0, 2.8, 3.5, 4.3, 5.1, 5.8]
    const trainY = trainT.map((t, idx) => {
      // Deterministic pseudorandom perturbation based on index
      const pseudoNoise = Math.sin(idx * 4.7) * state.noiseSigma
      return Math.cos(omega * t) + pseudoNoise
    })

    // Calculate energy drift at horizon maxT
    const finalPredX = predTrajectory[predTrajectory.length - 1]
    const finalPredE = 0.5 * Math.pow(finalPredX, 2)
    const energyDriftPct = (Math.abs(finalPredE - 0.5) / 0.5) * 100.0

    return {
      tGrid,
      trueTrajectory,
      predTrajectory,
      trainT,
      trainY,
      energyDriftPct,
    }
  }, [state.extrapolationTime, state.modelType, state.noiseSigma])

  const axisX = useAxis({
    label: 'Time t (s) [Training cutoff at t=6s]',
    range: [0, 16.5],
  })

  const axisY = useAxis({
    label: 'State Position x(t) [meters]',
    range: [-1.4, 1.4],
  })

  return (
    <Figure
      title="Physical Constraints and Long-Horizon Extrapolation in Scientific Telemetry"
      state={state}
      hoverReadout={false}
      caption="Simulated and empirical scientific datasets obey fundamental conservation laws (e.g., energy conservation in Hamiltonian mechanics). Pure data-driven regressors fit sparse observations but violate physical invariants during extrapolation, whereas physics-informed architectures guarantee valid conservation."
      readouts={
        <>
          <Readout label="Training Data Horizon" value="t in [0s, 6s] (8 samples)" />
          <Readout
            label="Energy Conservation"
            value={
              state.modelType === 'physics_constrained'
                ? 'Invariant (Hamiltonian preserved)'
                : `Violated (${results.energyDriftPct.toFixed(0)}% drift)`
            }
          />
          <Readout
            label="Extrapolation Fidelity"
            value={
              state.modelType === 'physics_constrained' ? 'Phase-accurate long horizon' : 'Non-physical dampening decay'
            }
          />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="System Dynamics: True vs Predicted Evolution">
        <Curve
          name="Analytical Physical Truth x(t)"
          x={results.tGrid}
          y={results.trueTrajectory}
          slot={0}
          width={1.8}
          dashed
        />
        <Curve
          name="Model Prediction"
          x={results.tGrid}
          y={results.predTrajectory}
          slot={state.modelType === 'physics_constrained' ? 0 : 2}
          width={2.5}
        />
        <Points name="Noisy Training Observations" x={results.trainT} y={results.trainY} slot={1} size={7} />
      </Plot>
    </Figure>
  )
}
