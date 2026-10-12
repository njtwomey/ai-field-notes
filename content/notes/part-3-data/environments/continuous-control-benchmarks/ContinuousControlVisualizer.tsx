import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { normal, stream } from 'aifn-compute/foundation/random'

export function ContinuousControlVisualizer() {
  const state = useFigureState({
    benchmarkTask: choice<'half_cheetah' | 'hopper' | 'ant'>(
      [
        { value: 'half_cheetah', label: 'HalfCheetah-v4 (17-d state, 6-d action: continuous locomotion)' },
        { value: 'hopper', label: 'Hopper-v4 (11-d state, 3-d action: unstable monopod balance)' },
        { value: 'ant', label: 'Ant-v4 (27-d state, 8-d action: 3D multi-contact quadruped)' },
      ],
      'half_cheetah',
      { label: 'continuous benchmark' },
    ),
    policyCoordination: choice<'coordinated' | 'uncoordinated'>(
      [
        { value: 'coordinated', label: 'Coordinated rhythmic gait (trained actor-critic policy)' },
        { value: 'uncoordinated', label: 'Uncoordinated joint chatter (high torque / low progress)' },
      ],
      'coordinated',
      { label: 'motor coordination' },
    ),
    ctrlPenalty: slider(0.01, 0.3, 0.1, { step: 0.02, label: 'torque penalty coefficient c_ctrl' }),
  })

  const results = useMemo(() => {
    const rng = stream(82)
    const T = 60
    const timeIndices: number[] = []
    const forwardX: number[] = []
    const torqueEffort: number[] = []

    let currentX = 0
    let totalForwardReward = 0
    let totalCtrlCost = 0
    let survived = true

    let stateDim = 17
    let actionDim = 6

    if (state.benchmarkTask === 'half_cheetah') {
      stateDim = 17
      actionDim = 6
    } else if (state.benchmarkTask === 'hopper') {
      stateDim = 11
      actionDim = 3
    } else {
      stateDim = 27
      actionDim = 8
    }

    for (let t = 0; t <= T; t++) {
      timeIndices.push(t)

      let stepVx = 0
      let stepTorqueNorm = 0

      if (state.policyCoordination === 'coordinated') {
        if (state.benchmarkTask === 'half_cheetah') {
          // Steady forward gallop
          stepVx = 3.5 + 0.4 * Math.sin(0.4 * t) + normal(rng, 0, 0.1)
          stepTorqueNorm = 0.8 + 0.2 * Math.cos(0.4 * t)
        } else if (state.benchmarkTask === 'hopper') {
          // Periodic jumping contact
          const inAir = Math.sin(0.6 * t) > 0.2
          stepVx = inAir ? 2.8 : 0.8
          stepTorqueNorm = inAir ? 0.4 : 1.6
        } else {
          // Quadruped trot
          stepVx = 2.4 + normal(rng, 0, 0.15)
          stepTorqueNorm = 1.2
        }
      } else {
        // Uncoordinated motor chattering: high energy, low forward progress
        stepVx = 0.3 + normal(rng, 0, 0.2)
        stepTorqueNorm = 2.8 + normal(rng, 0, 0.3)
        if (state.benchmarkTask === 'hopper' && t > 18) {
          // Hopper falls over without balance
          survived = false
          stepVx = 0
          stepTorqueNorm = 0
        }
      }

      currentX += stepVx * 0.05
      forwardX.push(currentX)
      torqueEffort.push(stepTorqueNorm)

      totalForwardReward += stepVx
      totalCtrlCost += state.ctrlPenalty * stepTorqueNorm ** 2

      if (!survived) break
    }

    const netEpisodeReturn = totalForwardReward - totalCtrlCost

    return {
      timeIndices,
      forwardX,
      torqueEffort,
      stateDim,
      actionDim,
      netEpisodeReturn,
      avgSpeed: currentX / Math.max(1, timeIndices.length * 0.05),
      survived,
    }
  }, [state.benchmarkTask, state.policyCoordination, state.ctrlPenalty])

  const xAxis = useAxis({
    label: 'simulation time step t (delta_t = 0.05s)',
    range: [0, 60],
  })
  const yAxis = useAxis({
    label: 'forward displacement x (meters)',
    range: [0, 12],
  })

  return (
    <Figure
      title="Continuous control locomotion benchmarks: state dimensions, forward velocity, and torque penalty"
      state={state}
      hoverReadout={false}
      caption="Evaluating continuous control locomotion benchmarks across the MuJoCo suite. The reward function balances forward progress against energy expenditure: r = v_forward - c_ctrl ||a||². A coordinated policy achieves steady forward velocity (v_x > 3 m/s) with modest torque effort, maximizing return. In contrast, uncoordinated motor chattering expends high torque with negligible forward displacement, and causes unstable systems like Hopper to collapse."
      readouts={
        <>
          <Readout label="state space" value={`${results.stateDim}-d continuous vector`} />
          <Readout label="action space" value={`${results.actionDim}-d continuous torques`} />
          <Readout label="average forward velocity" value={`${results.avgSpeed.toFixed(2)} m/s`} />
          <Readout label="net episode return" value={results.netEpisodeReturn.toFixed(1)} />
          <Readout
            label="survival status"
            value={results.survived ? 'Survived horizon' : 'Terminated early (Torso fell)'}
          />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} title="Forward body displacement over time">
        <Curve name="forward distance x(t)" x={results.timeIndices} y={results.forwardX} slot={0} width={2.5} />
        <Points name="locomotion phase steps" x={results.timeIndices} y={results.forwardX} slot={0} size={3} />
      </Plot>
    </Figure>
  )
}
