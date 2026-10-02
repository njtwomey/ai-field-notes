import { useMemo } from 'react'
import { choice, row, slider, useFigureState } from '@lab/state'
import { GymTrainer, gymEnvironment, gymSetup, trainingRun } from '@lab/views'

type AgentKind = 'swing' | 'lqr' | 'random'

const HANGING = Math.PI

export function PendulumAgentSpecimen() {
  const state = useFigureState({
    setup: row('1 · environment and agent', {
      agent: choice(
        [
          { value: 'swing', label: 'energy swing-up + LQR' },
          { value: 'lqr', label: 'LQR alone' },
          { value: 'random', label: 'random' },
        ],
        'swing',
        { label: 'agent' },
      ),
      actions: choice(
        [
          { value: 'box', label: 'box [−u, u]' },
          { value: '5', label: '5 torque levels' },
        ],
        'box',
        { label: 'actions' },
      ),
      maxTorque: slider(0.5, 3, 2, { label: 'torque limit u (N·m)', step: 0.1 }),
      reset: choice(
        [
          { value: 'hanging', label: 'hanging' },
          { value: 'random', label: 'random (Gymnasium)' },
        ],
        'hanging',
        { label: 'training starts' },
      ),
    }),
    evaluation: row('2 · evaluation', {
      theta0: slider(-Math.PI, Math.PI, HANGING, { label: 'evaluation start θ₀ (rad, 0 upright)', step: 0.01 }),
    }),
    run: trainingRun({ episodes: 8, label: '3 · training run' }),
  })
  const kind = state.setup.agent as AgentKind
  const torques = state.setup.actions === 'box' ? 0 : 5
  const { maxTorque, reset } = state.setup
  const { theta0 } = state.evaluation
  const setup = useMemo(() => {
    const envParams = {
      maxTorque,
      torques,
      ...(reset === 'hanging' && { start: { theta: HANGING, thetaDot: 0 } }),
    }
    const agentParams = kind === 'random' ? {} : { swingUp: kind === 'swing' }
    return gymSetup('pendulumEnvironment', envParams, kind === 'random' ? 'randomAgent' : 'swingUpAgent', agentParams)
  }, [kind, maxTorque, torques, reset])
  // Evaluation plays in the same pendulum from the dragged start θ₀ (not part of training, so no retrain).
  const evaluationEnv = useMemo(
    () => gymEnvironment('pendulumEnvironment', { maxTorque, torques, start: { theta: theta0, thetaDot: 0 } }),
    [maxTorque, torques, theta0],
  )
  const rendererOptions = useMemo(
    () => ({ start: { angle: theta0, onDrag: (angle: number) => state.set('evaluation.theta0', angle) } }),
    [theta0, state],
  )
  return (
    <GymTrainer
      title="Environment × agent: an inverted pendulum"
      purpose="The environment protocol on a continuous control problem: Gymnasium's Pendulum-v1 on core's RK4 solver, with a box action, a differentiable dynamics model and a pendulum render. An LQR gain from autodiff Jacobians of the model balances the rod near the top; an energy-pumping swing-up brings it there from hanging."
      state={state}
      setup={{ ...setup, evaluationEnv }}
      initialMode="evaluate"
      rendererOptions={rendererOptions}
      caption="aifn pendulumEnvironment (dt = 0.05 s, RK4; reward −(θ² + 0.1 θ̇² + 0.001 u²); 200 steps per episode). Evaluate plays the agent from the start θ₀: drag the start marker on the rod (θ = π hangs, θ = 0 is upright). Below: θ (wrapped to [−π, π)), θ̇ and the torque over the episode. The LQR alone cannot lift the rod from hanging; the swing-up hands over to it once |θ| < 0.6."
    />
  )
}
