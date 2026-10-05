import { useMemo } from 'react'
import { choice, float, row, slider, useFigureState } from 'aifn-render/state'
import { gymSetup } from 'aifn-methods/gym'
import { GymTrainer, trainingRun } from 'aifn-render/gym'

/** The maze agents by registry key, with their labels. */
const MAZE_AGENTS = {
  qLearningAgent: 'Q-learning',
  sarsaAgent: 'SARSA',
  expectedSarsaAgent: 'expected SARSA',
  nStepSarsaAgent: '4-step SARSA',
  monteCarloControlAgent: 'Monte Carlo control',
  reinforceAgent: 'REINFORCE',
  randomAgent: 'random',
} as const
type MazeAgentKey = keyof typeof MAZE_AGENTS

export function MazeTrainerSpecimen() {
  const state = useFigureState({
    setup: row('1 · environment and agent', {
      layout: choice(
        [
          { value: 'small', label: 'small 5 × 5' },
          { value: 'classic', label: 'classic 10 × 6' },
          { value: 'traps', label: 'traps 8 × 5' },
        ],
        'classic',
        { label: 'maze' },
      ),
      slip: slider(0, 0.4, 0, { label: 'slip probability' }),
      agent: choice(
        Object.entries(MAZE_AGENTS).map(([value, label]) => ({ value, label })),
        'qLearningAgent',
        { label: 'agent' },
      ),
      epsilon: float(0.1, { label: 'exploration ε', ge: 0, le: 1, step: 0.05, suggestions: [0.01, 0.05, 0.1, 0.2] }),
      alpha: float(0.5, { label: 'step size α', gt: 0, le: 1, step: 0.05, suggestions: [0.05, 0.1, 0.5, 1] }),
    }),
    run: trainingRun({ episodes: 300, label: '2 · training run' }),
  })
  const { layout, slip, epsilon, alpha } = state.setup
  const key = state.setup.agent as MazeAgentKey
  const setup = useMemo(() => {
    const envParams = { layout, slip }
    const agentParams =
      key === 'randomAgent'
        ? {}
        : key === 'reinforceAgent'
          ? { learningRate: alpha / 5 }
          : key === 'monteCarloControlAgent'
            ? { epsilon }
            : { epsilon, learningRate: alpha }
    return gymSetup('mazeEnvironment', envParams, key, agentParams)
  }, [layout, slip, key, epsilon, alpha])
  return (
    <GymTrainer
      title="Train in the maze, then play any episode"
      purpose="Training runs headless in the worker and streams its learning curve; picking an episode on the curve replays that episode as it happened, or evaluates the greedy policy as it stood then."
      state={state}
      setup={setup}
      caption={`aifn mazeEnvironment('${layout}') (goal +10, trap −20 and back to start, step −1) and ${MAZE_AGENTS[key]}.`}
    />
  )
}

const BANDIT_AGENTS = {
  ucb1: 'UCB1',
  klUcb: 'KL-UCB',
  thompsonBernoulli: 'Thompson sampling',
  epsilonGreedy: 'ε-greedy',
  exp3: 'EXP3',
} as const
type BanditAgentKey = keyof typeof BANDIT_AGENTS

export function BanditTrainerSpecimen() {
  const state = useFigureState({
    arms: row('1 · Bernoulli arms and policy', {
      m1: slider(0.05, 0.95, 0.3, { label: 'μ₁' }),
      m2: slider(0.05, 0.95, 0.5, { label: 'μ₂' }),
      m3: slider(0.05, 0.95, 0.6, { label: 'μ₃' }),
      agent: choice(
        Object.entries(BANDIT_AGENTS).map(([value, label]) => ({ value, label })),
        'ucb1',
        { label: 'policy' },
      ),
    }),
    run: trainingRun({ episodes: 1000, unit: 'rounds', label: '2 · training run' }),
  })
  const { m1, m2, m3 } = state.arms
  const key = state.arms.agent as BanditAgentKey
  const setup = useMemo(() => gymSetup('bernoulliBandit', { means: [m1, m2, m3] }, key, {}), [m1, m2, m3, key])
  return (
    <GymTrainer
      title="Train a bandit policy, then inspect any round"
      purpose="Each episode of a bandit is one pull; the learning curve is the reward per round and the cumulative regret, and picking a round shows the pulls up to it and the arm pulled then."
      state={state}
      setup={setup}
      caption={`aifn bernoulliBandit with means ${m1}, ${m2}, ${m3} and ${BANDIT_AGENTS[key]}; regret from the environment's oracle.`}
    />
  )
}
