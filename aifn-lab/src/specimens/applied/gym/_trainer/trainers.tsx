import { useMemo } from 'react'
import {
  epsilonGreedy,
  exp3,
  expectedSarsaAgent,
  klUcb,
  monteCarloControlAgent,
  nStepSarsaAgent,
  qLearningAgent,
  randomAgent,
  reinforceAgent,
  sarsaAgent,
  thompsonBernoulli,
  ucb1,
} from 'aifn-applied/gym/agents'
import { bernoulliBandit, mazeEnvironment } from 'aifn-applied/gym/environments'
import type { Agent, Environment } from 'aifn/foundation/contracts'
import { call, choice, number, row, slider, useFigureState } from '@lab/state'
import { GymTrainer, type GymSetup } from '@lab/views'

type AnyEnv = Environment<unknown, unknown, unknown>
type AnyAgent = Agent<unknown, unknown, unknown>

/** The maze agents by registry key, with the page-side factory (the worker builds the same from `gym/agents/<key>`). */
const MAZE_AGENTS = {
  qLearningAgent: { label: 'Q-learning', make: qLearningAgent },
  sarsaAgent: { label: 'SARSA', make: sarsaAgent },
  expectedSarsaAgent: { label: 'expected SARSA', make: expectedSarsaAgent },
  nStepSarsaAgent: { label: '4-step SARSA', make: nStepSarsaAgent },
  monteCarloControlAgent: { label: 'Monte Carlo control', make: monteCarloControlAgent },
  reinforceAgent: { label: 'REINFORCE', make: reinforceAgent },
  randomAgent: { label: 'random', make: randomAgent },
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
        Object.entries(MAZE_AGENTS).map(([value, a]) => ({ value, label: a.label })),
        'qLearningAgent',
        { label: 'agent' },
      ),
      epsilon: slider(0, 0.5, 0.1, { label: 'exploration ε' }),
      alpha: slider(0.05, 1, 0.5, { label: 'step size α' }),
    }),
    run: row('2 · training run', {
      episodes: choice([100, 300, 1000, 3000], 300, { label: 'episodes' }),
      seed: number(1, { label: 'seed', min: 0, max: 9999, step: 1 }),
    }),
  })
  const { layout, slip, epsilon, alpha } = state.setup
  const key = state.setup.agent as MazeAgentKey
  const { episodes, seed } = state.run
  const setup = useMemo((): GymSetup => {
    const envParams = { layout, slip }
    const agentParams =
      key === 'randomAgent'
        ? {}
        : key === 'reinforceAgent'
          ? { learningRate: alpha / 5 }
          : key === 'monteCarloControlAgent'
            ? { epsilon }
            : { epsilon, learningRate: alpha }
    const make = MAZE_AGENTS[key].make as (p: object) => AnyAgent
    return {
      env: mazeEnvironment(envParams as never) as AnyEnv,
      agent: make(agentParams),
      envTask: call('gym/environments/mazeEnvironment', envParams),
      agentTask: call(`gym/agents/${key}`, agentParams),
      episodes,
      seed,
    }
  }, [layout, slip, key, epsilon, alpha, episodes, seed])
  return (
    <GymTrainer
      title="Train in the maze, then play any episode"
      purpose="Training runs headless in the worker and streams its learning curve; picking an episode on the curve replays that episode as it happened, or evaluates the greedy policy as it stood then."
      state={state}
      setup={setup}
      caption={`aifn mazeEnvironment('${layout}') (goal +10, trap −20 and back to start, step −1) and ${MAZE_AGENTS[key].label}.`}
    />
  )
}

const BANDIT_AGENTS = {
  ucb1: { label: 'UCB1', make: ucb1 },
  klUcb: { label: 'KL-UCB', make: klUcb },
  thompsonBernoulli: { label: 'Thompson sampling', make: thompsonBernoulli },
  epsilonGreedy: { label: 'ε-greedy', make: epsilonGreedy },
  exp3: { label: 'EXP3', make: exp3 },
} as const
type BanditAgentKey = keyof typeof BANDIT_AGENTS

export function BanditTrainerSpecimen() {
  const state = useFigureState({
    arms: row('1 · Bernoulli arms and policy', {
      m1: slider(0.05, 0.95, 0.3, { label: 'μ₁' }),
      m2: slider(0.05, 0.95, 0.5, { label: 'μ₂' }),
      m3: slider(0.05, 0.95, 0.6, { label: 'μ₃' }),
      agent: choice(
        Object.entries(BANDIT_AGENTS).map(([value, a]) => ({ value, label: a.label })),
        'ucb1',
        { label: 'policy' },
      ),
    }),
    run: row('2 · training run', {
      episodes: choice([300, 1000, 3000], 1000, { label: 'rounds' }),
      seed: number(1, { label: 'seed', min: 0, max: 9999, step: 1 }),
    }),
  })
  const { m1, m2, m3 } = state.arms
  const key = state.arms.agent as BanditAgentKey
  const { episodes, seed } = state.run
  const setup = useMemo((): GymSetup => {
    const envParams = { means: [m1, m2, m3] }
    const make = BANDIT_AGENTS[key].make as (p: object) => AnyAgent
    return {
      env: bernoulliBandit(envParams) as AnyEnv,
      agent: make({}),
      envTask: call('gym/environments/bernoulliBandit', envParams),
      agentTask: call(`gym/agents/${key}`, {}),
      episodes,
      seed,
    }
  }, [m1, m2, m3, key, episodes, seed])
  return (
    <GymTrainer
      title="Train a bandit policy, then inspect any round"
      purpose="Each episode of a bandit is one pull; the learning curve is the reward per round and the cumulative regret, and picking a round shows the pulls up to it and the arm pulled then."
      state={state}
      setup={setup}
      caption={`aifn bernoulliBandit with means ${m1}, ${m2}, ${m3} and ${BANDIT_AGENTS[key].label}; regret from the environment's oracle.`}
    />
  )
}
