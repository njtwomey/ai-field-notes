import { useMemo } from 'react'
import {
  epsilonStepsFor,
  SB3_CARTPOLE,
  SB3_CARTPOLE_EPSILON_FRACTION,
  SB3_CARTPOLE_STEPS,
} from 'aifn-methods/gym/agents'
import { choice, float, int, row, setting, slider, useFigureState, type AnyValues } from 'aifn-render/state'
import { Button } from 'aifn-render/ui/button'
import { gymSetup } from 'aifn-methods/gym'
import { GymTrainer, trainingRun } from 'aifn-render/gym'

/** The cart-pole agents by registry key, with their labels. */
const AGENTS = {
  crossEntropyAgent: { label: 'cross-entropy method' },
  dqnAgent: { label: 'DQN' },
  lqrBangBangAgent: { label: 'LQR bang-bang' },
  randomAgent: { label: 'random' },
} as const
type AgentKey = keyof typeof AGENTS

/**
 * The RL Baselines3 Zoo's DQN recipe for CartPole-v1 (`SB3_CARTPOLE`, 50 000 steps) as page fields, with ε's decay as a
 * fraction of the step budget as SB3 states it (the budget itself is the training-run row's). It is the page's
 * default; the network fields are separate and stay 32 × 2.
 */
const PRESET = {
  learningRate: SB3_CARTPOLE.learningRate,
  updateEvery: SB3_CARTPOLE.updateEvery,
  gradientSteps: SB3_CARTPOLE.gradientSteps,
  batchSize: SB3_CARTPOLE.batchSize,
  targetSync: SB3_CARTPOLE.targetSync,
  bufferSize: SB3_CARTPOLE.bufferSize,
  double: SB3_CARTPOLE.double,
  gamma: SB3_CARTPOLE.gamma,
  warmup: SB3_CARTPOLE.warmup,
  epsilonEnd: SB3_CARTPOLE.epsilonEnd,
  epsilonFraction: SB3_CARTPOLE_EPSILON_FRACTION,
  clipNorm: SB3_CARTPOLE.clipNorm,
} as const
const ACTIVATIONS = ['relu', 'tanh', 'gelu', 'elu', 'silu'] as const
const isDqn = (v: AnyValues) => (v.setup as { agent?: string } | undefined)?.agent === 'dqnAgent'

export function CartPoleTrainerSpecimen() {
  const state = useFigureState({
    setup: row('1 · environment and agent', {
      agent: choice(
        Object.entries(AGENTS).map(([value, a]) => ({ value, label: a.label })),
        'crossEntropyAgent',
        { label: 'agent' },
      ),
      jitter: slider(0, 0.2, 0.05, { label: 'initial jitter', step: 0.01 }),
      population: int(20, {
        label: 'CEM population',
        ge: 4,
        le: 1000,
        suggestions: [10, 20, 50, 100],
        when: (v) => v.agent === 'crossEntropyAgent',
      }),
    }),
    dqn: {
      ...row('2 · DQN', {
        learningRate: float(PRESET.learningRate, {
          label: 'learning rate',
          gt: 0,
          scale: 'log10',
          suggestions: [1e-4, 3e-4, 1e-3, 2.3e-3],
        }),
        updateEvery: int(PRESET.updateEvery, { label: 'train every (steps)', ge: 1, suggestions: [1, 4, 10, 256] }),
        gradientSteps: int(PRESET.gradientSteps, {
          label: 'gradient steps per round',
          ge: 1,
          le: 4096,
          suggestions: [1, 16, 64, 128],
        }),
        batchSize: int(PRESET.batchSize, { label: 'batch size', ge: 1, le: 4096, suggestions: [32, 64, 128, 256] }),
        targetSync: int(PRESET.targetSync, { label: 'target sync (steps)', ge: 1, suggestions: [10, 100, 500, 1000] }),
        bufferSize: int(PRESET.bufferSize, { label: 'buffer size', ge: 1, suggestions: [10_000, 50_000, 100_000] }),
        double: setting(PRESET.double, 'double DQN'),
        advanced: setting(false, 'advanced'),
      }),
      when: isDqn,
    },
    advanced: {
      ...row('2b · DQN, advanced', {
        gamma: float(PRESET.gamma, { label: 'γ', ge: 0, le: 1, suggestions: [0.95, 0.98, 0.99, 0.995] }),
        warmup: int(PRESET.warmup, { label: 'learning starts (steps)', ge: 0, suggestions: [0, 500, 1000, 5000] }),
        epsilonEnd: float(PRESET.epsilonEnd, { label: 'final ε', ge: 0, le: 1, suggestions: [0.01, 0.04, 0.05, 0.1] }),
        epsilonFraction: float(PRESET.epsilonFraction, {
          label: 'ε decay (fraction of budget)',
          ge: 0,
          le: 1,
          suggestions: [0.1, 0.16, 0.25, 0.5],
        }),
        clipNorm: float(PRESET.clipNorm, { label: 'gradient clip (norm)', gt: 0, suggestions: [1, 10, 100] }),
      }),
      when: (v: AnyValues) => isDqn(v) && !!(v.dqn as { advanced?: boolean } | undefined)?.advanced,
    },
    network: {
      ...row('3 · network', {
        depth: slider(1, 4, 2, { label: 'hidden layers', step: 1 }),
        width: choice([16, 32, 64, 128, 256], 32, { label: 'width' }),
        activation: choice(ACTIVATIONS, 'relu', { label: 'activation' }),
        layerNorm: setting(false, 'layer norm'),
      }),
      when: isDqn,
    },
    run: trainingRun({ steps: SB3_CARTPOLE_STEPS, label: '4 · training run' }),
  })
  const key = state.setup.agent as AgentKey
  const { jitter, population } = state.setup
  const dqn = state.dqn
  const isDqnKey = key === 'dqnAgent'
  // SB3 states ε's decay as a fraction of the step budget; under an episode budget the steps field's value is used.
  const budget = state.run.steps
  const { learningRate, updateEvery, gradientSteps, batchSize, targetSync, bufferSize, double } = dqn
  const { gamma, warmup, epsilonEnd, epsilonFraction, clipNorm } = state.advanced
  const epsilonSteps = epsilonStepsFor(epsilonFraction, budget)
  const { depth, width, activation, layerNorm } = state.network
  const network = `4 → ${Array.from({ length: depth }, () => `${width} → `).join('')}2, ${activation}${layerNorm ? ' with layer norm' : ''}`
  const setup = useMemo(() => {
    const envParams = { jitter }
    const agentParams =
      key === 'crossEntropyAgent'
        ? { population }
        : key === 'dqnAgent'
          ? {
              learningRate,
              updateEvery,
              gradientSteps,
              batchSize,
              epsilonSteps,
              epsilonEnd,
              targetSync,
              bufferSize,
              double,
              gamma,
              warmup,
              clipNorm,
              hidden: new Array<number>(depth).fill(width),
              activation,
              layerNorm,
            }
          : {}
    return gymSetup('cartPoleEnvironment', envParams, key, agentParams)
  }, [
    gradientSteps,
    epsilonEnd,
    gamma,
    warmup,
    clipNorm,
    key,
    jitter,
    population,
    learningRate,
    updateEvery,
    batchSize,
    epsilonSteps,
    targetSync,
    bufferSize,
    double,
    depth,
    width,
    activation,
    layerNorm,
  ])
  return (
    <GymTrainer
      title="Environment × agent: a cart-pole"
      purpose={`Gymnasium's CartPole-v1 on the environment protocol: two actions, +1 per step, terminated when the pole passes 12° or the cart leaves the track, truncated at 500. The cross-entropy method learns a linear threshold policy from returns alone; DQN learns action values with a ${network} network from replayed transitions; LQR bang-bang balances from the model's autodiff linearisation without learning.`}
      state={state}
      setup={setup}
      // 120 steps a second is about 2.4× real time at τ = 0.02 s, so a 500-step episode plays in about 4 s.
      playbackSpeed={120}
      scalars={key === 'dqnAgent' ? DQN_SCALARS : undefined}
      actions={
        isDqnKey && (
          <Button
            size="sm"
            variant="outline"
            aria-label="SB3 RL Zoo preset"
            onClick={() => {
              state.set('run.budget', 'steps')
              state.set('run.steps', SB3_CARTPOLE_STEPS)
              for (const [k, v] of Object.entries(PRESET))
                state.set(
                  k === 'gamma' || k === 'warmup' || k.startsWith('epsilon') || k === 'clipNorm'
                    ? `advanced.${k}`
                    : `dqn.${k}`,
                  v,
                )
            }}
          >
            SB3 RL Zoo preset
          </Button>
        )
      }
      caption={`aifn cartPoleEnvironment (Euler, τ = 0.02 s, force ±10 N, start jitter ±${jitter}) and ${AGENTS[key].label}${caption(key, population, { ...dqn, ...state.advanced, budget, epsilonSteps, network })}.`}
    />
  )
}

const DQN_SCALARS = ['ε', 'loss', 'mean max Q'] as const

/** The agent's settings, for the caption. */
function caption(
  key: AgentKey,
  population: number,
  dqn: {
    learningRate: number
    budget: number
    updateEvery: number
    gradientSteps: number
    batchSize: number
    targetSync: number
    bufferSize: number
    epsilonSteps: number
    epsilonEnd: number
    gamma: number
    warmup: number
    clipNorm: number
    double: boolean
    network: string
  },
) {
  if (key === 'crossEntropyAgent')
    return ` (population ${population}, elite 20 %, one episode per member; the greedy policy is the mean weight vector)`
  if (key === 'dqnAgent')
    return ` (${dqn.double ? 'double DQN' : 'DQN'} with a ${dqn.network} Q-network, on Stable-Baselines3's schedule (budget as set below), random actions for the first ${dqn.warmup}, then ε-greedy with ε from 1 to ${dqn.epsilonEnd} over ${dqn.epsilonSteps.toLocaleString()} steps (that fraction of a ${dqn.budget.toLocaleString()}-step budget); every ${dqn.updateEvery} steps ${dqn.gradientSteps} Adam steps at ${dqn.learningRate} on the Huber loss of batches of ${dqn.batchSize} (γ ${dqn.gamma}, gradients clipped to norm ${dqn.clipNorm}), from the last ${dqn.bufferSize.toLocaleString()} transitions; the target network copied every ${dqn.targetSync} steps. Fewer, larger training rounds cost the same per gradient step. ε is its value at each episode's end; the loss and the mean max Q at the sampled next states are averaged over the episode's updates. DQN tends to learn and then partly forget, here and in a PyTorch implementation of the same recipe)`
  return ''
}
