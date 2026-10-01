import { useMemo } from 'react'
import { crossEntropyAgent, lqrBangBangAgent, randomAgent } from 'aifn-applied/gym/agents'
import { cartPoleEnvironment } from 'aifn-applied/gym/environments'
import type { Agent, Environment } from 'aifn/foundation/contracts'
import { call, choice, number, row, slider, useFigureState } from '@lab/state'
import { GymTrainer, type GymSetup } from '@lab/views'

type AnyEnv = Environment<unknown, unknown, unknown>
type AnyAgent = Agent<unknown, unknown, unknown>

/** The cart-pole agents by registry key, with the page-side factory (the worker builds the same by address). */
const AGENTS = {
  crossEntropyAgent: { label: 'cross-entropy method', make: crossEntropyAgent, address: 'gym/agents/control' },
  lqrBangBangAgent: { label: 'LQR bang-bang', make: lqrBangBangAgent, address: 'gym/agents/control' },
  randomAgent: { label: 'random', make: randomAgent, address: 'gym/agents' },
} as const
type AgentKey = keyof typeof AGENTS

export function CartPoleTrainerSpecimen() {
  const state = useFigureState({
    setup: row('1 · environment and agent', {
      agent: choice(
        Object.entries(AGENTS).map(([value, a]) => ({ value, label: a.label })),
        'crossEntropyAgent',
        { label: 'agent' },
      ),
      jitter: slider(0, 0.2, 0.05, { label: 'initial jitter', step: 0.01 }),
      population: slider(4, 60, 20, { label: 'CEM population', step: 1, when: (v) => v.agent === 'crossEntropyAgent' }),
    }),
    run: row('2 · training run', {
      episodes: choice([200, 400, 800, 1600], 800, { label: 'episodes' }),
      seed: number(1, { label: 'seed', min: 0, max: 9999, step: 1 }),
    }),
  })
  const key = state.setup.agent as AgentKey
  const { jitter, population } = state.setup
  const { episodes, seed } = state.run
  const setup = useMemo((): GymSetup => {
    const envParams = { jitter }
    const agentParams = key === 'crossEntropyAgent' ? { population } : {}
    const make = AGENTS[key].make as (p: object) => AnyAgent
    return {
      env: cartPoleEnvironment(envParams) as unknown as AnyEnv,
      agent: make(agentParams),
      envTask: call('gym/environments/control/cartPoleEnvironment', envParams),
      agentTask: call(`${AGENTS[key].address}/${key}`, agentParams),
      episodes,
      seed,
    }
  }, [key, jitter, population, episodes, seed])
  return (
    <GymTrainer
      title="Environment × agent: a cart-pole"
      purpose="Gymnasium's CartPole-v1 on the environment protocol: two actions, +1 per step, terminated when the pole passes 12° or the cart leaves the track, truncated at 500. The cross-entropy method learns a linear threshold policy from returns alone; LQR bang-bang balances from the model's autodiff linearisation without learning."
      state={state}
      setup={setup}
      caption={`aifn cartPoleEnvironment (Euler, τ = 0.02 s, force ±10 N, start jitter ±${jitter}) and ${AGENTS[key].label}${key === 'crossEntropyAgent' ? ` (population ${population}, elite 20 %, one episode per member; the greedy policy is the mean weight vector)` : ''}.`}
    />
  )
}
