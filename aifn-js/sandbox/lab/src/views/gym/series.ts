/**
 * Per-step series for `GymTrainer`: what to plot against step under the Player for the chosen episode. Each render
 * kind declares its state series in `GYM_SERIES`; the action series follow from the action domain (a discrete action
 * is a strip of its names, a box action a line per element).
 */
import { wrapAngle } from 'aifn-methods/gym/environments/control'
import type { Environment } from 'aifn/foundation/contracts'

/** A numeric series of the environment state, one value per state of the trajectory. */
export type StateSeries = { name: string; value: (state: unknown) => number }

const field = (k: string) => (s: unknown) => (s as Record<string, number>)[k]

/** The state series per `render.kind`; kinds without an entry plot actions only. */
export const GYM_SERIES: Record<string, readonly StateSeries[]> = {
  pendulum: [
    { name: 'θ (rad)', value: (s) => wrapAngle(field('theta')(s)) },
    { name: 'θ̇ (rad/s)', value: field('thetaDot') },
  ],
  cartpole: [
    { name: 'x (m)', value: field('x') },
    { name: 'θ (rad)', value: field('theta') },
  ],
}

/** The action series of an environment: a strip for a discrete action, else one line per element of a box. */
export type ActionSeries =
  { kind: 'strip'; name: string; names: readonly string[] } | { kind: 'line'; name: string; index: number }

export function actionSeries(env: Environment<unknown, unknown, unknown>): ActionSeries[] {
  const a = env.action
  if (a.kind === 'discrete')
    return [{ kind: 'strip', name: 'action', names: a.names ?? Array.from({ length: a.n }, (_, i) => `${i}`) }]
  return a.low.map((_, i) => ({ kind: 'line', name: a.names?.[i] ?? `action ${i}`, index: i }))
}

/** Whether `GymTrainer` draws per-step series for this environment (state series, or a non-trivial episode). */
export const hasStepSeries = (env: Environment<unknown, unknown, unknown>, kind: string): boolean =>
  (GYM_SERIES[kind]?.length ?? 0) > 0 && env.horizon > 1
