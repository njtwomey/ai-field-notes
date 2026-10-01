/**
 * Environment renderers for `GymTrainer`, one per `render.kind`: each draws the chosen episode at a step as one `Plot`.
 * `grid` draws a gridworld or maze (cells by kind, the path so far, the agent); `bandit` (an environment without a
 * render whose episodes are single pulls) draws the pulls per arm up to the chosen episode. Add a kind to
 * `GYM_RENDERERS` (`registry.ts`). `pendulum` and `cartpole` draw
 * classic control with `PendulumView` and `CartPoleView`.
 */
import { useMemo, type ComponentType } from 'react'
import type { Trajectory, Training } from 'aifn-applied/gym'
import type { CartPoleRender, Environment, EpisodeEnd, GridRender, PendulumRender } from 'aifn/foundation/contracts'
import { Bars, Curve, Plot, Points, Raster, useAxis } from '@lab/viz'
import { CartPoleView } from './CartPoleView'
import { PendulumView, type PendulumViewProps } from './PendulumView'

/** What a renderer receives: the environment, the chosen episode's trajectory and step, and the training run. */
export type GymRenderProps = {
  env: Environment<unknown, unknown, unknown>
  trajectory: Trajectory<unknown, unknown, unknown>
  /** Index into `trajectory.states` (0 is the reset). */
  step: number
  training: Training<unknown>
  /** The chosen episode (1-based). */
  episode: number
  /** Page-specific options of a renderer (the pendulum's draggable `start`). */
  options?: Readonly<Record<string, unknown>>
  /** At the episode's last step, how it ended: the scene is drawn in the destructive or the success tone. */
  end?: EpisodeEnd | null
}

/** The tone of a scene at an ending: destructive for a failure, success for a success. */
const toneOf = (end: EpisodeEnd | null | undefined): 'destructive' | 'success' | undefined =>
  end ? (end.success ? 'success' : 'destructive') : undefined

export type GymRenderer = ComponentType<GymRenderProps>

/** Cell kinds drawn in colour, in slot order; other cells are blank. */
const KINDS = ['wall', 'goal', 'trap', 'start', 'hole', 'cliff', 'terminal'] as const

export function GridRenderer({ env, trajectory, step, end }: GymRenderProps) {
  const r = env.render as GridRender<unknown>
  const { width, height } = r
  const xs = useMemo(() => Array.from({ length: width }, (_, i) => i), [width])
  const ys = useMemo(() => Array.from({ length: height }, (_, i) => i), [height])
  const map = useMemo(
    () =>
      Array.from({ length: height }, (_, y) =>
        Array.from({ length: width }, (_, x) => KINDS.indexOf(r.cells[y * width + x] as (typeof KINDS)[number])),
      ),
    [r, width, height],
  )
  const cells = useMemo(
    () => trajectory.states.map((s) => r.cell(s)).map((c) => [c % width, Math.floor(c / width)] as const),
    [trajectory, r, width],
  )
  const upTo = cells.slice(0, step + 1)
  const [px, py] = cells[Math.min(step, cells.length - 1)]
  const xa = useAxis({ label: 'x' })
  const ya = useAxis({ label: 'y', equal: xa })
  return (
    <Plot x={xa} y={ya} title={end ? `${env.name}: ${end.reason}` : env.name}>
      <Raster x={xs} y={ys} z={map} scale="categorical" categoryNames={KINDS} />
      <Curve name="path" x={upTo.map((c) => c[0])} y={upTo.map((c) => c[1])} slot={1} tone={toneOf(end)} showPoints />
      <Points name="agent" x={[px]} y={[py]} emphasis={!end} tone={toneOf(end)} size={end ? 16 : undefined} />
    </Plot>
  )
}

export function BanditRenderer({ env, trajectory, step, training, episode }: GymRenderProps) {
  const arms = env.action.kind === 'discrete' ? env.action.n : 0
  const names = useMemo(
    () => (env.action.kind === 'discrete' && env.action.names) || Array.from({ length: arms }, (_, a) => `${a + 1}`),
    [env, arms],
  )
  const x = useMemo(() => Array.from({ length: arms }, (_, a) => a), [arms])
  const pulls = useMemo(() => {
    const c = new Array<number>(arms).fill(0)
    for (let i = 0; i < Math.min(episode, training.firstAction.length); i++) c[training.firstAction[i]] += 1
    return c
  }, [arms, episode, training])
  const arm = step > 0 ? (trajectory.actions[0] as number) : -1
  const xa = useAxis({ label: 'arm', categories: names })
  const ya = useAxis({ label: 'pulls so far', hold: 'union' })
  return (
    <Plot x={xa} y={ya} title={`pulls after episode ${episode}`}>
      <Bars name="pulls" x={x} y={pulls} slot={0} width={0.6} />
      {arm >= 0 && <Points name="this pull" x={[arm]} y={[pulls[arm]]} emphasis />}
    </Plot>
  )
}

/** A pendulum at the step, with the bob's path over the previous 20 steps. */
export function PendulumRenderer({ env, trajectory, step, options, end }: GymRenderProps) {
  const at = Math.min(step, trajectory.states.length - 1)
  const trail = useMemo(() => trajectory.states.slice(Math.max(0, at - 20), at + 1), [trajectory, at])
  return (
    <PendulumView
      render={env.render as PendulumRender<unknown>}
      state={trajectory.states[at]}
      trail={trail}
      title={env.name}
      start={options?.start as PendulumViewProps<unknown>['start']}
      tone={toneOf(end)}
    />
  )
}

/** A cart-pole at the step. */
export function CartPoleRenderer({ env, trajectory, step, end }: GymRenderProps) {
  const at = Math.min(step, trajectory.states.length - 1)
  return (
    <CartPoleView
      render={env.render as CartPoleRender<unknown>}
      state={trajectory.states[at]}
      title={end ? `${env.name}: ${end.reason}` : env.name}
      tone={toneOf(end)}
    />
  )
}
