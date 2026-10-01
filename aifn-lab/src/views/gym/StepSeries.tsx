/**
 * The chosen episode against step, under `GymTrainer`'s Player: the state series of its render kind (`GYM_SERIES`) and
 * the actions (a strip of names for a discrete action, a line per element of a box), with a draggable cursor at the
 * played step. Step k's action is the one taken from state k.
 */
import { useMemo } from 'react'
import type { Trajectory } from 'aifn-applied/gym'
import type { Environment } from 'aifn/foundation/contracts'
import { Curve, Handle, Plot, Plots, Raster, useAxis } from '@lab/viz'
import { actionSeries, GYM_SERIES, type StateSeries } from './series'

const NONE: readonly StateSeries[] = []

export type StepSeriesProps = {
  env: Environment<unknown, unknown, unknown>
  kind: string
  trajectory: Trajectory<unknown, unknown, unknown>
  step: number
  onStep: (step: number) => void
  /** The share of the figure's height. */
  scale: number
}

export function StepSeries({ env, kind, trajectory, step, onStep, scale }: StepSeriesProps) {
  const states = GYM_SERIES[kind] ?? NONE
  const actions = useMemo(() => actionSeries(env), [env])
  const data = useMemo(() => {
    const xs = trajectory.states.map((_, i) => i)
    const xa = trajectory.actions.map((_, i) => i)
    return {
      xs,
      xa,
      states: states.map((s) => trajectory.states.map(s.value)),
      actions: actions.map((a) =>
        a.kind === 'strip'
          ? [trajectory.actions.map((v) => v as number)]
          : trajectory.actions.map((v) => (v as ArrayLike<number>)[a.index]),
      ),
    }
  }, [trajectory, states, actions])
  const sa = useAxis({ label: 'step' })
  // One y axis per panel: at most four panels (two state series, two action series) in practice.
  const y0 = useAxis({ label: states[0]?.name ?? '' })
  const y1 = useAxis({ label: states[1]?.name ?? '' })
  const y2 = useAxis({ label: states[2]?.name ?? '' })
  const ya = [
    useAxis({ label: actions[0]?.name ?? 'action', ...(actions[0]?.kind === 'strip' && { categories: [''] }) }),
  ]
  const ys = [y0, y1, y2]
  const cursor = <Handle kind="x" at={step} label="step" onDrag={(v) => onStep(Math.max(0, Math.round(v)))} />
  const panels = [
    ...states.slice(0, 3).map((s, i) => (
      <Plot key={s.name} x={sa} y={ys[i]} legend={false}>
        <Curve name={s.name} x={data.xs} y={data.states[i]} slot={i} />
        {cursor}
      </Plot>
    )),
    ...actions.slice(0, 1).map((a, i) =>
      a.kind === 'strip' ? (
        <Plot key={a.name} x={sa} y={ya[0]} legend={false}>
          <Raster x={data.xa} y={[0]} z={data.actions[i] as number[][]} scale="categorical" categoryNames={a.names} />
          {cursor}
        </Plot>
      ) : (
        <Plot key={a.name} x={sa} y={ya[0]} legend={false}>
          <Curve name={a.name} x={data.xa} y={data.actions[i] as number[]} slot={3} />
          {cursor}
        </Plot>
      ),
    ),
  ]
  return (
    <Plots
      rows={panels.length}
      cols={1}
      tight
      scale={scale}
      heights={panels.map((_, i) => (i < states.length || actions[0]?.kind === 'line' ? 1 : 0.6))}
    >
      {panels}
    </Plots>
  )
}
