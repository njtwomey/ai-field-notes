import { useMemo, useState } from 'react'
import {
  choice,
  Curve,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Player,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { GridView } from 'aifn-render/gym'
import { greedyPath, valueIteration } from 'aifn-methods/gym/agents'
import { mazeEnvironment } from 'aifn-methods/gym/environments'
import { stream } from 'aifn/foundation/random'
import { toFlat } from 'aifn/foundation/tensor'
import { trace } from 'aifn/foundation/trace'
import { MAZE_OPTIONS, type MazeName } from './mazes'

const MAX_SWEEPS = 300

/**
 * The same mazes solved by planning: value iteration on the known model (aifn `valueIteration` on the tabular model of
 * `mazeEnvironment`), one Bellman optimality sweep per position of the player, drawn by the shared `GridView`.
 */
export function MazePlanner() {
  const state = useFigureState({
    maze: choice<MazeName>(MAZE_OPTIONS, 'routes', { label: 'maze' }),
    gamma: slider(0.8, 0.99, 0.95, { step: 0.01, label: 'discount γ' }),
    slip: slider(0, 0.3, 0, { step: 0.01, label: 'slip probability' }),
  })
  const { maze, gamma, slip } = state
  const env = useMemo(() => mazeEnvironment({ layout: maze, gamma, slip }), [maze, gamma, slip])
  const start = useMemo(() => env.reset(stream(0)).state, [env])

  // Every sweep from V₀ = 0: values, the greedy policy (no arrow where all actions tie) and the largest change.
  const sweeps = useMemo(() => {
    const run = trace(valueIteration(env.model, { tolerance: 1e-9 }), undefined, MAX_SWEEPS)
    const A = env.model.actions
    return run.steps.map((s, k) => {
      const V = Float64Array.from(toFlat(s.V))
      const q = toFlat(s.Q)
      const policy = Int32Array.from(toFlat(s.policy), (a, st) => {
        if (a < 0) return -1
        for (let b = 1; b < A; b++) if (q[st * A + b] !== q[st * A]) return a
        return -1
      })
      const prev = k > 0 ? toFlat(run.steps[k - 1].V) : null
      const residual = prev ? Math.max(...V.map((v, i) => Math.abs(v - prev[i]))) : NaN
      return { V, policy, residual }
    })
  }, [env])
  const total = sweeps.length - 1
  const vStar = sweeps[total].V[start]

  const [position, setPosition] = useState({ sweeps, k: 0 })
  const k = position.sweeps === sweeps ? Math.min(position.k, total) : 0
  const go = (v: number) => setPosition({ sweeps, k: Math.min(total, Math.max(0, Math.round(v))) })
  const { V, policy, residual } = sweeps[k]
  const route = useMemo(() => Array.from(toFlat(greedyPath(env.model, start, policy))), [env, start, policy])
  const range = useMemo((): [number, number] => {
    const final = sweeps[total].V
    const lo = Math.min(...final.filter((_, s) => !env.model.terminal[s]))
    return [Math.floor(lo), 10]
  }, [sweeps, total, env])
  const field = useMemo(
    () => ({ values: V, label: 'V_k', range, scale: 'sequential' as const, fillOpacity: 0.7 }),
    [V, range],
  )

  const curve = useMemo(() => {
    const ks = sweeps.map((_, i) => i)
    return { x: ks, y: ks.map((i) => sweeps[i].V[start]) }
  }, [sweeps, start])
  const kx = useAxis({ label: 'sweep k', range: [0, Math.max(10, total)], integer: true })
  const vy = useAxis({ label: 'value of the start', hold: 'union', key: `${maze}:${gamma}:${slip}` })

  return (
    <Figure
      title="Solving a maze by value iteration"
      state={state}
      caption={
        <>
          The model is known, so each position is one sweep of the Bellman optimality backup over every open cell, from
          V₀ = 0. Colour is V<sub>k</sub>, arrows the greedy policy (no arrow while a cell's actions tie), and the line
          of arrows is the route the greedy policy intends from the start. Rewards and slip are as in the figure above.
          Below: V<sub>k</sub>(start) against the sweep, with v*(start) dashed; drag its line or play to choose k.
        </>
      }
      controls={<Player value={k} onChange={go} count={total + 1} label="sweep" format={(v) => `sweep ${v}`} />}
      readouts={
        <>
          <Readout label="sweeps to converge" value={String(total)} />
          <Readout label="largest change at k" value={k === 0 ? '–' : formatNumber(residual)} />
          <Readout label="V_k(start)" value={formatNumber(V[start])} />
          <Readout label="v*(start)" value={formatNumber(vStar)} />
        </>
      }
    >
      <GridView
        render={env.render!}
        value={field}
        policy={policy}
        path={route}
        inkLatest={false}
        agent={false}
        height={Math.round(Math.min(480, Math.max(260, (env.render!.height / env.render!.width) * 560 + 60)))}
        ariaLabel="Maze with the value of each cell after k sweeps, the greedy policy and its route"
      />
      <Plot x={kx} y={vy} height={220} ariaLabel="Value of the start after each sweep">
        <Curve name="V_k(start)" x={curve.x} y={curve.y} slot={0} />
        <Curve name="v*(start)" x={[0, Math.max(10, total)]} y={[vStar, vStar]} dashed slot={1} />
        <Handle kind="x" at={k} label={String(k)} onDrag={go} />
      </Plot>
    </Figure>
  )
}
