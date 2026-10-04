import { useMemo, useState } from 'react'
import {
  choice,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  setting,
  type SwitchDef,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { ALGORITHMS, DEFAULT_TUNING, laiRobbinsConstant, simulate, type AlgorithmId, type Tuning } from './bandits'
import { seededRand } from './rand'

const START_MEANS = [0.6, 0.5, 0.45, 0.35, 0.2]
const HORIZONS = ['500', '1000', '2000', '5000'] as const
type Horizon = (typeof HORIZONS)[number]
const ARM_COUNTS = ['2', '3', '5'] as const
type ArmCount = (typeof ARM_COUNTS)[number]
type TuningKey = keyof Tuning

export type BanditSimulatorProps = {
  /** Algorithms switched on when the figure opens. */
  algorithms?: AlgorithmId[]
  /** Algorithms offered at all. Defaults to every algorithm. */
  offer?: AlgorithmId[]
  /** Tuning sliders to show. */
  tuning?: TuningKey[]
  /** Show the Lai-Robbins curve when the figure opens. */
  lowerBound?: boolean
  title?: string
  caption?: string
}

/** Every active algorithm on the same seeded reward draws, keyed by algorithm. Plain values in, so memoisation is exact. */
function simulateAll(
  activeKey: string,
  meansKey: string,
  horizon: number,
  runCount: number,
  seed: number,
  epsilon: number,
  etcM: number,
  decayC: number,
) {
  const ids = (activeKey ? activeKey.split(',') : []) as AlgorithmId[]
  const mu = meansKey.split(',').map(Number)
  const tuning = { epsilon, etcM, decayC }
  return new Map(ids.map((id) => [id, simulate(id, mu, horizon, runCount, seed, tuning, seededRand)] as const))
}

/**
 * A live Bernoulli bandit: cumulative pseudo-regret of several algorithms, averaged over seeded runs, and the share of
 * pulls one algorithm gives each arm. The arms' true means are draggable.
 */
export function BanditSimulator({
  algorithms = ['egreedy', 'ucb1', 'ts'],
  offer = ALGORITHMS.map((a) => a.id),
  tuning: tuningKeys = [],
  lowerBound = false,
  title = 'Bandit simulator',
  caption,
}: BanditSimulatorProps) {
  const [means, setMeans] = useState(START_MEANS)
  const state = useFigureState({
    armCount: choice<ArmCount>(
      ARM_COUNTS.map((v) => ({ value: v, label: v })),
      '3',
      { label: 'arms' },
    ),
    horizon: choice<Horizon>(
      HORIZONS.map((v) => ({ value: v, label: v })),
      '2000',
      { label: 'horizon T' },
    ),
    runs: int(20, { min: 1, max: 50, step: 1, label: 'runs averaged', suggestions: [1, 5, 20, 50] }),
    seed: int(1, { min: 1, max: 30, step: 1, label: 'seed' }),
    epsilon: float(DEFAULT_TUNING.epsilon, {
      min: 0,
      max: 0.5,
      step: 0.01,
      label: 'ε (constant ε-greedy)',
      when: () => tuningKeys.includes('epsilon'),
    }),
    decayC: float(DEFAULT_TUNING.decayC, {
      min: 0.5,
      max: 50,
      step: 0.5,
      label: 'c (decaying ε = min(1, cK/t))',
      format: (v) => String(v),
      when: () => tuningKeys.includes('decayC'),
    }),
    etcM: float(DEFAULT_TUNING.etcM, {
      min: 1,
      max: 300,
      step: 1,
      label: 'm (explore-then-commit pulls per arm)',
      format: (v) => String(v),
      when: () => tuningKeys.includes('etcM'),
    }),
    showBound: setting(lowerBound, 'Lai–Robbins bound'),
    focus: choice<AlgorithmId>(
      ALGORITHMS.filter((a) => offer.includes(a.id)).map((a) => ({ value: a.id, label: a.label })),
      algorithms[algorithms.length - 1] ?? 'ucb1',
      { label: 'bars show', when: (v) => ALGORITHMS.filter((a) => offer.includes(a.id) && v[`on_${a.id}`]).length > 1 },
    ),
    // One switch per offered algorithm.
    ...(Object.fromEntries(
      ALGORITHMS.map((a) => [
        `on_${a.id}`,
        setting(algorithms.includes(a.id), { label: a.label, when: () => offer.includes(a.id) }),
      ]),
    ) as Record<`on_${AlgorithmId}`, SwitchDef>),
  })
  const focus = state.focus

  const k = Number(state.armCount)
  const T = Number(state.horizon)
  const armMeans = useMemo(() => means.slice(0, k), [means, k])
  const eps = state.epsilon
  const m = state.etcM
  const c = state.decayC
  const shown = ALGORITHMS.filter((a) => offer.includes(a.id))
  const active = shown.filter((a) => state[`on_${a.id}`])

  // A string key, so the simulation reruns only when the set of algorithms changes, not on every render.
  const activeKey = active.map((a) => a.id).join(',')
  const meansKey = armMeans.join(',')
  const runCount = state.runs
  const seedValue = state.seed
  // The React Compiler cannot prove this memo is preserved and skips the component; the memo itself is exact, because
  // every dependency is a string or number, and it keeps the simulation from rerunning while a handle is dragged.
  /* oxlint-disable react/preserve-manual-memoization */
  const results = useMemo(
    () => simulateAll(activeKey, meansKey, T, runCount, seedValue, eps, m, c),
    [activeKey, meansKey, T, runCount, seedValue, eps, m, c],
  )
  /* oxlint-enable react/preserve-manual-memoization */

  const regretSeries = (() => {
    const out: SeriesSpec[] = ALGORITHMS.filter((a) => results.has(a.id)).map((a) => {
      const r = results.get(a.id)!
      return { name: a.label, type: 'line', x: r.t, y: r.regret, slot: a.slot }
    })
    if (state.showBound) {
      const lr = laiRobbinsConstant(armMeans)
      const ts = results.values().next().value?.t ?? [T]
      out.push({
        name: 'Lai–Robbins bound (asymptotic)',
        type: 'line',
        x: ts,
        y: ts.map((t) => lr * Math.log(t)),
        emphasis: true,
        dashed: true,
      })
    }
    return out
  })()

  const focused = active.find((a) => a.id === focus) ?? active[active.length - 1]
  const focusResult = focused ? results.get(focused.id) : undefined
  const arms = armMeans.map((_, i) => i + 1)
  const armSeries: SeriesSpec[] = [
    ...(focused && focusResult
      ? [
          {
            name: `share of pulls, ${focused.label}`,
            type: 'bar' as const,
            x: arms,
            y: focusResult.pulls.map((p) => p / T),
            slot: focused.slot,
          },
        ]
      : []),
    { name: 'true mean (drag)', type: 'scatter', x: arms, y: armMeans, emphasis: true },
  ]
  const handles: Handle[] = armMeans.map((m, i) => ({
    kind: 'point',
    at: [i + 1, m],
    label: `arm ${i + 1}`,
    onDrag: ([, y]) =>
      setMeans((prev) => prev.map((v, j) => (j === i ? Math.min(0.99, Math.max(0.01, Math.round(y * 100) / 100)) : v))),
  }))

  const xAxis = useAxis({ label: 'round t', range: [0, T] })
  const yAxis = useAxis({ label: 'cumulative regret', range: [0, undefined], hold: 'union' })
  const xAxis2 = useAxis({ label: 'arm', range: [0.5, k + 0.5] })
  const yAxis2 = useAxis({ label: 'mean / share', range: [0, 1] })
  return (
    <Figure
      title={title}
      state={state}
      caption={
        caption ??
        `Bernoulli arms with the true means shown as diamonds; drag a diamond to change an arm. Each curve is the cumulative pseudo-regret, the expected reward lost to pulling suboptimal arms, averaged over ${state.runs} seeded runs. Every algorithm sees the same reward draws. A curve that keeps a constant slope has linear regret; a curve that flattens has sublinear regret. The bars show how one algorithm split its pulls.`
      }
      readouts={active.map((a) => (
        <Readout key={a.id} label={`${a.label}: regret`} value={formatNumber(results.get(a.id)!.regret.at(-1)!)} />
      ))}
    >
      <div className="grid gap-4 md:grid-cols-[2fr_1fr]">
        <Plot x={xAxis} y={yAxis}>
          {seriesLayers(regretSeries)}
        </Plot>
        <Plot x={xAxis2} y={yAxis2}>
          {seriesLayers(armSeries)}
          {(handles ?? []).map((h, i) => (
            <Handle key={i} {...h} />
          ))}
        </Plot>
      </div>
    </Figure>
  )
}
