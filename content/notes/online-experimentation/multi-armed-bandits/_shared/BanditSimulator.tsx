import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  ParamSwitch,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import { rng } from '@/lib/math'
import { ALGORITHMS, DEFAULT_TUNING, laiRobbinsConstant, simulate, type AlgorithmId, type Tuning } from './bandits'

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
  return new Map(ids.map((id) => [id, simulate(id, mu, horizon, runCount, seed, tuning, rng)] as const))
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
  const [armCount, setArmCount] = useState<ArmCount>('3')
  const [horizon, setHorizon] = useState<Horizon>('2000')
  const [enabled, setEnabled] = useState<AlgorithmId[]>(algorithms)
  const [showBound, setShowBound] = useState(lowerBound)
  const [focus, setFocus] = useState<AlgorithmId>(algorithms[algorithms.length - 1] ?? 'ucb1')
  const runs = useParam(20, { min: 1, max: 50, step: 1 })
  const seed = useParam(1, { min: 1, max: 30, step: 1 })
  const epsilon = useParam(DEFAULT_TUNING.epsilon, { min: 0, max: 0.5, step: 0.01 })
  const etcM = useParam(DEFAULT_TUNING.etcM, { min: 1, max: 300, step: 1 })
  const decayC = useParam(DEFAULT_TUNING.decayC, { min: 0.5, max: 50, step: 0.5 })

  const k = Number(armCount)
  const T = Number(horizon)
  const armMeans = useMemo(() => means.slice(0, k), [means, k])
  const eps = epsilon.value
  const m = etcM.value
  const c = decayC.value
  const shown = ALGORITHMS.filter((a) => offer.includes(a.id))
  const active = shown.filter((a) => enabled.includes(a.id))

  // A string key, so the simulation reruns only when the set of algorithms changes, not on every render.
  const activeKey = active.map((a) => a.id).join(',')
  const meansKey = armMeans.join(',')
  const runCount = runs.value
  const seedValue = seed.value
  // The React Compiler cannot prove this memo is preserved and skips the component; the memo itself is exact, because
  // every dependency is a string or number, and it keeps the simulation from rerunning while a handle is dragged.
  /* oxlint-disable react/preserve-manual-memoization */
  const results = useMemo(
    () => simulateAll(activeKey, meansKey, T, runCount, seedValue, eps, m, c),
    [activeKey, meansKey, T, runCount, seedValue, eps, m, c],
  )
  /* oxlint-enable react/preserve-manual-memoization */

  const regretSeries = (() => {
    const out: XYSeries[] = ALGORITHMS.filter((a) => results.has(a.id)).map((a) => {
      const r = results.get(a.id)!
      return { name: a.label, type: 'line', x: r.t, y: r.regret, slot: a.slot }
    })
    if (showBound) {
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
  const armSeries: XYSeries[] = [
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

  const toggle = (id: AlgorithmId, on: boolean) =>
    setEnabled((prev) => (on ? [...prev, id] : prev.filter((x) => x !== id)))

  return (
    <Interactive
      title={title}
      caption={
        caption ??
        `Bernoulli arms with the true means shown as diamonds; drag a diamond to change an arm. Each curve is the cumulative pseudo-regret, the expected reward lost to pulling suboptimal arms, averaged over ${runs.value} seeded runs. Every algorithm sees the same reward draws. A curve that keeps a constant slope has linear regret; a curve that flattens has sublinear regret. The bars show how one algorithm split its pulls.`
      }
      controls={
        <>
          <ParamChoice
            label="arms"
            value={armCount}
            onChange={setArmCount}
            options={ARM_COUNTS.map((v) => ({ value: v, label: v }))}
          />
          <ParamChoice
            label="horizon T"
            value={horizon}
            onChange={setHorizon}
            options={HORIZONS.map((v) => ({ value: v, label: v }))}
          />
          <ParamSlider label="runs averaged" param={runs} format={(v) => String(v)} />
          <ParamSlider label="seed" param={seed} format={(v) => String(v)} withArrows />
          {tuningKeys.includes('epsilon') && <ParamSlider label="ε (constant ε-greedy)" param={epsilon} />}
          {tuningKeys.includes('decayC') && (
            <ParamSlider label="c (decaying ε = min(1, cK/t))" param={decayC} format={(v) => String(v)} />
          )}
          {tuningKeys.includes('etcM') && (
            <ParamSlider label="m (explore-then-commit pulls per arm)" param={etcM} format={(v) => String(v)} />
          )}
          {active.length > 1 && (
            <ParamChoice
              label="bars show"
              value={focused?.id ?? focus}
              onChange={setFocus}
              options={active.map((a) => ({ value: a.id, label: a.label }))}
            />
          )}
          <div className="flex flex-wrap gap-x-4 gap-y-2 sm:col-span-2 lg:col-span-3">
            {shown.map((a) => (
              <ParamSwitch
                key={a.id}
                label={a.label}
                checked={enabled.includes(a.id)}
                onChange={(on) => toggle(a.id, on)}
              />
            ))}
            <ParamSwitch label="Lai–Robbins bound" checked={showBound} onChange={setShowBound} />
          </div>
        </>
      }
      readout={active.map((a) => (
        <Readout key={a.id} label={`${a.label}: regret`} value={formatNumber(results.get(a.id)!.regret.at(-1)!)} />
      ))}
    >
      <div className="grid gap-4 md:grid-cols-[2fr_1fr]">
        <XYChart
          series={regretSeries}
          xLabel="round t"
          yLabel="cumulative regret"
          xRange={[0, T]}
          yRange={[0, undefined]}
        />
        <XYChart
          series={armSeries}
          xLabel="arm"
          yLabel="mean / share"
          xRange={[0.5, k + 0.5]}
          yRange={[0, 1]}
          handles={handles}
        />
      </div>
    </Interactive>
  )
}
