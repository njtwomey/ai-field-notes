import { useMemo } from 'react'
import {
  choice,
  Figure,
  float,
  formatNumber,
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
import { ALGORITHMS, DEFAULT_TUNING, checkpoints, makePolicy, type AlgorithmId } from '../_shared/bandits'
import { seededRand } from '../_shared/rand'
import { normalCdf } from 'aifn/numerics/special'

type DesignId = 'uniform' | 'etc' | 'ucb1' | 'ts' | 'tsfloor'
/** Designs that are algorithms of the category keep that algorithm's colour slot; the floored variant takes a free one. */
const slotOf = (id: AlgorithmId) => ALGORITHMS.find((a) => a.id === id)!.slot
const DESIGNS: { id: DesignId; label: string; slot: number }[] = [
  { id: 'uniform', label: 'A/B test (50/50)', slot: slotOf('uniform') },
  { id: 'etc', label: 'A/B test, then ship', slot: slotOf('etc') },
  { id: 'ucb1', label: 'UCB1', slot: slotOf('ucb1') },
  { id: 'ts', label: 'Thompson sampling', slot: slotOf('ts') },
  { id: 'tsfloor', label: 'Thompson, 10% floor per arm', slot: 5 },
]
const HORIZONS = ['2000', '10000', '30000'] as const
type Horizon = (typeof HORIZONS)[number]
const Z_ALPHA = 1.96
const Z_POWER = 0.8416

type Curves = { t: number[]; regret: number[]; se: number[]; power: number }

/** Per-arm sample size of a two-sided 5% test with 80% power for the difference pB − pA. */
function abSampleSize(pA: number, pB: number): number {
  const v = pA * (1 - pA) + pB * (1 - pB)
  return Math.ceil(((Z_ALPHA + Z_POWER) ** 2 * v) / (pB - pA) ** 2)
}

/**
 * Runs one design on two Bernoulli arms and averages, over seeded runs, the cumulative regret and the standard error
 * √(pA(1 − pA)/N_A + pB(1 − pB)/N_B) that the final allocation gives the estimated difference.
 */
function simulateDesign(id: DesignId, pA: number, pB: number, horizon: number, seed: number): Curves {
  const means = [pA, pB]
  const best = Math.max(pA, pB)
  const gap = Math.abs(pB - pA)
  const ts = checkpoints(horizon, 120)
  const regret = new Array<number>(ts.length).fill(0)
  const se = new Array<number>(ts.length).fill(0)
  // Fewer runs for long horizons keep a redraw under a few hundred milliseconds.
  const runs = horizon > 10000 ? 8 : 16
  let power = 0
  const m = Math.min(Math.floor(horizon / 2), abSampleSize(pA, pB))
  for (let run = 0; run < runs; run++) {
    const env = seededRand(seed * 7919 + run)
    const agent = seededRand(seed * 104729 + run + 1)
    const base: AlgorithmId = id === 'tsfloor' ? 'ts' : id
    const policy = makePolicy(base, 2, { ...DEFAULT_TUNING, etcM: m }, agent)
    let cum = 0
    let next = 0
    for (let t = 1; t <= horizon; t++) {
      // The floor keeps each arm's assignment probability at 10% or more: with probability 0.2, pick uniformly.
      const a = id === 'tsfloor' && agent.uniform() < 0.2 ? (agent.uniform() < 0.5 ? 0 : 1) : policy.choose(t)
      policy.update(a, env.uniform() < means[a] ? 1 : 0)
      cum += best - means[a]
      if (t === ts[next]) {
        const [nA, nB] = policy.n
        const s = Math.sqrt((pA * (1 - pA)) / Math.max(nA, 1) + (pB * (1 - pB)) / Math.max(nB, 1))
        regret[next] += cum
        se[next] += s
        if (t === horizon) power += normalCdf(gap / s - Z_ALPHA)
        next++
      }
    }
  }
  return { t: ts, regret: regret.map((v) => v / runs), se: se.map((v) => v / runs), power: power / runs }
}

/**
 * Regret against estimation precision on a two-arm conversion test: each design's cumulative regret, and the standard
 * error its allocation gives the estimated lift.
 */
export function RegretVsEstimation() {
  const state = useFigureState({
    base: float(0.1, { min: 0.02, max: 0.5, step: 0.01, label: 'conversion rate pA' }),
    lift: float(0.01, { min: 0.002, max: 0.05, step: 0.001, label: 'lift pB − pA', format: (v) => v.toFixed(3) }),
    horizon: choice<Horizon>(
      HORIZONS.map((v) => ({ value: v, label: Number(v).toLocaleString() })),
      '10000',
      { label: 'users T' },
    ),
    seed: int(1, { min: 1, max: 30, step: 1, label: 'seed' }),
    // One switch per design; UCB1 starts off.
    ...(Object.fromEntries(DESIGNS.map((d) => [`on_${d.id}`, setting(d.id !== 'ucb1', d.label)])) as Record<
      `on_${DesignId}`,
      SwitchDef
    >),
  })
  const enabled = DESIGNS.filter((d) => state[`on_${d.id}`]).map((d) => d.id)

  const pA = state.base
  const pB = Math.min(0.99, pA + state.lift)
  const T = Number(state.horizon)
  const activeKey = enabled.join(',')
  const results = useMemo(
    () =>
      new Map(
        (activeKey ? (activeKey.split(',') as DesignId[]) : []).map(
          (id) => [id, simulateDesign(id, pA, pB, T, state.seed)] as const,
        ),
      ),
    [activeKey, pA, pB, T, state.seed],
  )
  const active = DESIGNS.filter((d) => results.has(d.id))
  const needed = (pB - pA) / (Z_ALPHA + Z_POWER)

  const regretSeries: SeriesSpec[] = active.map((d) => {
    const r = results.get(d.id)!
    return { name: d.label, type: 'line', x: r.t, y: r.regret, slot: d.slot }
  })
  const seSeries: SeriesSpec[] = [
    ...active.map((d): SeriesSpec => {
      const r = results.get(d.id)!
      return { name: d.label, type: 'line', x: r.t, y: r.se, slot: d.slot }
    }),
    { name: 'SE for 80% power', type: 'line', x: [0, T], y: [needed, needed], emphasis: true, dashed: true },
  ]

  const xAxis = useAxis({ label: 'users t', range: [0, T] })
  const yAxis = useAxis({ label: 'lost conversions', range: [0, undefined], hold: 'union' })
  const xAxis2 = useAxis({ label: 'users t', range: [0, T] })
  const yAxis2 = useAxis({ label: 'SE of the lift', hold: 'union', log: true })
  return (
    <Figure
      title="Earning against learning"
      state={state}
      caption="Two arms with conversion rates pA and pB = pA + lift. Left: cumulative regret, the conversions lost to showing the worse arm, averaged over seeded runs. Right: the standard error of the estimated lift that each design's allocation provides, on a log scale. The 50/50 test pays regret at a constant rate and reaches the precision needed for 80% power at the planned sample size (ETC ships the winner there). The bandits pay far less regret but starve the worse arm, so their standard error stalls well above the dashed line: they rarely have the data to show the lift is real. The floor on each arm's assignment probability buys back some precision at the cost of some regret."
      readouts={
        <>
          <Readout label="planned A/B size per arm" value={abSampleSize(pA, pB).toLocaleString()} />
          {active.map((d) => {
            const r = results.get(d.id)!
            return (
              <Readout
                key={d.id}
                label={`${d.label}: regret, SE, power`}
                value={`${formatNumber(r.regret.at(-1)!)}, ${formatNumber(r.se.at(-1)!)}, ${Math.round(100 * r.power)}%`}
              />
            )
          })}
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis}>
          {seriesLayers(regretSeries)}
        </Plot>
        <Plot x={xAxis2} y={yAxis2}>
          {seriesLayers(seSeries)}
        </Plot>
      </div>
    </Figure>
  )
}
