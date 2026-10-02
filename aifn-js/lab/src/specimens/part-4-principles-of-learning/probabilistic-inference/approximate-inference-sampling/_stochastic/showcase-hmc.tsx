import { banana, funnel, nonCentredFunnel } from 'aifn-applied/data/targets'
import {
  effectiveSampleSize,
  hmc,
  mala,
  randomWalkMetropolis,
  sampleChains,
  splitRhat,
  type HmcState,
  type LogDensity,
} from 'aifn/inference/stochastic'
import { stream } from 'aifn/foundation/random'
import { linspace, tensor, toFlat, type Tensor } from 'aifn/foundation/tensor'
import type { Trace } from 'aifn/foundation/trace'
import { useMemo, useState } from 'react'
import { Player, Switch } from '@lab/controls'
import { Dashboard, DashboardCell, DashboardRow, Figure } from '@lab/layout'
import { call, choice, row, slider, useComputed, useFigureState, variants, type Task } from '@lab/state'
import { Bars, Curve, Handle, Plot, Plots, Points, Raster, Readout, useAxis } from '@lab/viz'
import { formatValue } from '@lab/views'
import { seriesColor, useTheme } from '@lab/design'
import { cn } from '@lab/lib/utils'

const f3 = (v: number) => formatValue(Number(v.toPrecision(3)))
const grid = (lo: number, hi: number, n: number) => toFlat(linspace(lo, hi, n))
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))
type P2 = [number, number]

// ── Targets, each drawn in plot coordinates ─────────────────────────────────────────────────────────────────────────
//
// The funnel's parameters are θ = (v, x) with v the log-variance; the plot puts x across and v up, the usual picture.
// The non-centred funnel samples θ = (v, z) with x = z·e^{v/2}; its draws are mapped back by the target's `toOriginal`
// and drawn on the same funnel, so the two parameterisations are compared on one picture.

type Target = {
  /** What the sampler runs on. */
  density: LogDensity
  /** The same density as a worker task: its registry address and parameters. */
  task: Task<LogDensity>
  /** Sampler coordinates → plot coordinates. */
  toPlot: (theta: ArrayLike<number>) => P2
  /** Plot coordinates → sampler coordinates (for the draggable start). */
  fromPlot: (p: P2) => number[]
  /** The density drawn behind (always the centred model, in plot coordinates). */
  shown: LogDensity
  shownAt: (p: P2) => number[]
  box: { x: P2; y: P2 }
  labels: { x: string; y: string; first: string; second: string }
  start: P2
}

const FUNNEL = funnel()
const NC_FUNNEL = nonCentredFunnel()
const BANANA_PARAMS = { a: 1, b: 2 }
const HARD_BANANA = banana(BANANA_PARAMS)
const FUNNEL_BOX = { x: [-12, 12] as P2, y: [-8, 8] as P2 }
const funnelLabels = { x: 'x', y: 'v (log variance)', first: 'v', second: 'x' }

const TARGETS: Record<'centred funnel' | 'non-centred funnel' | 'hard banana', Target> = {
  'centred funnel': {
    density: FUNNEL,
    task: call('data/targets/funnel'),
    toPlot: (t) => [t[1], t[0]],
    fromPlot: ([x, v]) => [v, x],
    shown: FUNNEL,
    shownAt: ([x, v]) => [v, x],
    box: FUNNEL_BOX,
    labels: funnelLabels,
    start: [0.2, -4.5],
  },
  'non-centred funnel': {
    density: NC_FUNNEL,
    task: call('data/targets/nonCentredFunnel'),
    toPlot: (t) => {
      const c = toFlat(NC_FUNNEL.toOriginal(tensor(Array.from(t))) as Tensor)
      return [c[1], c[0]]
    },
    fromPlot: ([x, v]) => [v, x * Math.exp(-v / 2)],
    shown: FUNNEL,
    shownAt: ([x, v]) => [v, x],
    box: FUNNEL_BOX,
    labels: funnelLabels,
    start: [0.2, -4.5],
  },
  'hard banana': {
    density: HARD_BANANA,
    task: call('data/targets/banana', BANANA_PARAMS),
    toPlot: (t) => [t[0], t[1]],
    fromPlot: (p) => [...p],
    shown: HARD_BANANA,
    shownAt: (p) => [...p],
    box: { x: [-4, 4], y: [-4, 22] },
    labels: { x: 'x', y: 'y', first: 'x', second: 'y' },
    start: [-2.5, 10],
  },
}
type TargetKey = keyof typeof TARGETS
const TARGET_KEYS = Object.keys(TARGETS) as TargetKey[]

/** Each target with its own start (placed on the chart, moved by its handle; each target keeps its own). */
const startOf = (t: Target) => ({
  sx: slider(t.box.x[0], t.box.x[1], t.start[0], { onChart: true, label: 'start x' }),
  sy: slider(t.box.y[0], t.box.y[1], t.start[1], { onChart: true, label: 'start y' }),
})
const TARGET = variants(
  {
    'centred funnel': { label: 'centred funnel', params: startOf(TARGETS['centred funnel']) },
    'non-centred funnel': { label: 'non-centred funnel', params: startOf(TARGETS['non-centred funnel']) },
    'hard banana': { label: 'hard banana', params: startOf(TARGETS['hard banana']) },
  },
  { label: '1 · target', choiceLabel: 'target' },
)
const integrator = (label: string) =>
  row(label, {
    eps: slider(0.02, 1, 0.3, { label: 'step size ε', step: 0.01 }),
    L: slider(2, 60, 25, { label: 'path length L (steps)', step: 1 }),
  })

/** log π of the shown density on a grid in plot coordinates, floored so that the funnel's neck does not wash out. */
function logDensityGrid(t: Target, n = 110) {
  const x = grid(t.box.x[0], t.box.x[1], n)
  const y = grid(t.box.y[0], t.box.y[1], n)
  const z = y.map((yi) => x.map((xj) => Math.max(-12, t.shown.logDensity(tensor(t.shownAt([xj, yi]))) as number)))
  return { x, y, z }
}

/** Rows of an (L + 1)×2 trajectory, mapped to plot coordinates. */
function trajectoryPoints(traj: Tensor, t: Target): P2[] {
  const flat = toFlat(traj)
  return Array.from({ length: flat.length / 2 }, (_, i) => t.toPlot([flat[2 * i], flat[2 * i + 1]]))
}

const ITERATIONS = 40

type Iteration = {
  points: P2[]
  dH: number[]
  energyError: number
  divergent: boolean
  accepted: boolean
  sample: P2
}

export function HmcFunnelTrajectories() {
  const state = useFigureState({ target: TARGET, integrator: integrator('2 · integrator') })
  const key = state.target.key
  const { eps, L } = state.integrator
  // A new target starts from its first frame: the position is kept with the target it was played on.
  const [played, setPlayed] = useState({ key, pos: 0 })
  const pos = played.key === key ? played.pos : 0
  const setPos = (p: number) => setPlayed({ key, pos: p })
  const target = TARGETS[key]
  const surface = useMemo(() => logDensityGrid(target), [target])
  const { sx, sy } = state.target.values
  const start = useMemo((): P2 => [sx, sy], [sx, sy])
  const x0 = useMemo(() => target.fromPlot(start), [target, start])

  // Frames: 0 is the start, then L leapfrog points per iteration.
  const frameCount = ITERATIONS * L
  const at = Math.min(pos, frameCount)
  const need = at === 0 ? 0 : Math.floor((at - 1) / L) + 1
  // The handle follows the pointer at once; the chain (all 40 trajectories, ≈ 15 ms) reruns in a Web Worker on every
  // change, latest wins, so a drag never waits for it: the chain shows dimmed until the answer for the newest start
  // lands. The task names aifn's exports by address with plain inputs; `then` maps the trace to plot coordinates.
  const run = useComputed(
    () =>
      call<Trace<HmcState>>(
        'foundation/trace/trace',
        call('inference/stochastic/hmc', target.task, { stepSize: eps, steps: L }),
        { x0 },
        ITERATIONS,
        { stream: call('foundation/random/stream', 'showcase-hmc') },
      ),
    [target, eps, L, x0],
    {
      mode: 'worker',
      initial: [] as Iteration[],
      then: (tr) =>
        tr.steps.slice(1).map((s): Iteration => {
          const H = toFlat(s.energies)
          return {
            points: trajectoryPoints(s.trajectory, target),
            dH: H.map((h) => h - H[0]),
            energyError: s.energyError,
            divergent: s.divergent,
            accepted: s.accepted,
            sample: target.toPlot(toFlat(s.x)),
          }
        }),
    },
  )
  const iterations = run.value

  // The played frame as numbers (fi = −1 at the start), so the memos below key on values. A trajectory that turned
  // non-finite stops early; its later frames show its last point.
  const fi = need - 1
  const current = fi >= 0 ? iterations[fi] : null
  const fk = current ? Math.min(((at - 1) % L) + 1, current.points.length - 1) : 0
  const finished = fi >= 0 && ((at - 1) % L) + 1 === L ? fi + 1 : Math.max(fi, 0)
  const done = useMemo(() => iterations.slice(0, finished), [iterations, finished])

  const layers = useMemo(() => {
    const past = iterations.slice(0, Math.max(fi, 0)).flatMap((it) => [...it.points, [NaN, NaN] as P2])
    const partial = current ? current.points.slice(0, fk + 1) : []
    const samples = [start, ...done.map((d) => d.sample)]
    const divergentStarts = done.flatMap((d, j) => (d.divergent ? [j === 0 ? start : done[j - 1].sample] : []))
    const xy = (pts: P2[]) => ({ x: pts.map((p) => p[0]), y: pts.map((p) => p[1]) })
    return { past: xy(past), partial: xy(partial), samples: xy(samples), divergent: xy(divergentStarts) }
  }, [iterations, fi, fk, current, done, start])

  const energy = useMemo(() => {
    const dH = current ? current.dH.slice(0, fk + 1) : [0]
    return { x: dH.map((_, k) => k), y: dH }
  }, [current, fk])

  const perIteration = useMemo(() => {
    const xs = done.map((_, j) => j + 1)
    return {
      x: xs,
      ok: done.map((d) => (d.divergent ? NaN : clamp(d.energyError, -3, 3))),
      bad: done.map((d) => (d.divergent ? 3 : NaN)),
    }
  }, [done])
  const px = useAxis({ label: target.labels.x, range: target.box.x, key: key === 'hard banana' ? 'banana' : 'funnel' })
  const py = useAxis({ label: target.labels.y, range: target.box.y, key: key === 'hard banana' ? 'banana' : 'funnel' })
  const stepAxis = useAxis({ label: 'leapfrog step', range: [0, L] })
  const dhAxis = useAxis({ label: 'H − H₀', range: [-3, 3] })
  const itAxis = useAxis({ label: 'iteration', range: [0, ITERATIONS + 1] })
  const finalAxis = useAxis({ label: 'final ΔH', range: [-3, 3] })

  const divergences = done.filter((d) => d.divergent).length
  const accepted = done.filter((d) => d.accepted).length
  return (
    <Figure
      title="HMC on the funnel and a hard banana, leapfrog step by leapfrog step"
      purpose="A fixed step size that suits the funnel's mouth is unstable in its neck, so trajectories there blow up (divergences); the non-centred parameterisation turns the funnel into a Gaussian and the divergences vanish."
      defaultSize="L"
      state={state}
      controls={
        <>
          <Switch
            label="non-centred (x = z·exp(v/2))"
            checked={key === 'non-centred funnel'}
            disabled={key === 'hard banana'}
            onChange={(on: boolean) => state.set('target', on ? 'non-centred funnel' : 'centred funnel')}
          />
          <Player
            label="3 · play"
            value={at}
            onChange={setPos}
            count={frameCount + 1}
            format={(p) => (p === 0 ? 'start' : `iteration ${Math.floor((p - 1) / L) + 1}, step ${((p - 1) % L) + 1}`)}
          />
        </>
      }
      readouts={{
        'so far': (
          <>
            <Readout label="iteration" value={fi >= 0 ? `${fi + 1} of ${ITERATIONS}` : '0'} />
            <Readout label="ΔH now" value={current ? f3(current.dH[fk]) : '0'} />
            <Readout label="divergent trajectories" value={`${divergences} of ${done.length}`} />
            <Readout label="accepted" value={`${accepted} of ${done.length}`} />
          </>
        ),
      }}
      caption="Drag the start (the large dot) to rerun the chain from there. Left: log π (floored at −12) with the finished trajectories (thin), the current one up to the played leapfrog step, the samples, and a diamond where each divergent trajectory began. Top right: the energy error H − H₀ along the current trajectory; an exact flow keeps it at 0. Bottom right: the final energy error of each finished trajectory, clipped to ±3; divergent ones (|ΔH| > 1000 or non-finite) are drawn at the top in their own colour. On the centred funnel, started in the neck at v = −4.5 with ε = 0.3, ε is above the neck's stable step (about 2·exp(v/2) ≈ 0.21 there), so most trajectories gain energy and are rejected, the chain sticks, and about a quarter diverge. Switch to non-centred and every trajectory keeps |ΔH| small, is accepted, and none diverge. On the non-centred funnel the sampler runs on (v, z) and every point is mapped back to (x, v)."
    >
      <Dashboard>
        <DashboardRow minHeight={380}>
          <DashboardCell ratio={1.3}>
            <Plot x={px} y={py} renderer="canvas">
              <Raster {...surface} valueLabel="log π" colorBar={false} fillOpacity={0.75} />
              {/* The chain's marks are live: patched as the start is dragged, the raster never resent. */}
              <Curve name="earlier trajectories" {...layers.past} slot={1} thin live stale={run.stale} />
              <Curve name="current trajectory" {...layers.partial} slot={2} showPoints live stale={run.stale} />
              <Points name="samples" {...layers.samples} slot={3} live stale={run.stale} />
              <Points name="divergent start" {...layers.divergent} emphasis shape={4} live stale={run.stale} />
              <Handle {...state.handle(['target.sx', 'target.sy'], { label: 'start' })} />
            </Plot>
          </DashboardCell>
          <DashboardCell>
            <Plots rows={2}>
              <Plot x={stepAxis} y={dhAxis} renderer="canvas">
                <Curve name="ΔH = H − H₀ along the trajectory" {...energy} slot={2} showPoints live />
                <Curve name="zero" x={[0, L]} y={[0, 0]} muted dashed />
              </Plot>
              <Plot x={itAxis} y={finalAxis} renderer="canvas">
                <Bars name="final ΔH (clipped to ±3)" x={perIteration.x} y={perIteration.ok} slot={0} live />
                <Bars name="divergent" x={perIteration.x} y={perIteration.bad} slot={3} live />
              </Plot>
            </Plots>
          </DashboardCell>
        </DashboardRow>
      </Dashboard>
    </Figure>
  )
}

// ── Several chains: ESS and R̂ of random-walk MH, MALA and HMC ────────────────────────────────────────────────────────

const SAMPLERS = ['random-walk MH', 'MALA', 'HMC'] as const
const CHAINS = 4
const DRAWS = 600
const WARMUP = 100

/** Per-coordinate draws of every chain, in plot coordinates (the funnel's (x, v), the banana's (x, y)). */
function plotDraws(draws: Tensor, t: Target) {
  const [m, n] = draws.shape
  const flat = toFlat(draws)
  const first: number[][] = []
  const second: number[][] = []
  for (let c = 0; c < m; c++) {
    const a: number[] = []
    const b: number[] = []
    for (let i = 0; i < n; i++) {
      const o = (c * n + i) * 2
      const p = t.toPlot([flat[o], flat[o + 1]])
      a.push(p[1])
      b.push(p[0])
    }
    first.push(a)
    second.push(b)
  }
  return { first, second }
}

export function HmcSamplerComparison() {
  const state = useFigureState({
    target: row('1 · target', { key: choice(TARGET_KEYS, 'centred funnel', { label: 'target' }) }),
    integrator: integrator('2 · HMC'),
  })
  const key: TargetKey = state.target.key
  const { eps, L } = state.integrator
  const target = TARGETS[key]
  // Twelve chains of 700 steps: rerun when a slider is released.
  const computed = useComputed(
    () => {
      const starts = (k: number) => ({
        x0: target.fromPlot([2 * Math.cos(k * 1.9), target === TARGETS['hard banana'] ? 2 : -1 + k]),
      })
      const algs = {
        'random-walk MH': randomWalkMetropolis(target.density, { scale: 1 }),
        MALA: mala(target.density, { stepSize: 0.1 }),
        HMC: hmc(target.density, { stepSize: eps, steps: L }),
      }
      return SAMPLERS.map((name, j) => {
        const res = sampleChains(algs[name], starts, {
          chains: CHAINS,
          steps: DRAWS + WARMUP,
          warmup: WARMUP,
          stream: stream(`showcase-compare-${j}`),
          record: { divergent: (s) => ('divergent' in s && s.divergent ? 1 : 0) },
        })
        const d = plotDraws(res.draws, target)
        const divergent = res.traces.reduce(
          (acc, tr) => acc + toFlat(tr.series.divergent).reduce((a, b) => a + b, 0),
          0,
        )
        return {
          name,
          essFirst: effectiveSampleSize(d.first),
          essSecond: effectiveSampleSize(d.second),
          rhatFirst: splitRhat(d.first),
          rhatSecond: splitRhat(d.second),
          trace: d.first[0],
          divergent,
          rate: res.traces.reduce((a, tr) => a + (tr.final as { acceptanceRate: number }).acceptanceRate, 0) / CHAINS,
        }
      })
    },
    [target, eps, L],
    { mode: 'release' },
  )
  const results = computed.value

  const names = target.labels
  const steps = useMemo(() => Array.from({ length: DRAWS }, (_, i) => i + WARMUP + 1), [])
  const sx = useAxis({ label: 'step', range: [WARMUP, WARMUP + DRAWS] })
  const sy = useAxis({ label: names.first, key })
  const mode = useTheme().resolved
  const ess = (v: number) => formatValue(Math.round(v))
  const rhat = (v: number) => v.toFixed(3)
  return (
    <Figure
      title="Four chains each: ESS and R̂ of random-walk MH, MALA and HMC"
      purpose="On the centred funnel every sampler under-explores v, which shows as a small bulk ESS and an R̂ above 1.01; the non-centred funnel is easy for all three, and HMC's long, gradient-guided moves give it the largest ESS per draw."
      defaultSize="M"
      state={state}
      readouts={
        <div className="max-w-full overflow-x-auto">
          <table className="text-xs whitespace-nowrap">
            <thead className="text-muted-foreground">
              <tr>
                <th className="pr-4 text-left font-normal">sampler</th>
                <th className="px-2 text-right font-normal">ESS {names.first}</th>
                <th className="px-2 text-right font-normal">ESS {names.second}</th>
                <th className="px-2 text-right font-normal">R̂ {names.first}</th>
                <th className="px-2 text-right font-normal">R̂ {names.second}</th>
                <th className="px-2 text-right font-normal">acceptance</th>
                <th className="pl-2 text-right font-normal">divergent</th>
              </tr>
            </thead>
            <tbody className="font-mono tabular-nums">
              {results.map((r, j) => (
                <tr key={r.name}>
                  <td className="pr-4 font-sans">
                    <span
                      aria-hidden
                      className="mr-1.5 inline-block size-2 rounded-full"
                      style={{ background: seriesColor(mode, j + 2) }}
                    />
                    {r.name}
                  </td>
                  <td className="px-2 text-right">{ess(r.essFirst)}</td>
                  <td className="px-2 text-right">{ess(r.essSecond)}</td>
                  <td className={cn('px-2 text-right', r.rhatFirst > 1.01 && 'text-destructive')}>
                    {rhat(r.rhatFirst)}
                  </td>
                  <td className={cn('px-2 text-right', r.rhatSecond > 1.01 && 'text-destructive')}>
                    {rhat(r.rhatSecond)}
                  </td>
                  <td className="px-2 text-right">{f3(r.rate)}</td>
                  <td className="pl-2 text-right">{r.name === 'HMC' ? r.divergent : '–'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      }
      caption={`Each sampler runs ${CHAINS} chains of ${DRAWS} draws after ${WARMUP} warm-up steps from dispersed starts: random-walk MH with proposal sd 1, MALA with step 0.1, HMC with the ε and L above (L gradient evaluations per draw, so HMC does L times the work of MALA). The table gives the bulk ESS (rank-normalised, split chains) of each coordinate out of ${CHAINS * DRAWS} draws and the rank-normalised split R̂, red above 1.01. The chart is chain 1's trace of ${names.first} for each sampler. On the funnel, v is the hard coordinate: the centred chains stick in the neck.`}
    >
      <Plot x={sx} y={sy}>
        {results.map((r, j) => (
          <Curve key={r.name} name={r.name} x={steps} y={r.trace} slot={j + 2} thin stale={computed.stale} />
        ))}
      </Plot>
    </Figure>
  )
}
