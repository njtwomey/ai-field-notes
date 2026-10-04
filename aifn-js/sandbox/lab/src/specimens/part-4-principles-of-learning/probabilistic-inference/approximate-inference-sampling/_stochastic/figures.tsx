import { banana, funnel, gaussianTarget } from 'aifn-methods/data/targets'
import {
  bivariateGaussianConditionals,
  conditionalMean,
  effectiveSampleSize,
  gaussianConditionals,
  gibbs,
  hmc,
  integratedAutocorrelationTime,
  mala,
  monteCarloStandardError,
  nuts,
  particleFilter,
  raoBlackwell,
  randomWalkMetropolis,
  resamplingSchemes,
  sampleChains,
  unadjustedLangevin,
  type HmcState,
  type NutsState,
  type ResamplingScheme,
  type LogDensity,
} from 'aifn/inference/stochastic'
import { child, normal, stream } from 'aifn/foundation/random'
import { autocorrelation, histogram } from 'aifn/probability/stats'
import { linspace, tensor, toFlat, variance, type Vector } from 'aifn/foundation/tensor'
import { trace, type Trace } from 'aifn/foundation/trace'
import { useMemo, useState } from 'react'
import { Player } from '@lab/controls'
import { Figure } from '@lab/layout'
import { choice, row, slider, useComputed, useFigureState, variants } from '@lab/state'
import { Annotation, Bars, Curve, Handle, Plot, Plots, Points, Raster, Readout, useAxis } from '@lab/viz'
import { ChainPanel, formatValue, histogramBars } from '@lab/views'

const f3 = (v: number) => formatValue(Number(v.toPrecision(3)))
const grid = (lo: number, hi: number, n: number) => toFlat(linspace(lo, hi, n))
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))
const startSliders = (box: Box, at: [number, number]) => ({
  sx: slider(box.x[0], box.x[1], at[0], { onChart: true, label: 'start θ₀' }),
  sy: slider(box.y[0], box.y[1], at[1], { onChart: true, label: 'start θ₁' }),
})

type Box = { x: [number, number]; y: [number, number] }

/** The target density exp(log π) on a grid, for the heatmap. */
function densityGrid(target: LogDensity, box: Box, n = 90) {
  const x = grid(box.x[0], box.x[1], n)
  const y = grid(box.y[0], box.y[1], n)
  const z = y.map((yi) => x.map((xj) => Math.exp(target.logDensity(tensor([xj, yi])) as number)))
  return { x, y, z }
}

/** Columns 0 and 1 of an n×d matrix (or of the rows of stacked states). */
function columns(flat: readonly number[], d: number) {
  const n = flat.length / d
  // A point with a non-finite coordinate (a diverged trajectory) is dropped whole, so no marker is placed at NaN.
  const ok = (i: number) => Number.isFinite(flat[i * d]) && Number.isFinite(flat[i * d + 1])
  return {
    xs: Array.from({ length: n }, (_, i) => (ok(i) ? flat[i * d] : NaN)),
    ys: Array.from({ length: n }, (_, i) => (ok(i) ? flat[i * d + 1] : NaN)),
  }
}

// ── Random-walk Metropolis on a banana ─────────────────────────────────────────────────────────────────────────────

const BANANA = banana({ a: 1, b: 1 })
const BANANA_BOX: Box = { x: [-3.5, 3.5], y: [-2.5, 8] }
const SCALES = grid(Math.log10(0.03), Math.log10(10), 25).map((v) => 10 ** v)

export function MetropolisBanana() {
  const state = useFigureState({
    proposal: row('1 · proposal', { scale: slider(0.03, 10, 0.6, { label: 'proposal scale σ', step: 0.01 }) }),
    run: row('2 · run', { steps: slider(100, 5000, 1500, { label: 'steps', step: 100 }) }),
    ...startSliders(BANANA_BOX, [-2.5, 6]),
  })
  const { scale } = state.proposal
  const { steps } = state.run
  const start = useMemo((): [number, number] => [state.sx, state.sy], [state.sx, state.sy])
  const surface = useMemo(() => densityGrid(BANANA, BANANA_BOX), [])

  const chain = useComputed(() => {
    const tr = trace(randomWalkMetropolis(BANANA, { scale }), { x0: start }, steps, {
      stream: stream('mh-banana'),
      record: { x: (s) => s.x, proposal: (s) => s.proposal, accepted: (s) => (s.accepted ? 1 : 0) },
    })
    const path = columns(toFlat(tr.series.x), 2)
    const prop = columns(toFlat(tr.series.proposal), 2)
    const acc = toFlat(tr.series.accepted)
    const rejected = { xs: [] as number[], ys: [] as number[] }
    acc.forEach((a, i) => {
      if (i > 0 && !a && rejected.xs.length < 150) {
        rejected.xs.push(prop.xs[i])
        rejected.ys.push(prop.ys[i])
      }
    })
    const last = tr.steps[tr.steps.length - 1]
    return { path, rejected, rate: last.acceptanceRate, iact: integratedAutocorrelationTime(path.ys.slice(1)) }
  }, [scale, steps, start])

  // Acceptance rate and ESS per step of y over a sweep of scales (one chain of 2000 steps each, from the mode).
  const sweep = useMemo(() => {
    const rows = SCALES.map((s) => {
      const tr = trace(randomWalkMetropolis(BANANA, { scale: s }), { x0: [0, 0] }, 2000, {
        stream: stream('mh-sweep'),
        record: { y: (st) => toFlat(st.x)[1] },
      })
      const y = toFlat(tr.series.y).slice(1)
      return {
        rate: tr.steps[tr.steps.length - 1].acceptanceRate,
        ess: effectiveSampleSize(y, { method: 'mean' }) / y.length,
      }
    })
    return { rate: rows.map((r) => r.rate), ess: rows.map((r) => r.ess) }
  }, [])
  const c = chain.value
  const bx = useAxis({ label: 'x', range: BANANA_BOX.x })
  const by = useAxis({ label: 'y', range: BANANA_BOX.y })
  const sx = useAxis({ label: 'proposal scale σ', log: true })
  const sy = useAxis({ label: 'rate (log scale)', log: true })
  return (
    <Figure
      title="Random-walk Metropolis on a banana"
      purpose="The proposal scale trades acceptance against step length: tiny steps are almost always accepted but crawl, large steps are almost always rejected; efficiency peaks in between."
      defaultSize="L"
      state={state}
      readouts={{
        chain: (
          <>
            <Readout label="acceptance rate" value={f3(c.rate)} />
            <Readout label="integrated autocorrelation time of y" value={f3(c.iact)} />
          </>
        ),
      }}
      caption="Left: the banana log π = −x²/2 − (y − (x² − 1))²/2 with the chain (line) and up to 150 rejected proposals (points); drag the start. Right: over proposal scales from 0.03 to 10 (log axis), the acceptance rate and the effective sample size per step of y, from one 2000-step chain each; drag the vertical line to set σ. Both axes are logarithmic. The ESS per step peaks at σ between 1 and 3, where the acceptance rate is 0.2–0.4 (the dashed line marks 0.234); smaller steps are accepted more often but explore less."
    >
      <Plots cols={2} widths={[1.2, 1]}>
        <Plot x={bx} y={by}>
          <Raster {...surface} valueLabel="π(x, y)" colorBar={false} fillOpacity={0.7} />
          <Points name="rejected proposals" x={c.rejected.xs} y={c.rejected.ys} slot={3} thin stale={chain.stale} />
          <Curve name="chain" x={c.path.xs} y={c.path.ys} slot={1} thin stale={chain.stale} />
          <Handle {...state.handle(['sx', 'sy'], { label: 'start' })} />
        </Plot>
        <Plot x={sx} y={sy}>
          <Curve name="acceptance rate" x={SCALES} y={sweep.rate} slot={0} showPoints />
          <Curve name="ESS per step (y)" x={SCALES} y={sweep.ess} slot={1} showPoints />
          <Curve name="0.234" x={[SCALES[0], SCALES[SCALES.length - 1]]} y={[0.234, 0.234]} muted dashed />
          <Handle {...state.handle('proposal.scale', { label: 'σ' })} />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── Gibbs zig-zag ───────────────────────────────────────────────────────────────────────────────────────────────────

const GIBBS_BOX: Box = { x: [-3.5, 3.5], y: [-3.5, 3.5] }
/** Sweeps of the zig-zag run; the player walks through them. */
const ZIGZAG_SWEEPS = 60

export function GibbsZigZag() {
  const state = useFigureState({
    target: row('1 · target', { rho: slider(-0.99, 0.99, 0.9, { label: 'correlation ρ', step: 0.01 }) }),
    ...startSliders(GIBBS_BOX, [-3, 2.5]),
  })
  const { rho } = state.target
  const [sweep, setSweep] = useState(0)
  const start = useMemo((): [number, number] => [state.sx, state.sy], [state.sx, state.sy])
  const target = useMemo(
    () =>
      gaussianTarget(
        [0, 0],
        [
          [1, rho],
          [rho, 1],
        ],
      ),
    [rho],
  )
  const surface = useMemo(() => densityGrid(target, GIBBS_BOX), [target])
  const alg = useMemo(() => gibbs(bivariateGaussianConditionals(rho)), [rho])
  const run = useMemo(() => {
    const tr = trace(alg, { x0: start }, ZIGZAG_SWEEPS, { stream: stream('gibbs-zigzag') })
    const moves: number[] = [...start]
    for (const s of tr.steps.slice(1)) moves.push(...toFlat(s.moves).slice(2))
    return {
      moves: columns(moves, 2),
      ends: columns(
        tr.steps.flatMap((s) => toFlat(s.x)),
        2,
      ),
    }
  }, [alg, start])
  // Sweep k has made 2k coordinate moves from the start, and ended k times.
  const path = useMemo(
    () => ({
      moves: { xs: run.moves.xs.slice(0, 1 + 2 * sweep), ys: run.moves.ys.slice(0, 1 + 2 * sweep) },
      ends: { xs: run.ends.xs.slice(0, sweep + 1), ys: run.ends.ys.slice(0, sweep + 1) },
    }),
    [run, sweep],
  )
  const long = useMemo(() => {
    const tr = trace(alg, { x0: [0, 0] }, 4000, { stream: stream('gibbs-long'), record: { x: (s) => toFlat(s.x)[0] } })
    const x = toFlat(tr.series.x).slice(1)
    return { iact: integratedAutocorrelationTime(x) }
  }, [alg])
  const gx = useAxis({ label: 'x₀', range: GIBBS_BOX.x })
  const gy = useAxis({ label: 'x₁', range: GIBBS_BOX.y, equal: gx })
  return (
    <Figure
      title="Gibbs sampling zig-zags along the axes"
      purpose="Each update moves one coordinate to a draw from its conditional N(ρ·other, 1 − ρ²), so a strong correlation makes every step short."
      state={state}
      controls={<Player label="2 · sweep" value={sweep} onChange={setSweep} count={ZIGZAG_SWEEPS + 1} />}
      readouts={{
        mixing: (
          <>
            <Readout label="conditional sd √(1 − ρ²)" value={f3(Math.sqrt(1 - rho * rho))} />
            <Readout label="integrated autocorrelation time of x₀ (4000 sweeps)" value={f3(long.iact)} />
          </>
        ),
      }}
      caption="Play the sweeps. The path alternates horizontal (x₀ update) and vertical (x₁ update) moves; points mark the end of each sweep. Drag the start. As |ρ| → 1 the steps shrink and the autocorrelation time, (1 + ρ²)/(1 − ρ²) for this sampler, grows without bound."
    >
      <Plot x={gx} y={gy}>
        <Raster {...surface} valueLabel="π" colorBar={false} fillOpacity={0.7} />
        <Curve name="coordinate moves" x={path.moves.xs} y={path.moves.ys} slot={1} live />
        <Points name="end of each sweep" x={path.ends.xs} y={path.ends.ys} slot={2} live />
        <Handle {...state.handle(['sx', 'sy'], { label: 'start' })} />
      </Plot>
    </Figure>
  )
}

export function GibbsChains() {
  const state = useFigureState({
    target: row('1 · target', { rho: slider(0, 0.995, 0.95, { label: 'correlation ρ', step: 0.005 }) }),
    run: row('2 · run', { steps: slider(100, 5000, 1000, { label: 'sweeps', step: 100 }) }),
  })
  const { rho } = state.target
  const { steps } = state.run
  const result = useComputed(
    () =>
      sampleChains(
        gibbs(bivariateGaussianConditionals(rho)),
        (k: number) => ({ x0: [3 * Math.cos(k * 1.7), 3 * Math.sin(k * 1.7)] }),
        { chains: 4, steps, stream: stream('gibbs-chains') },
      ),
    [rho, steps],
    { mode: 'release' },
  )
  return (
    <Figure
      title="Gibbs chains on a correlated Gaussian"
      purpose="Slow mixing shows as long trends in the trace, a slowly decaying ACF and a small ESS: a strong correlation makes Gibbs chains crawl."
      defaultSize="L"
      state={state}
      caption="Four chains start on a circle of radius 3. With ρ = 0.95 the ACF decays slowly and the bulk ESS is a small fraction of the draws; lower ρ and the chains mix within a few sweeps."
    >
      <ChainPanel
        draws={result.value.draws}
        steps={result.value.steps}
        names={['x₀', 'x₁']}
        density={(x) => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI)}
        means={[0, 0]}
      />
    </Figure>
  )
}

// ── Block Gibbs and Rao–Blackwellisation ──────────────────────────────────────────────────────────────────────────

const BLOCK_SAMPLERS = [
  { key: 'single', name: 'single-site {x₀}, {x₁}, {x₂}', partition: [[0], [1], [2]] },
  { key: 'block', name: 'block {x₀, x₁}, {x₂}', partition: [[0, 1], [2]] },
] as const
const ACF_LAGS = 40
const TRACE_SHOWN = 300

export function BlockGibbs() {
  const state = useFigureState({
    target: row('1 · target', {
      rho: slider(0, 0.995, 0.98, { label: 'correlation of x₀ and x₁ ρ', step: 0.005 }),
      c: slider(0, 0.6, 0.4, { label: 'correlation of each with x₂', step: 0.05 }),
    }),
    run: row('2 · run', { steps: slider(200, 4000, 1000, { label: 'sweeps per chain', step: 100 }) }),
  })
  const { rho, c } = state.target
  const { steps } = state.run
  const result = useComputed(
    () => {
      const cov = [
        [1, rho, c],
        [rho, 1, c],
        [c, c, 1],
      ]
      return BLOCK_SAMPLERS.map(({ key, name, partition }) => {
        const blocks = gaussianConditionals(
          [0, 0, 0],
          cov,
          partition.map((b) => [...b]),
        )
        const { draws } = sampleChains(
          gibbs(blocks),
          (k: number) => ({ x0: [3 * Math.cos(k * 1.7), 3 * Math.sin(k * 1.7), 0] }),
          { chains: 4, steps, stream: stream(`block-gibbs-${key}`) },
        )
        const [m, n] = draws.shape
        const flat = toFlat(draws)
        const x0 = Array.from({ length: m }, (_, k) => Array.from({ length: n }, (_, i) => flat[(k * n + i) * 3]))
        // E[x | x outside each block] at every draw: its x₀ component is a chain of its own.
        const rb = toFlat(raoBlackwell(draws, conditionalMean(blocks)).values)
        const rb0 = Array.from({ length: m }, (_, k) => Array.from({ length: n }, (_, i) => rb[(k * n + i) * 3]))
        return {
          name,
          trace: x0[0].slice(0, TRACE_SHOWN),
          acf: toFlat(autocorrelation(x0[0], { maxLag: ACF_LAGS })),
          ess: effectiveSampleSize(x0),
          draws: m * n,
          mcse: monteCarloStandardError(x0),
          mcseRb: monteCarloStandardError(rb0),
        }
      })
    },
    [rho, c, steps],
    { mode: 'release' },
  )
  const r = result.value
  const iterations = useMemo(() => Array.from({ length: TRACE_SHOWN }, (_, i) => i + 1), [])
  const lags = useMemo(() => Array.from({ length: ACF_LAGS + 1 }, (_, k) => k), [])
  const tx = useAxis({ label: 'sweep', range: [1, TRACE_SHOWN], integer: true })
  const ty = useAxis({ label: 'x₀ (chain 1)', range: [-4, 4] })
  const lx = useAxis({ label: 'lag', range: [0, ACF_LAGS], integer: true })
  const ly = useAxis({ label: 'autocorrelation of x₀', range: [-0.2, 1] })
  const group = (k: number) => (
    <>
      <Readout label="bulk ESS of x₀" value={`${f3(r[k].ess)} of ${r[k].draws}`} />
      <Readout label="MCSE of the mean of x₀, draws" value={f3(r[k].mcse)} />
      <Readout label="MCSE, Rao–Blackwellised" value={f3(r[k].mcseRb)} />
    </>
  )
  return (
    <Figure
      title="Block Gibbs and Rao–Blackwellisation"
      purpose="Drawing the strongly correlated pair (x₀, x₁) as one block removes the zig-zag that slows single-site Gibbs; averaging the conditional mean E[x₀ | rest] in place of the draws lowers the Monte Carlo error further."
      description="A trivariate Gaussian target, zero mean and unit variances; four chains per sampler from gaussianConditionals with two partitions, and raoBlackwell with conditionalMean."
      defaultSize="L"
      state={state}
      readouts={{ [BLOCK_SAMPLERS[0].name]: group(0), [BLOCK_SAMPLERS[1].name]: group(1) }}
      caption="Left: the first 300 sweeps of x₀ in chain 1 for each sampler. Right: the autocorrelation of x₀. With ρ near 1 the single-site chain wanders and its autocorrelation decays over tens of lags, while the block chain's falls almost at once. The Rao–Blackwellised estimate averages E[x₀ | the coordinates outside its block]: for the block sampler that is a multiple of x₂, which varies far less than x₀ itself."
    >
      <Plots cols={2} widths={[3, 2]}>
        <Plot x={tx} y={ty} legend>
          {r.map((s, k) => (
            <Curve key={s.name} name={s.name} x={iterations} y={s.trace} slot={k} stale={result.stale} />
          ))}
        </Plot>
        <Plot x={lx} y={ly} legend>
          {r.map((s, k) => (
            <Curve key={s.name} name={s.name} x={lags} y={s.acf} slot={k} showPoints stale={result.stale} />
          ))}
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── HMC trajectories ────────────────────────────────────────────────────────────────────────────────────────────────

const CORRELATED_BOX: Box = { x: [-3.5, 3.5], y: [-3.5, 3.5] }
const FUNNEL_BOX: Box = { x: [-7, 7], y: [-8, 8] }
const HMC_TARGETS: Record<string, { target: LogDensity; box: Box }> = {
  banana: { target: BANANA, box: BANANA_BOX },
  'correlated Gaussian': {
    target: gaussianTarget(
      [0, 0],
      [
        [1, 0.95],
        [0.95, 1],
      ],
    ),
    box: CORRELATED_BOX,
  },
  funnel: { target: funnel(), box: FUNNEL_BOX },
}
const HMC_TARGET = variants(
  {
    banana: { label: 'banana', params: startSliders(BANANA_BOX, [-2, 4]) },
    'correlated Gaussian': { label: 'correlated Gaussian', params: startSliders(CORRELATED_BOX, [-2.5, -1]) },
    funnel: { label: 'funnel', params: startSliders(FUNNEL_BOX, [2, 3]) },
  },
  { label: '1 · target', choiceLabel: 'target' },
)

const NUTS_VARIANTS = [
  { value: 'multinomial', label: 'multinomial (Stan)' },
  { value: 'slice', label: 'slice (Hoffman and Gelman)' },
] as const
type NutsVariant = (typeof NUTS_VARIANTS)[number]['value']

const HMC_SAMPLER = variants(
  {
    HMC: { label: 'HMC', params: { L: slider(1, 80, 20, { label: 'leapfrog steps L', step: 1 }) } },
    NUTS: { label: 'NUTS', params: { variant: choice(NUTS_VARIANTS, 'multinomial', { label: 'NUTS variant' }) } },
  },
  {
    label: '2 · sampler',
    choiceLabel: 'sampler',
    shared: { eps: slider(0.01, 1.5, 0.25, { label: 'step size ε', step: 0.01 }) },
  },
)

export function HmcTrajectories() {
  const state = useFigureState({ target: HMC_TARGET, sampler: HMC_SAMPLER })
  const which = state.target.key
  const sampler = state.sampler.key
  const { eps } = state.sampler.values
  const L = state.sampler.key === 'HMC' ? state.sampler.values.L : 0
  const variant: NutsVariant =
    state.sampler.key === 'NUTS' ? (state.sampler.values.variant as NutsVariant) : 'multinomial'
  const start = useMemo(
    (): [number, number] => [state.target.values.sx, state.target.values.sy],
    [state.target.values.sx, state.target.values.sy],
  )
  const [at, setAt] = useState(0)
  const spec = HMC_TARGETS[which]
  const surface = useMemo(() => densityGrid(spec.target, spec.box), [spec])
  const N = 30
  // 30 trajectories: rerun on release while the start is dragged.
  const run = useComputed(
    (): Trace<HmcState> | Trace<NutsState> => {
      const options = { stream: stream('hmc-trajectories') }
      return sampler === 'HMC'
        ? trace(hmc(spec.target, { stepSize: eps, steps: L }), { x0: start }, N, options)
        : trace(nuts(spec.target, { stepSize: eps, variant }), { x0: start }, N, options)
    },
    [spec, sampler, eps, L, variant, start],
    { mode: 'release' },
  )
  const tr = run.value
  const i = Math.min(at, tr.steps.length - 1)
  const st = tr.steps[i]
  const layers = useMemo(() => {
    const past = tr.steps.slice(1, i).flatMap((s) => {
      const c = columns(toFlat(s.trajectory), 2)
      return [...c.xs.map((x, k) => [x, c.ys[k]]), [NaN, NaN]]
    })
    const cur = st.trajectory ? columns(toFlat(st.trajectory), 2) : { xs: [], ys: [] }
    const pts = columns(
      tr.steps.slice(0, i + 1).flatMap((s) => toFlat(s.x)),
      2,
    )
    return { past: { x: past.map((p) => p[0]), y: past.map((p) => p[1]) }, cur, pts }
  }, [tr, i, st])
  const energy = useMemo(() => {
    const H = st.energies ? toFlat(st.energies).map((h) => (Number.isFinite(h) ? h : NaN)) : []
    const times =
      'trajectoryTimes' in st && st.trajectoryTimes ? toFlat(st.trajectoryTimes as Vector) : H.map((_, k) => k)
    // Before the first iteration there is no trajectory yet.
    if (H.length < 2) return { times: [], H: [], start: { x: [], y: [] } }
    const h0 = H[times.indexOf(0)]
    return { times, H, start: { x: [Math.min(...times), Math.max(...times)], y: [h0, h0] } }
  }, [st])
  const last = tr.steps[tr.steps.length - 1]
  // The mean distance between successive samples over the whole run: how far each iteration moves the chain.
  const meanJump = useMemo(() => {
    const xs = tr.steps.map((s) => toFlat(s.x))
    let sum = 0
    for (let k = 1; k < xs.length; k++) sum += Math.hypot(xs[k][0] - xs[k - 1][0], xs[k][1] - xs[k - 1][1])
    return xs.length > 1 ? sum / (xs.length - 1) : NaN
  }, [tr])
  const tx = useAxis({ label: 'θ₀', range: spec.box.x })
  const ty = useAxis({ label: 'θ₁', range: spec.box.y })
  // Before the first iteration the energy panel is empty: fixed axes rather than a degenerate fit.
  const empty = energy.H.length === 0
  const ex = useAxis({ label: 'leapfrog step', range: empty ? [0, 1] : undefined })
  const ey = useAxis({ label: 'H', range: empty ? [0, 1] : undefined })
  return (
    <Figure
      title="HMC trajectories and their energy"
      purpose="Leapfrog integration follows the Hamiltonian flow, so H stays nearly constant and distant proposals are accepted; too large a step makes H drift and the proposal is rejected or diverges."
      defaultSize="L"
      state={state}
      controls={<Player label="3 · iteration" value={i} onChange={setAt} count={N + 1} />}
      readouts={{
        'this iteration': (
          <>
            <Readout
              label="energy error ΔH"
              value={
                'energyError' in st
                  ? f3(st.energyError as number)
                  : f3(Math.max(...toFlat(st.energies)) - Math.min(...toFlat(st.energies)))
              }
            />
            <Readout label="accepted" value={st.accepted ? 'yes' : 'no'} />
            {'treeDepth' in st && <Readout label="tree depth" value={st.treeDepth as number} />}
          </>
        ),
        'whole run': (
          <>
            <Readout label="acceptance rate" value={f3(last.acceptanceRate)} />
            <Readout label="divergent trajectories" value={last.divergentCount} />
            <Readout label="mean jump |θₜ − θₜ₋₁|" value={f3(meanJump)} />
          </>
        ),
      }}
      caption="Play the iterations from the start. Top: every trajectory up to the current iteration (thin), the current one with its leapfrog points, and the samples. Drag the start. Bottom: the Hamiltonian H = −log π + ½|p|² at each leapfrog point of the current trajectory (for NUTS, by integration time, both directions). On the funnel the neck needs a small ε, so a step that suits the mouth diverges there. NUTS picks its sample from the trajectory either in proportion to exp(−H) over every point (multinomial, Stan's default) or uniformly among the points inside a slice drawn under exp(−H₀) (slice, Hoffman and Gelman's Algorithm 3); compare their mean jumps."
    >
      <Plots rows={2} heights={[2, 1.2]}>
        <Plot x={tx} y={ty}>
          <Raster {...surface} valueLabel="π" colorBar={false} fillOpacity={0.7} />
          <Curve name="earlier trajectories" {...layers.past} slot={1} thin stale={run.stale} />
          <Curve name="current trajectory" x={layers.cur.xs} y={layers.cur.ys} slot={2} showPoints stale={run.stale} />
          <Points name="samples" x={layers.pts.xs} y={layers.pts.ys} slot={3} stale={run.stale} />
          <Handle {...state.handle(['target.sx', 'target.sy'], { label: 'start' })} />
        </Plot>
        <Plot x={ex} y={ey}>
          <Curve name="H along the trajectory" x={energy.times} y={energy.H} slot={2} showPoints />
          <Curve name="H at the start" {...energy.start} muted dashed />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── Warmup: dual-averaging step-size adaptation ───────────────────────────────────────────────────────────────────

const ADAPT_SAMPLER = variants(
  {
    NUTS: { label: 'NUTS', params: { variant: choice(NUTS_VARIANTS, 'multinomial', { label: 'NUTS variant' }) } },
    HMC: { label: 'HMC', params: { L: slider(1, 40, 10, { label: 'leapfrog steps L', step: 1 }) } },
  },
  { label: '2 · sampler', choiceLabel: 'sampler' },
)

export function WarmupAdaptation() {
  const state = useFigureState({
    target: choice(Object.keys(HMC_TARGETS), 'correlated Gaussian', { label: '1 · target' }),
    sampler: ADAPT_SAMPLER,
    adapt: row('3 · adaptation', {
      eps0: slider(0.01, 2, 1.5, { label: 'initial step size ε₀', step: 0.01 }),
      delta: slider(0.5, 0.95, 0.8, { label: 'target acceptance δ', step: 0.01 }),
      warmup: slider(20, 300, 150, { label: 'warmup steps', step: 10 }),
    }),
  })
  const spec = HMC_TARGETS[state.target]
  const sampler = state.sampler.key
  const L = state.sampler.key === 'HMC' ? state.sampler.values.L : 0
  const variant: NutsVariant =
    state.sampler.key === 'NUTS' ? (state.sampler.values.variant as NutsVariant) : 'multinomial'
  const { eps0, delta, warmup } = state.adapt
  // Warmup and as many sampling steps again: ε is frozen at ε̄ after warmup.
  const N = 2 * warmup
  const run = useComputed(
    () => {
      const adapt = { warmup, targetAcceptance: delta }
      const options = { stream: stream('hmc-warmup') }
      const start = { x0: [0.5, 0.5] }
      const steps =
        sampler === 'HMC'
          ? trace(hmc(spec.target, { stepSize: eps0, steps: L, adapt }), start, N, options).steps
          : trace(nuts(spec.target, { stepSize: eps0, variant, adapt }), start, N, options).steps
      // Step 0 is the start, before any transition: its acceptance statistic is undefined.
      const rest = steps.slice(1)
      const iteration = rest.map((_, k) => k + 1)
      const stepSize = rest.map((st) => st.stepSize)
      const stepSizeBar = rest.map((st) => st.stepSizeBar)
      const accept = rest.map((st) => st.acceptStat)
      // The running mean of the acceptance statistic over the warmup, which dual averaging steers towards δ.
      let sum = 0
      const runningMean = accept.map((a, k) => (k < warmup ? (sum += a) / (k + 1) : NaN))
      const post = accept.slice(warmup)
      return {
        iteration,
        stepSize,
        stepSizeBar,
        accept,
        runningMean,
        postMean: post.reduce((a, b) => a + b, 0) / Math.max(1, post.length),
        divergent: rest.at(-1)?.divergentCount ?? 0,
      }
    },
    [spec, sampler, L, variant, eps0, delta, warmup, N],
    { mode: 'release' },
  )
  const r = run.value
  const [at, setAt] = useState(0)
  const i = Math.min(at, r.iteration.length)
  const walked = useMemo(
    () => ({
      x: r.iteration.slice(0, i),
      eps: r.stepSize.slice(0, i),
      bar: r.stepSizeBar.slice(0, i),
      accept: r.accept.slice(0, i),
      mean: r.runningMean.slice(0, i),
    }),
    [r, i],
  )
  const x = useAxis({ label: 'iteration', range: [0, N], integer: true })
  const yEps = useAxis({ label: 'step size ε', log: true })
  const yAccept = useAxis({ label: 'acceptance statistic', range: [0, 1] })
  const cursor = i > 0 ? i : undefined
  return (
    <Figure
      title="Warmup: dual-averaging step-size adaptation"
      purpose="During warmup, dual averaging moves log ε against the error δ − a of each step's acceptance statistic a, so the running mean of a reaches δ; after warmup ε is frozen at the averaged ε̄."
      description="Hoffman and Gelman (2014), §3.2: the same adaptation for HMC (Algorithm 5) and NUTS (Algorithm 6)."
      state={state}
      controls={<Player label="4 · iteration" value={i} onChange={setAt} count={r.iteration.length + 1} />}
      readouts={{
        'this iteration': (
          <>
            <Readout label="phase" value={i === 0 ? '—' : i <= warmup ? 'warmup' : 'sampling'} />
            <Readout label="ε used" value={i > 0 ? f3(r.stepSize[i - 1]) : '—'} />
            <Readout label="ε̄ so far" value={i > 0 ? f3(r.stepSizeBar[i - 1]) : '—'} />
            <Readout label="acceptance statistic" value={i > 0 ? f3(r.accept[i - 1]) : '—'} />
          </>
        ),
        'whole run': (
          <>
            <Readout label="final ε = ε̄" value={f3(r.stepSizeBar.at(-1) ?? NaN)} />
            <Readout label="mean acceptance after warmup" value={f3(r.postMean)} />
            <Readout label="divergent trajectories" value={r.divergent} />
          </>
        ),
      }}
      caption="Play the iterations. Faint lines show the whole run; the solid ones the iterations so far. Top: the step size each iteration used (log scale) and the average ε̄ it converges to. Bottom: each iteration's acceptance statistic (points), its running mean over warmup, and the target δ. A large ε₀ starts with low acceptance, so ε shrinks; a small ε₀ starts near 1, so ε grows."
    >
      <Plots rows={2} heights={[1, 1]} hoverGroup>
        <Plot x={x} y={yEps} legend>
          <Curve name="ε (whole run)" x={r.iteration} y={r.stepSize} slot={0} thin muted stale={run.stale} />
          <Curve name="ε used" x={walked.x} y={walked.eps} slot={0} live stale={run.stale} />
          <Curve name="ε̄" x={walked.x} y={walked.bar} slot={1} live stale={run.stale} />
          <Annotation x={warmup} text="end of warmup" dashed />
          {cursor !== undefined && <Annotation x={cursor} />}
        </Plot>
        <Plot x={x} y={yAccept} legend>
          <Points name="acceptance (whole run)" x={r.iteration} y={r.accept} slot={0} muted stale={run.stale} />
          <Curve
            name="running mean (whole run)"
            x={r.iteration}
            y={r.runningMean}
            slot={1}
            thin
            muted
            stale={run.stale}
          />
          <Points name="acceptance statistic" x={walked.x} y={walked.accept} slot={0} live stale={run.stale} />
          <Curve name="running mean (warmup)" x={walked.x} y={walked.mean} slot={1} live stale={run.stale} />
          <Annotation y={delta} text="δ" dashed />
          <Annotation x={warmup} dashed />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── MALA against ULA ───────────────────────────────────────────────────────────────────────────────────────────────

const STD_NORMAL = gaussianTarget([0], [[1]])
const H_GRID = grid(0.05, 1.5, 12)
const DENSITY_X = grid(-4.5, 4.5, 181)
const DENSITY_Y = DENSITY_X.map((x) => Math.exp(STD_NORMAL.logDensity(tensor([x])) as number))
const EDGES = grid(-4.5, 4.5, 46)

function langevinDraws(kind: 'ula' | 'mala', h: number, steps: number) {
  const alg = kind === 'ula' ? unadjustedLangevin(STD_NORMAL, { stepSize: h }) : mala(STD_NORMAL, { stepSize: h })
  const { draws } = sampleChains(
    alg,
    { x0: [0] },
    { chains: 2, steps, stream: stream(`langevin-${kind}`), warmup: 100 },
  )
  return toFlat(draws)
}

export function LangevinBias() {
  const state = useFigureState({ h: slider(0.05, 1.5, 0.6, { label: 'step size h', step: 0.05 }) })
  const h = state.h
  // 2 × 2 chains of 10 000 steps: rerun on release while h is dragged.
  const view = useComputed(
    () => {
      const ula = langevinDraws('ula', h, 10000)
      const ma = langevinDraws('mala', h, 10000)
      const bars = (x: number[]) => {
        const hs = histogramBars(histogram(x, { bins: EDGES }))
        return { x: hs.x, y: hs.density, edges: hs.edges }
      }
      return {
        ula: bars(ula),
        mala: bars(ma),
        ulaVar: variance(tensor(ula), null, false, 1),
        malaVar: variance(tensor(ma), null, false, 1),
      }
    },
    [h],
    { mode: 'release' },
  )
  const curve = useMemo(() => {
    const u = H_GRID.map((v) => variance(tensor(langevinDraws('ula', v, 4000)), null, false, 1))
    const m = H_GRID.map((v) => variance(tensor(langevinDraws('mala', v, 4000)), null, false, 1))
    return { u, m }
  }, [])
  const v = view.value
  const x = useAxis({ label: 'x', range: [-4.5, 4.5] })
  const d = useAxis({ label: 'density', hold: 'union' })
  const hx = useAxis({ label: 'step size h' })
  const vy = useAxis({ label: 'variance' })
  return (
    <Figure
      title="MALA corrects the bias of unadjusted Langevin"
      purpose="ULA keeps every Euler step and so samples a law wider than the target by O(h); MALA's Metropolis test removes the bias."
      state={state}
      readouts={{
        'at h': (
          <>
            <Readout label="ULA variance" value={f3(v.ulaVar)} />
            <Readout label="MALA variance" value={f3(v.malaVar)} />
          </>
        ),
      }}
      caption="Target N(0, 1); two chains of 10 000 steps each. Left: the histograms of ULA and MALA draws against the target. Right: the sample variance of each sampler over step sizes (4000 steps per point); drag the vertical line to set h. ULA's variance grows like 1/(1 − h/2); MALA stays at 1 while its acceptance rate falls."
    >
      <Plots cols={2}>
        <Plot x={x} y={d}>
          <Bars name="ULA draws" x={v.ula.x} y={v.ula.y} edges={v.ula.edges} slot={0} stale={view.stale} />
          <Curve name="MALA draws" x={v.mala.x} y={v.mala.y} slot={1} showPoints stale={view.stale} />
          <Curve name="target N(0, 1)" x={DENSITY_X} y={DENSITY_Y} emphasis />
        </Plot>
        <Plot x={hx} y={vy}>
          <Curve name="ULA variance" x={H_GRID} y={curve.u} slot={0} showPoints />
          <Curve name="MALA variance" x={H_GRID} y={curve.m} slot={1} showPoints />
          <Curve name="target variance 1" x={[H_GRID[0], H_GRID[H_GRID.length - 1]]} y={[1, 1]} muted dashed />
          <Handle {...state.handle('h', { label: 'h' })} />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── Diagnostics ─────────────────────────────────────────────────────────────────────────────────────────────────────

export function Diagnostics() {
  const state = useFigureState({
    run: row('1 · chains', {
      chains: slider(1, 8, 4, { label: 'chains', step: 1 }),
      steps: slider(50, 5000, 400, { label: 'steps per chain', step: 50 }),
    }),
    proposal: row('2 · proposal', { scale: slider(0.05, 3, 0.3, { label: 'proposal scale σ', step: 0.05 }) }),
  })
  const { chains, steps } = state.run
  const { scale } = state.proposal
  const result = useComputed(
    () => {
      const starts = stream('dispersed')
      return sampleChains(
        randomWalkMetropolis(BANANA, { scale }),
        (k: number) => ({ x0: [normal(child(starts, k, 0), 0, 2), normal(child(starts, k, 1), 3, 3)] }),
        { chains, steps, stream: stream('diagnostics') },
      )
    },
    [chains, steps, scale],
    { mode: 'release' },
  )
  return (
    <Figure
      title="ESS and R̂ for random-walk Metropolis on the banana"
      purpose="Chains from dispersed starts disagree until they have forgotten their starts: R̂ stays well above 1 and the ESS is tiny until then."
      defaultSize="L"
      state={state}
      caption="With 400 steps and σ = 0.3 the chains are still apart: R̂ is well above 1.01 and the ESS is tiny. Raise the steps (or tune σ near 1) and R̂ falls towards 1 while the ESS grows."
    >
      <ChainPanel draws={result.value.draws} steps={result.value.steps} names={['x', 'y']} means={[0, 0]} />
    </Figure>
  )
}

// ── Particle filter ─────────────────────────────────────────────────────────────────────────────────────────────────

/** The nonlinear growth model (Kitagawa, 1996; Gordon et al., 1993): its observation y = x²/20 hides the sign of x. */
const GROWTH = {
  dim: 1,
  sampleInitial: (s: ReturnType<typeof stream>) => normal(s, 0, Math.sqrt(5)),
  sampleTransition: (x: Vector, t: number, s: ReturnType<typeof stream>) => {
    const v = toFlat(x)[0]
    return normal(s, v / 2 + (25 * v) / (1 + v * v) + 8 * Math.cos(1.2 * t), Math.sqrt(10))
  },
  logObservation: (y: number, x: Vector) => {
    const v = toFlat(x)[0]
    return -0.5 * (y - (v * v) / 20) ** 2 - 0.5 * Math.log(2 * Math.PI)
  },
}
const T = 60
const TIMES = Array.from({ length: T }, (_, t) => t)

export function ParticleFilter() {
  const state = useFigureState({
    filter: row('1 · filter', {
      N: slider(20, 3000, 500, { label: 'particles N', step: 10 }),
      scheme: choice(resamplingSchemes, 'systematic', { label: 'resampling' }),
      threshold: slider(0, 1, 0.5, { label: 'resample when ESS / N <', step: 0.05 }),
    }),
  })
  const { N, threshold } = state.filter
  const scheme = state.filter.scheme as ResamplingScheme
  const [at, setAt] = useState(0)
  const world = useMemo(() => {
    const s = stream('growth-model')
    const xs: number[] = []
    const ys: number[] = []
    let x = GROWTH.sampleInitial(child(s, 'x0'))
    for (let t = 0; t < T; t++) {
      if (t > 0) x = GROWTH.sampleTransition(tensor([x]), t, child(s, 'x', t))
      xs.push(x)
      ys.push(normal(child(s, 'y', t), (x * x) / 20, 1))
    }
    return { xs, ys }
  }, [])
  // Up to 3000 particles × 60 steps: rerun on release.
  const run = useComputed(
    () =>
      trace(particleFilter(GROWTH, { particles: N, resampling: scheme, threshold }), { observations: world.ys }, T, {
        stream: stream('pf'),
        record: {
          mean: (s) => s.mean,
          variance: (s) => s.variance,
          ess: (s) => s.ess,
          resampled: (s) => (s.resampled ? 1 : 0),
        },
      }),
    [N, scheme, threshold, world],
    { mode: 'release' },
  )
  const tr = run.value
  // Player position 0 is the first filtered step (t = 0).
  const i = clamp(at + 1, 1, T)
  const st = tr.steps[i]
  const band = useMemo(() => {
    const mean = toFlat(tr.series.mean).slice(1)
    const sd = toFlat(tr.series.variance).slice(1).map(Math.sqrt)
    return { mean, hi: mean.map((m, k) => m + 2 * sd[k]), lo: mean.map((m, k) => m - 2 * sd[k]) }
  }, [tr])
  const cloud = useMemo(() => {
    const p = toFlat(st.particles)
    const keep = p.filter((_, k) => k % Math.max(1, Math.floor(N / 300)) === 0)
    return { x: keep.map(() => i - 1), y: keep }
  }, [st, i, N])
  const essSeries = useMemo(() => {
    const ess = toFlat(tr.series.ess).slice(1)
    const res = toFlat(tr.series.resampled).slice(1)
    return { ess, rx: TIMES.filter((_, k) => res[k]), ry: ess.filter((_, k) => res[k]) }
  }, [tr])
  const last = tr.steps[tr.steps.length - 1]
  const resamples = toFlat(tr.series.resampled).reduce((a, b) => a + b, 0)
  const t = useAxis({ label: 't', range: [0, T - 1] })
  const xa = useAxis({ label: 'x' })
  const ea = useAxis({ label: 'ESS', range: [0, undefined] })
  return (
    <Figure
      title="A bootstrap particle filter on the nonlinear growth model"
      purpose="Particles move through the transition, are weighted by the observation and resampled when the ESS falls; y = x²/20 hides the sign of x, so the filtering distribution is often bimodal."
      defaultSize="L"
      state={state}
      controls={<Player label="2 · time" value={i - 1} onChange={setAt} count={T} format={(k) => `t = ${k}`} />}
      readouts={{
        'at t': <Readout label="ESS at t" value={f3(st.ess)} />,
        'whole run': (
          <>
            <Readout label="resampling steps" value={resamples} />
            <Readout label="log p(y₀:T) estimate" value={f3(last.logEvidence)} />
          </>
        ),
      }}
      caption="xₜ = xₜ₋₁/2 + 25xₜ₋₁/(1 + xₜ₋₁²) + 8 cos(1.2t) + N(0, 10), yₜ = xₜ²/20 + N(0, 1). Top: the true state, the filter mean ± 2 sd and the particle cloud at the player's time (up to 300 particles). Bottom: the ESS after weighting, with points where the particles were resampled. With few particles or threshold 0 (never resample) the weights collapse and the filter loses the state."
    >
      <Plots rows={2} heights={[2, 1.2]} hoverGroup>
        <Plot x={t} y={xa}>
          <Curve name="true state" x={TIMES} y={world.xs} emphasis />
          <Curve name="filter mean" x={TIMES} y={band.mean} slot={0} stale={run.stale} />
          <Curve name="mean ± 2 sd" x={TIMES} y={band.hi} slot={0} dashed stale={run.stale} />
          <Curve name="mean ± 2 sd" x={TIMES} y={band.lo} slot={0} dashed stale={run.stale} />
          <Points name="particles at t" x={cloud.x} y={cloud.y} slot={2} thin stale={run.stale} />
        </Plot>
        <Plot x={t} y={ea}>
          <Curve name="ESS" x={TIMES} y={essSeries.ess} slot={0} stale={run.stale} />
          <Points name="resampled" x={essSeries.rx} y={essSeries.ry} slot={1} />
          <Curve name="threshold" x={[0, T - 1]} y={[threshold * N, threshold * N]} muted dashed />
        </Plot>
      </Plots>
    </Figure>
  )
}
