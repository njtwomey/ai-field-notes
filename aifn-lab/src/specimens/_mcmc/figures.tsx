import {
  banana,
  bivariateGaussianConditionals,
  effectiveSampleSize,
  funnel,
  gaussianTarget,
  gibbs,
  hmc,
  integratedAutocorrelationTime,
  mala,
  nuts,
  particleFilter,
  randomWalkMetropolis,
  resamplingSchemes,
  sampleChains,
  unadjustedLangevin,
  type HmcState,
  type NutsState,
  type ResamplingScheme,
  type Target,
} from 'aifn/mcmc'
import { normal, stream } from 'aifn/random'
import { histogram, variance } from 'aifn/stats'
import { linspace, tensor, toFlat, type Vector } from 'aifn/tensor'
import { trace, type Trace } from 'aifn/trace'
import { useMemo, useState } from 'react'
import { Player, Select, Slider } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { Heatmap, Panel, Readout, Subplots, XYChart, type HeatmapOverlay, type XYSeries } from '@lab/viz'
import { ChainView, formatValue } from '@lab/views'

const f3 = (v: number) => formatValue(Number(v.toPrecision(3)))
const grid = (lo: number, hi: number, n: number) => toFlat(linspace(lo, hi, n))
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))

type Box = { x: [number, number]; y: [number, number] }

/** The target density exp(log π) on a grid, for the heatmap. */
function densityGrid(target: Target, box: Box, n = 90) {
  const x = grid(box.x[0], box.x[1], n)
  const y = grid(box.y[0], box.y[1], n)
  const z = y.map((yi) => x.map((xj) => Math.exp(target.logDensity(tensor([xj, yi])) as number)))
  return { x, y, z }
}

/** Columns 0 and 1 of an n×d matrix (or of the rows of stacked states). */
function columns(flat: readonly number[], d: number) {
  const n = flat.length / d
  return {
    xs: Array.from({ length: n }, (_, i) => flat[i * d]),
    ys: Array.from({ length: n }, (_, i) => flat[i * d + 1]),
  }
}

// ── Random-walk Metropolis on a banana ─────────────────────────────────────────────────────────────────────────────

const BANANA = banana({ a: 1, b: 1 })
const BANANA_BOX: Box = { x: [-3.5, 3.5], y: [-2.5, 8] }
const SCALES = grid(Math.log10(0.03), Math.log10(10), 25).map((v) => 10 ** v)

export function MetropolisBanana() {
  const [scale, setScale] = useState(0.6)
  const [steps, setSteps] = useState(1500)
  const [start, setStart] = useState<[number, number]>([-2.5, 6])
  const surface = useMemo(() => densityGrid(BANANA, BANANA_BOX), [])

  const chain = useMemo(() => {
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
    return rows
  }, [])

  const overlay = useMemo<HeatmapOverlay[]>(
    () => [
      { name: 'rejected proposals', type: 'scatter', x: chain.rejected.xs, y: chain.rejected.ys, slot: 3 },
      { name: 'chain', type: 'line', x: chain.path.xs, y: chain.path.ys, slot: 1, thin: true },
    ],
    [chain],
  )
  const sweepSeries = useMemo<XYSeries[]>(
    () => [
      { name: 'acceptance rate', type: 'line', x: SCALES, y: sweep.map((r) => r.rate), slot: 0, showPoints: true },
      { name: 'ESS per step (y)', type: 'line', x: SCALES, y: sweep.map((r) => r.ess), slot: 1, showPoints: true },
      {
        name: '0.234',
        type: 'line',
        x: [SCALES[0], SCALES[SCALES.length - 1]],
        y: [0.234, 0.234],
        muted: true,
        dashed: true,
      },
    ],
    [sweep],
  )
  return (
    <Figure
      title="Random-walk Metropolis on a banana"
      description="The proposal scale trades acceptance against step length: tiny steps are almost always accepted but crawl, large steps are almost always rejected; efficiency peaks in between."
      defaultSize="L"
      controls={
        <>
          <ControlRow label="1 · proposal">
            <Slider label="proposal scale σ" value={scale} onChange={setScale} min={0.03} max={10} step={0.01} />
          </ControlRow>
          <ControlRow label="2 · run">
            <Slider label="steps" value={steps} onChange={setSteps} min={100} max={5000} step={100} />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout label="acceptance rate" value={f3(chain.rate)} />
          <Readout label="integrated autocorrelation time of y" value={f3(chain.iact)} />
        </>
      }
      caption="Left: the banana log π = −x²/2 − (y − (x² − 1))²/2 with the chain (line) and up to 150 rejected proposals (points); drag the start. Right: over proposal scales from 0.03 to 10 (log axis), the acceptance rate and the effective sample size per step of y, from one 2000-step chain each; drag the vertical line to set σ. Both axes are logarithmic. The ESS per step peaks at σ between 1 and 3, where the acceptance rate is 0.2–0.4 (the dashed line marks 0.234); smaller steps are accepted more often but explore less."
    >
      <Subplots cols={2} widthRatios={[1.2, 1]}>
        <Panel>
          <Heatmap
            {...surface}
            xLabel="x"
            yLabel="y"
            valueLabel="π(x, y)"
            colorBar={false}
            fillOpacity={0.7}
            overlay={overlay}
            handles={[
              {
                kind: 'point',
                at: start,
                onDrag: ([x, y]) => setStart([clamp(x, -3.4, 3.4), clamp(y, -2.4, 7.9)]),
                label: 'start',
              },
            ]}
          />
        </Panel>
        <Panel>
          <XYChart
            series={sweepSeries}
            xLog
            yLog
            xLabel="proposal scale σ"
            yLabel="rate (log scale)"
            handles={[{ kind: 'x', at: scale, onDrag: (v) => setScale(clamp(v, 0.03, 10)), label: 'σ' }]}
          />
        </Panel>
      </Subplots>
    </Figure>
  )
}

// ── Gibbs zig-zag ───────────────────────────────────────────────────────────────────────────────────────────────────

const GIBBS_BOX: Box = { x: [-3.5, 3.5], y: [-3.5, 3.5] }

export function GibbsZigZag() {
  const [rho, setRho] = useState(0.9)
  const [sweeps, setSweeps] = useState(15)
  const [start, setStart] = useState<[number, number]>([-3, 2.5])
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
  const path = useMemo(() => {
    const tr = trace(alg, { x0: start }, sweeps, { stream: stream('gibbs-zigzag') })
    const moves: number[] = [...start]
    for (const s of tr.steps.slice(1)) moves.push(...toFlat(s.moves).slice(2))
    const sweepEnds = columns(
      tr.steps.flatMap((s) => toFlat(s.x)),
      2,
    )
    return { moves: columns(moves, 2), ends: sweepEnds }
  }, [alg, start, sweeps])
  const long = useMemo(() => {
    const tr = trace(alg, { x0: [0, 0] }, 4000, { stream: stream('gibbs-long'), record: { x: (s) => toFlat(s.x)[0] } })
    const x = toFlat(tr.series.x).slice(1)
    return { iact: integratedAutocorrelationTime(x) }
  }, [alg])
  const overlay = useMemo<HeatmapOverlay[]>(
    () => [
      { name: 'coordinate moves', type: 'line', x: path.moves.xs, y: path.moves.ys, slot: 1 },
      { name: 'end of each sweep', type: 'scatter', x: path.ends.xs, y: path.ends.ys, slot: 2 },
    ],
    [path],
  )
  return (
    <Figure
      title="Gibbs sampling zig-zags along the axes"
      description="Each update moves one coordinate to a draw from its conditional N(ρ·other, 1 − ρ²), so a strong correlation makes every step short."
      controls={
        <>
          <ControlRow label="1 · target">
            <Slider label="correlation ρ" value={rho} onChange={setRho} min={-0.99} max={0.99} step={0.01} />
          </ControlRow>
          <ControlRow label="2 · run">
            <Slider label="sweeps" value={sweeps} onChange={setSweeps} min={1} max={60} step={1} />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout label="conditional sd √(1 − ρ²)" value={f3(Math.sqrt(1 - rho * rho))} />
          <Readout label="integrated autocorrelation time of x₀ (4000 sweeps)" value={f3(long.iact)} />
        </>
      }
      caption="The path alternates horizontal (x₀ update) and vertical (x₁ update) moves; points mark the end of each sweep. Drag the start. As |ρ| → 1 the steps shrink and the autocorrelation time, (1 + ρ²)/(1 − ρ²) for this sampler, grows without bound."
    >
      <Heatmap
        {...surface}
        xLabel="x₀"
        yLabel="x₁"
        valueLabel="π"
        colorBar={false}
        fillOpacity={0.7}
        equalAspect
        overlay={overlay}
        handles={[
          {
            kind: 'point',
            at: start,
            onDrag: ([x, y]) => setStart([clamp(x, -3.4, 3.4), clamp(y, -3.4, 3.4)]),
            label: 'start',
          },
        ]}
      />
    </Figure>
  )
}

export function GibbsChains() {
  const [rho, setRho] = useState(0.95)
  const [steps, setSteps] = useState(1000)
  const result = useMemo(
    () =>
      sampleChains(
        gibbs(bivariateGaussianConditionals(rho)),
        (k: number) => ({ x0: [3 * Math.cos(k * 1.7), 3 * Math.sin(k * 1.7)] }),
        {
          chains: 4,
          steps,
          stream: stream('gibbs-chains'),
        },
      ),
    [rho, steps],
  )
  return (
    <ChainView
      title="Gibbs chains on a correlated Gaussian"
      description="Slow mixing shows as long trends in the trace, a slowly decaying ACF and a small ESS."
      draws={result.draws}
      steps={result.steps}
      names={['x₀', 'x₁']}
      density={(x) => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI)}
      means={[0, 0]}
      controls={
        <>
          <Slider label="correlation ρ" value={rho} onChange={setRho} min={0} max={0.995} step={0.005} />
          <Slider label="sweeps" value={steps} onChange={setSteps} min={100} max={5000} step={100} />
        </>
      }
      caption="Four chains start on a circle of radius 3. With ρ = 0.95 the ACF decays slowly and the bulk ESS is a small fraction of the draws; lower ρ and the chains mix within a few sweeps."
    />
  )
}

// ── HMC trajectories ────────────────────────────────────────────────────────────────────────────────────────────────

const HMC_TARGETS: Record<string, { target: Target; box: Box; start: [number, number] }> = {
  banana: { target: BANANA, box: BANANA_BOX, start: [-2, 4] },
  'correlated Gaussian': {
    target: gaussianTarget(
      [0, 0],
      [
        [1, 0.95],
        [0.95, 1],
      ],
    ),
    box: { x: [-3.5, 3.5], y: [-3.5, 3.5] },
    start: [-2.5, -1],
  },
  funnel: { target: funnel(), box: { x: [-7, 7], y: [-8, 8] }, start: [2, 3] },
}

export function HmcTrajectories() {
  const [which, setWhich] = useState<string>('banana')
  const [sampler, setSampler] = useState<'HMC' | 'NUTS'>('HMC')
  const [eps, setEps] = useState(0.25)
  const [L, setL] = useState(20)
  const [start, setStart] = useState<[number, number]>(HMC_TARGETS.banana.start)
  const [at, setAt] = useState(30)
  const spec = HMC_TARGETS[which]
  const surface = useMemo(() => densityGrid(spec.target, spec.box), [spec])
  const N = 30
  const tr = useMemo((): Trace<HmcState> | Trace<NutsState> => {
    const options = { stream: stream('hmc-trajectories') }
    return sampler === 'HMC'
      ? trace(hmc(spec.target, { stepSize: eps, steps: L }), { x0: start }, N, options)
      : trace(nuts(spec.target, { stepSize: eps }), { x0: start }, N, options)
  }, [spec, sampler, eps, L, start])
  const i = Math.min(at, tr.steps.length - 1)
  const state = tr.steps[i]
  const overlay = useMemo<HeatmapOverlay[]>(() => {
    const past = tr.steps.slice(1, i).flatMap((s) => {
      const c = columns(toFlat(s.trajectory), 2)
      return [...c.xs.map((x, k) => [x, c.ys[k]]), [NaN, NaN]]
    })
    const cur = columns(toFlat(state.trajectory), 2)
    const pts = columns(
      tr.steps.slice(0, i + 1).flatMap((s) => toFlat(s.x)),
      2,
    )
    return [
      {
        name: 'earlier trajectories',
        type: 'line',
        x: past.map((p) => p[0]),
        y: past.map((p) => p[1]),
        slot: 1,
        thin: true,
      },
      { name: 'current trajectory', type: 'line', x: cur.xs, y: cur.ys, slot: 2, showPoints: true },
      { name: 'samples', type: 'scatter', x: pts.xs, y: pts.ys, slot: 3 },
    ]
  }, [tr, i, state])
  const energy = useMemo<XYSeries[]>(() => {
    const H = toFlat(state.energies)
    const times = 'trajectoryTimes' in state ? toFlat(state.trajectoryTimes as Vector) : H.map((_, k) => k)
    return [
      { name: 'H along the trajectory', type: 'line', x: times, y: H, slot: 2, showPoints: true },
      {
        name: 'H at the start',
        type: 'line',
        x: [Math.min(...times), Math.max(...times)],
        y: [H[times.indexOf(0)], H[times.indexOf(0)]],
        muted: true,
        dashed: true,
      },
    ]
  }, [state])
  const last = tr.steps[tr.steps.length - 1]
  return (
    <Figure
      title="HMC trajectories and their energy"
      description="Leapfrog integration follows the Hamiltonian flow, so H stays nearly constant and distant proposals are accepted; too large a step makes H drift and the proposal is rejected or diverges."
      defaultSize="L"
      controls={
        <>
          <ControlRow label="1 · target and sampler">
            <Select
              label="target"
              value={which}
              onChange={(v) => {
                setWhich(v)
                setStart(HMC_TARGETS[v].start)
              }}
              options={Object.keys(HMC_TARGETS)}
            />
            <Select label="sampler" value={sampler} onChange={setSampler} options={['HMC', 'NUTS']} />
          </ControlRow>
          <ControlRow label="2 · integrator">
            <Slider label="step size ε" value={eps} onChange={setEps} min={0.01} max={1.5} step={0.01} />
            {sampler === 'HMC' && (
              <Slider label="leapfrog steps L" value={L} onChange={setL} min={1} max={80} step={1} />
            )}
          </ControlRow>
          <ControlRow label="3 · iteration">
            <Player value={at} onChange={setAt} count={N + 1} />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout
            label="energy error ΔH"
            value={
              'energyError' in state
                ? f3(state.energyError as number)
                : f3(Math.max(...toFlat(state.energies)) - Math.min(...toFlat(state.energies)))
            }
          />
          <Readout label="accepted" value={state.accepted ? 'yes' : 'no'} />
          <Readout label="acceptance rate" value={f3(last.acceptanceRate)} />
          <Readout label="divergent trajectories" value={last.divergentCount} />
          {'treeDepth' in state && <Readout label="tree depth" value={state.treeDepth as number} />}
        </>
      }
      caption="Top: every trajectory up to the current iteration (thin), the current one with its leapfrog points, and the samples. Drag the start. Bottom: the Hamiltonian H = −log π + ½|p|² at each leapfrog point of the current trajectory (for NUTS, by integration time, both directions). On the funnel the neck needs a small ε, so a step that suits the mouth diverges there."
    >
      <Subplots rows={2} heightRatios={[2, 1.2]}>
        <Panel>
          <Heatmap
            {...surface}
            xLabel="θ₀"
            yLabel="θ₁"
            valueLabel="π"
            colorBar={false}
            fillOpacity={0.7}
            overlay={overlay}
            handles={[
              {
                kind: 'point',
                at: start,
                onDrag: ([x, y]) =>
                  setStart([clamp(x, spec.box.x[0], spec.box.x[1]), clamp(y, spec.box.y[0], spec.box.y[1])]),
                label: 'start',
              },
            ]}
          />
        </Panel>
        <Panel>
          <XYChart series={energy} xLabel="leapfrog step" yLabel="H" />
        </Panel>
      </Subplots>
    </Figure>
  )
}

// ── MALA against ULA ───────────────────────────────────────────────────────────────────────────────────────────────

const STD_NORMAL = gaussianTarget([0], [[1]])
const H_GRID = grid(0.05, 1.5, 12)

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
  const [h, setH] = useState(0.6)
  const view = useMemo(() => {
    const ula = langevinDraws('ula', h, 10000)
    const ma = langevinDraws('mala', h, 10000)
    const edges = grid(-4.5, 4.5, 46)
    const bars = (x: number[]) => {
      const hs = histogram(x, { bins: edges })
      return { x: Array.from(hs.counts, (_, i) => (hs.edges[i] + hs.edges[i + 1]) / 2), y: Array.from(hs.density) }
    }
    const xs = grid(-4.5, 4.5, 181)
    const hist: XYSeries[] = [
      { name: 'ULA draws', type: 'bar', ...bars(ula), slot: 0, thin: true },
      { name: 'MALA draws', type: 'line', ...bars(ma), slot: 1, showPoints: true },
      {
        name: 'target N(0, 1)',
        type: 'line',
        x: xs,
        y: xs.map((x) => Math.exp(STD_NORMAL.logDensity(tensor([x])) as number)),
        emphasis: true,
      },
    ]
    return { hist, ula: variance(ula, { sample: true }), mala: variance(ma, { sample: true }) }
  }, [h])
  const curve = useMemo<XYSeries[]>(() => {
    const u = H_GRID.map((v) => variance(langevinDraws('ula', v, 4000), { sample: true }))
    const m = H_GRID.map((v) => variance(langevinDraws('mala', v, 4000), { sample: true }))
    return [
      { name: 'ULA variance', type: 'line', x: H_GRID, y: u, slot: 0, showPoints: true },
      { name: 'MALA variance', type: 'line', x: H_GRID, y: m, slot: 1, showPoints: true },
      {
        name: 'target variance 1',
        type: 'line',
        x: [H_GRID[0], H_GRID[H_GRID.length - 1]],
        y: [1, 1],
        muted: true,
        dashed: true,
      },
    ]
  }, [])
  return (
    <Figure
      title="MALA corrects the bias of unadjusted Langevin"
      description="ULA keeps every Euler step and so samples a law wider than the target by O(h); MALA's Metropolis test removes the bias."
      controls={<Slider label="step size h" value={h} onChange={setH} min={0.05} max={1.5} step={0.05} />}
      readouts={
        <>
          <Readout label="ULA variance" value={f3(view.ula)} />
          <Readout label="MALA variance" value={f3(view.mala)} />
        </>
      }
      caption="Target N(0, 1); two chains of 10 000 steps each. Left: the histograms of ULA and MALA draws against the target. Right: the sample variance of each sampler over step sizes (4000 steps per point); drag the vertical line to set h. ULA's variance grows like 1/(1 − h/2); MALA stays at 1 while its acceptance rate falls."
    >
      <Subplots cols={2}>
        <Panel>
          <XYChart series={view.hist} xLabel="x" yLabel="density" rescaleOnChange={false} holdFit="union" />
        </Panel>
        <Panel>
          <XYChart
            series={curve}
            xLabel="step size h"
            yLabel="variance"
            handles={[{ kind: 'x', at: h, onDrag: (v) => setH(clamp(v, 0.05, 1.5)), label: 'h' }]}
          />
        </Panel>
      </Subplots>
    </Figure>
  )
}

// ── Diagnostics ─────────────────────────────────────────────────────────────────────────────────────────────────────

export function Diagnostics() {
  const [chains, setChains] = useState(4)
  const [steps, setSteps] = useState(400)
  const [scale, setScale] = useState(0.3)
  const result = useMemo(() => {
    const starts = stream('dispersed')
    return sampleChains(
      randomWalkMetropolis(BANANA, { scale }),
      (k: number) => ({ x0: [normal(starts.child(k, 0), 0, 2), normal(starts.child(k, 1), 3, 3)] }),
      { chains, steps, stream: stream('diagnostics') },
    )
  }, [chains, steps, scale])
  return (
    <ChainView
      title="ESS and R̂ for random-walk Metropolis on the banana"
      description="Chains from dispersed starts: R̂ stays well above 1 until they have forgotten their starts and agree."
      draws={result.draws}
      steps={result.steps}
      names={['x', 'y']}
      means={[0, 0]}
      controls={
        <>
          <Slider label="chains" value={chains} onChange={setChains} min={1} max={8} step={1} />
          <Slider label="steps per chain" value={steps} onChange={setSteps} min={50} max={5000} step={50} />
          <Slider label="proposal scale σ" value={scale} onChange={setScale} min={0.05} max={3} step={0.05} />
        </>
      }
      caption="With 400 steps and σ = 0.3 the chains are still apart: R̂ is well above 1.01 and the ESS is tiny. Raise the steps (or tune σ near 1) and R̂ falls towards 1 while the ESS grows."
    />
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

export function ParticleFilter() {
  const [N, setN] = useState(500)
  const [scheme, setScheme] = useState<ResamplingScheme>('systematic')
  const [threshold, setThreshold] = useState(0.5)
  const [at, setAt] = useState(20)
  const world = useMemo(() => {
    const s = stream('growth-model')
    const xs: number[] = []
    const ys: number[] = []
    let x = GROWTH.sampleInitial(s.child('x0'))
    for (let t = 0; t < T; t++) {
      if (t > 0) x = GROWTH.sampleTransition(tensor([x]), t, s.child('x', t))
      xs.push(x)
      ys.push(normal(s.child('y', t), (x * x) / 20, 1))
    }
    return { xs, ys }
  }, [])
  const tr = useMemo(
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
  )
  const i = clamp(at, 1, T)
  const state = tr.steps[i]
  const times = Array.from({ length: T }, (_, t) => t)
  const series = useMemo<XYSeries[]>(() => {
    const mean = toFlat(tr.series.mean).slice(1)
    const sd = toFlat(tr.series.variance).slice(1).map(Math.sqrt)
    const p = toFlat(state.particles)
    const w = toFlat(state.weights)
    const keep = p.map((v, k) => [v, w[k]]).filter((_, k) => k % Math.max(1, Math.floor(N / 300)) === 0)
    return [
      { name: 'true state', type: 'line', x: times, y: world.xs, emphasis: true },
      { name: 'filter mean', type: 'line', x: times, y: mean, slot: 0 },
      { name: 'mean + 2 sd', type: 'line', x: times, y: mean.map((m, k) => m + 2 * sd[k]), slot: 0, dashed: true },
      { name: 'mean − 2 sd', type: 'line', x: times, y: mean.map((m, k) => m - 2 * sd[k]), slot: 0, dashed: true },
      {
        name: `particles at t = ${i - 1}`,
        type: 'scatter',
        x: keep.map(() => i - 1),
        y: keep.map((q) => q[0]),
        slot: 2,
      },
    ]
  }, [tr, state, world, i, N, times])
  const essSeries = useMemo<XYSeries[]>(() => {
    const ess = toFlat(tr.series.ess).slice(1)
    const res = toFlat(tr.series.resampled).slice(1)
    return [
      { name: 'ESS', type: 'line', x: times, y: ess, slot: 0 },
      {
        name: 'resampled',
        type: 'scatter',
        x: times.filter((_, k) => res[k]),
        y: ess.filter((_, k) => res[k]),
        slot: 1,
      },
      { name: 'threshold', type: 'line', x: [0, T - 1], y: [threshold * N, threshold * N], muted: true, dashed: true },
    ]
  }, [tr, threshold, N, times])
  const last = tr.steps[tr.steps.length - 1]
  const resamples = toFlat(tr.series.resampled).reduce((a, b) => a + b, 0)
  return (
    <Figure
      title="A bootstrap particle filter on the nonlinear growth model"
      description="Particles move through the transition, are weighted by the observation and resampled when the ESS falls; y = x²/20 hides the sign of x, so the filtering distribution is often bimodal."
      defaultSize="L"
      controls={
        <>
          <ControlRow label="1 · filter">
            <Slider label="particles N" value={N} onChange={setN} min={20} max={3000} step={10} />
            <Select label="resampling" value={scheme} onChange={setScheme} options={resamplingSchemes} />
            <Slider
              label="resample when ESS / N <"
              value={threshold}
              onChange={setThreshold}
              min={0}
              max={1}
              step={0.05}
            />
          </ControlRow>
          <ControlRow label="2 · time">
            <Player value={at} onChange={setAt} count={T + 1} />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout label="ESS at t" value={f3(state.ess)} />
          <Readout label="resampling steps" value={resamples} />
          <Readout label="log p(y₀:T) estimate" value={f3(last.logEvidence)} />
        </>
      }
      caption="xₜ = xₜ₋₁/2 + 25xₜ₋₁/(1 + xₜ₋₁²) + 8 cos(1.2t) + N(0, 10), yₜ = xₜ²/20 + N(0, 1). Top: the true state, the filter mean ± 2 sd and the particle cloud at the player's time (up to 300 particles). Bottom: the ESS after weighting, with points where the particles were resampled. With few particles or threshold 0 (never resample) the weights collapse and the filter loses the state."
    >
      <Subplots rows={2} sharex heightRatios={[2, 1.2]}>
        <Panel>
          <XYChart series={series} xLabel="t" yLabel="x" legend />
        </Panel>
        <Panel>
          <XYChart series={essSeries} xLabel="t" yLabel="ESS" />
        </Panel>
      </Subplots>
    </Figure>
  )
}
