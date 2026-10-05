import {
  ddimSampler,
  ddpmSampler,
  forwardNoise,
  gaussianMixtureData,
  linearSchedule,
  mixtureLogDensity,
  mixtureNoisePredictor,
  mixtureScore,
  probabilityFlowSampler,
  sampleMixture,
  vpSde,
  type SamplerState,
} from 'aifn-methods/generative/diffusion'
import { normals, stream } from 'aifn-compute/foundation/random'
import { fromData, linspace, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { trace, type Trace } from 'aifn-compute/foundation/trace'
import { useMemo, useState } from 'react'
import { Player } from 'aifn-render/controls'
import { ControlRow, Figure } from 'aifn-render/layout'
import { row, slider, toggle, useFigureState } from 'aifn-render/state'
import {
  Curve,
  Plot,
  Plots,
  Points,
  Raster,
  Readout,
  useAxis,
  useElementSize,
  Vectors,
  type Vector,
} from 'aifn-render/viz'
import { formatValue } from '@lab/views'

const f3 = (v: number) => formatValue(Number(v.toPrecision(3)))
const grid = (lo: number, hi: number, n: number) => toFlat(linspace(lo, hi, n))

// Six components on a ring with unequal weights and one elongated blob in the middle.
const RING = Array.from({ length: 6 }, (_, k) => [2.4 * Math.cos((k * Math.PI) / 3), 2.4 * Math.sin((k * Math.PI) / 3)])
const MIX = gaussianMixtureData(
  [0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.1],
  [...RING, [0, 0]],
  [0.22, 0.22, 0.22, 0.22, 0.22, 0.22, [0.7, 0.18]],
)
const BOX: [number, number] = [-4, 4]
const GX = grid(BOX[0], BOX[1], 64)
const GRID = fromData(Float64Array.from(GX.flatMap((y) => GX.flatMap((x) => [x, y]))), [GX.length * GX.length, 2])
const AX = grid(-3.5, 3.5, 11)
const ARROWS_AT = fromData(Float64Array.from(AX.flatMap((y) => AX.flatMap((x) => [x, y]))), [AX.length * AX.length, 2])
const FLOOR = -8
const NO_ARROWS: Vector[] = []

// The continuous VP SDE of Song et al. (β from 0.1 to 20) and the discrete linear schedule it is the limit of, with
// T = 200 steps and β scaled so that T·βₜ runs over the same 0.1 … 20: discrete step k sits at time t = k/T.
const SDE = vpSde()
const T = 200
const SCHEDULE = linearSchedule(T, { betaStart: 0.1 / T, betaEnd: 20 / T })

/** log p_t on the grid, floored, as heatmap rows. */
function densityAt(t: number) {
  const v = toFlat(mixtureLogDensity(MIX, GRID, SDE.meanScale(t), SDE.std(t)))
  return GX.map((_, i) => v.slice(i * GX.length, (i + 1) * GX.length).map((z) => Math.max(FLOOR, z)))
}

/** Arrow grid spacing: the longest arrow at each t is drawn this long. */
const ARROW_SPACING = AX[1] - AX[0]

/**
 * Score arrows at time t: ∇log p_t rescaled so that the longest arrow is 0.8 of the grid spacing. The score's size
 * changes by orders of magnitude over t (it grows as 1/s(t) near the data), so a fixed scale shows only arrowheads at
 * one end of the player or overlapping shafts at the other; the readout gives the largest |s(t)·∇log p_t|.
 */
function scoreArrows(t: number): { vectors: Vector[]; largest: number } {
  const s = Math.max(SDE.std(t), 1e-3)
  const g = toFlat(mixtureScore(MIX, ARROWS_AT, SDE.meanScale(t), s))
  const p = toFlat(ARROWS_AT)
  let largest = 0
  for (let i = 0; i < g.length / 2; i++) largest = Math.max(largest, Math.hypot(g[2 * i], g[2 * i + 1]))
  const k = largest > 0 ? (0.8 * ARROW_SPACING) / largest : 0
  return {
    largest: s * largest,
    vectors: Array.from({ length: p.length / 2 }, (_, i) => ({
      from: [p[2 * i], p[2 * i + 1]] as [number, number],
      to: [p[2 * i] + k * g[2 * i], p[2 * i + 1] + k * g[2 * i + 1]] as [number, number],
    })),
  }
}

const cols = (x: Tensor) => {
  const f = toFlat(x)
  return { x: f.filter((_, i) => i % 2 === 0), y: f.filter((_, i) => i % 2 === 1) }
}

// ── Forward noising ────────────────────────────────────────────────────────────────────────────────────────────────

const X0 = sampleMixture(stream('showcase-diffusion-data'), MIX, 400)
const FORWARD_FRAMES = 60
// Frames are spaced as t = (k/K)², so the player spends its frames where the modes are still apart.
const timeOf = (k: number, frames: number) => (k / frames) ** 2

export function DiffusionForward() {
  const state = useFigureState({ reveal: row('1 · reveal', { arrows: toggle(true, 'score field ∇log p_t') }) })
  const { arrows } = state.reveal
  const [k, setK] = useState(0)
  const t = timeOf(k, FORWARD_FRAMES)
  const z = useMemo(() => densityAt(t), [t])
  // The same ε at every t, so each particle moves smoothly along x_t = m(t)·x₀ + s(t)·ε.
  const xt = useMemo(() => {
    const m = SDE.meanScale(t)
    return cols(forwardNoise(stream('showcase-diffusion-eps'), X0, m * m).x)
  }, [t])
  const x1 = useAxis({ label: 'x₁', range: BOX, nice: false })
  const x2 = useAxis({ label: 'x₂', range: BOX, nice: false, equal: x1 })
  const field = useMemo(() => scoreArrows(t), [t])
  const vectors = arrows ? field.vectors : NO_ARROWS
  return (
    <Figure
      title="Forward noising of a mixture, with its exact score"
      purpose="The forward process shrinks the data towards 0 and adds Gaussian noise, so p_t blurs from seven sharp components into one standard normal; the score ∇log p_t points back towards where the data was."
      state={state}
      defaultSize="L"
      controls={
        <ControlRow label="2 · time">
          <Player
            value={k}
            onChange={setK}
            count={FORWARD_FRAMES + 1}
            format={(p) => `t = ${f3(timeOf(p, FORWARD_FRAMES))}`}
          />
        </ControlRow>
      }
      readouts={
        <>
          <Readout label="t" value={f3(t)} />
          <Readout label="mean scale m(t)" value={f3(SDE.meanScale(t))} />
          <Readout label="noise sd s(t)" value={f3(SDE.std(t))} />
          <Readout label="SNR m²/s²" value={t === 0 ? '∞' : f3(SDE.meanScale(t) ** 2 / SDE.std(t) ** 2)} />
          <Readout label="largest |s(t)·∇log p_t| on the grid" value={t === 0 ? '—' : f3(field.largest)} />
        </>
      }
      caption="The data are a seven-component Gaussian mixture; x_t = m(t)·x₀ + s(t)·ε under the variance-preserving SDE (β from 0.1 to 20), with one ε per particle held fixed over t. The background is log p_t (floored at −8), exact because a noised mixture is a mixture. Arrows are the score ∇log p_t, rescaled at each t so that the longest is 0.8 of the grid spacing: the score grows as 1/s(t) near t = 0, so a fixed scale cannot show it over the whole player. s(t)·∇log p_t = −ε̂ is the exact noise prediction, negated; the readout gives its largest size on the grid. The player's frames are spaced as t = (k/K)²: the ring's modes have merged by t ≈ 0.2, and p_t is close to N(0, I) by t ≈ 0.6."
    >
      <Plot x={x1} y={x2}>
        <Raster x={GX} y={GX} z={z} range={[FLOOR, 1.5]} fillOpacity={0.75} valueLabel="log p_t" />
        <Points name="particles x_t" x={xt.x} y={xt.y} slot={1} thin />
        <Vectors vectors={vectors} />
      </Plot>
    </Figure>
  )
}

// ── Reverse sampling: DDPM, DDIM and the probability-flow ODE ────────────────────────────────────────────────────────

const PARTICLES = 120
/** Paths are drawn for the first few particles only, so that DDPM's noisy paths stay readable. */
const PATHS = 25
const XT = normals(stream('showcase-diffusion-prior'), [PARTICLES, 2])
const PREDICTOR = mixtureNoisePredictor(MIX)
const REVERSE_FRAMES = 80
const SAMPLERS = ['DDPM', 'DDIM', 'probability flow'] as const

type Run = { tr: Trace<SamplerState>; times: number[] }

/** The last state at or after continuous time t (sampling runs from t = 1 down to 0). */
function stateAt(run: Run, t: number) {
  let i = 0
  while (i + 1 < run.times.length && run.times[i + 1] >= t - 1e-9) i++
  return i
}

export function DiffusionReverse() {
  const state = useFigureState({
    samplers: row('1 · samplers', {
      ddimSteps: slider(2, 100, 20, { label: 'DDIM steps', step: 1 }),
      odeSteps: slider(2, 100, 30, { label: 'ODE steps (RK4)', step: 1 }),
    }),
    reveal: row('2 · reveal', {
      paths: toggle(true, 'paths'),
      arrows: toggle(false, 'score field at t'),
    }),
  })
  const { ddimSteps, odeSteps } = state.samplers
  const { paths, arrows } = state.reveal
  const [k, setK] = useState(0)
  const t = timeOf(REVERSE_FRAMES - k, REVERSE_FRAMES)
  const [frame, size] = useElementSize<HTMLDivElement>()
  const stacked = size.width > 0 && size.width < 600
  // One axis pair per panel: equal-unit panels that share an axis come out at different sizes (only the first shows
  // its ticks), so each panel keeps its own (component request in phase5c-3.md).
  const ax = [
    useAxis({ label: 'x₁', range: BOX, nice: false }),
    useAxis({ label: 'x₁', range: BOX, nice: false }),
    useAxis({ label: 'x₁', range: BOX, nice: false }),
  ]
  const ay = [
    useAxis({ label: 'x₂', range: BOX, nice: false, equal: ax[0] }),
    useAxis({ label: 'x₂', range: BOX, nice: false, equal: ax[1] }),
    useAxis({ label: 'x₂', range: BOX, nice: false, equal: ax[2] }),
  ]

  const runs = useMemo(() => {
    const discrete = (tr: Trace<SamplerState>): Run => ({ tr, times: tr.steps.map((s) => s.time / T) })
    const opts = { stream: stream('showcase-diffusion-reverse') }
    return {
      DDPM: discrete(trace(ddpmSampler(PREDICTOR, SCHEDULE), { x: XT }, T, opts)),
      DDIM: discrete(trace(ddimSampler(PREDICTOR, SCHEDULE, { steps: ddimSteps }), { x: XT }, T, opts)),
      'probability flow': (() => {
        const tr = trace(probabilityFlowSampler(PREDICTOR, SDE, { steps: odeSteps }), { x: XT }, odeSteps, opts)
        return { tr, times: tr.steps.map((s) => s.time) }
      })(),
    } satisfies Record<(typeof SAMPLERS)[number], Run>
  }, [ddimSteps, odeSteps])

  const z = useMemo(() => densityAt(Math.max(t, 1e-3)), [t])
  const vectors = useMemo(() => (arrows ? scoreArrows(Math.max(t, 1e-3)).vectors : NO_ARROWS), [arrows, t])

  const panels = SAMPLERS.map((name) => {
    const run = runs[name]
    const i = stateAt(run, t)
    const now = cols(run.tr.steps[i].x)
    let path: { x: number[]; y: number[] } | null = null
    if (paths) {
      // The first particles' paths from x_T to its state now, separated by NaN breaks.
      const px: number[] = []
      const py: number[] = []
      const flats = run.tr.steps.slice(0, i + 1).map((s) => toFlat(s.x))
      for (let p = 0; p < PATHS; p++) {
        for (const f of flats) {
          px.push(f[2 * p])
          py.push(f[2 * p + 1])
        }
        px.push(NaN)
        py.push(NaN)
      }
      path = { x: px, y: py }
    }
    return { name, now, path, evaluations: run.tr.steps[i].evaluations, steps: i, final: run.tr.final.x }
  })

  // Sample quality: the average data log-density of each sampler's final particles.
  const quality = useMemo(
    () =>
      SAMPLERS.map((name) => {
        const v = toFlat(mixtureLogDensity(MIX, runs[name].tr.final.x))
        return v.reduce((a, b) => a + b, 0) / v.length
      }),
    [runs],
  )
  const reference = useMemo(() => {
    const v = toFlat(mixtureLogDensity(MIX, X0))
    return v.reduce((a, b) => a + b, 0) / v.length
  }, [])

  return (
    <Figure
      title="Reverse sampling: DDPM against DDIM against the probability-flow ODE"
      purpose="All three samplers start from the same noise and follow the same exact score back to the data; DDPM adds fresh noise every step and wanders, while DDIM and the probability-flow ODE are deterministic, travel smooth paths and need far fewer steps."
      state={state}
      defaultSize="full"
      controls={
        <ControlRow label="3 · time">
          <Player
            value={k}
            onChange={setK}
            count={REVERSE_FRAMES + 1}
            format={(p) => `t = ${f3(timeOf(REVERSE_FRAMES - p, REVERSE_FRAMES))}`}
          />
        </ControlRow>
      }
      readouts={
        <>
          {panels.map((p, j) => (
            <Readout
              key={p.name}
              label={p.name}
              value={`${p.evaluations} calls so far · final mean log p₀ ${f3(quality[j])}`}
            />
          ))}
          <Readout label="true samples" value={`mean log p₀ ${f3(reference)}`} />
        </>
      }
      caption={`${PARTICLES} particles drawn once from N(0, I) at t = 1 are carried to t ≈ 0 by each sampler, all driven by the mixture’s exact noise predictor (so no network is trained). DDPM: ancestral sampling over all ${T} steps of the discrete schedule; DDIM (η = 0) on an evenly spaced subsequence; the probability-flow ODE of the VP SDE by RK4 (four predictor calls a step). The discrete step k is drawn at time k/${T}, where its ᾱ matches the SDE's. Paths are drawn for 25 of the particles. The player's frames are spaced as t = (k/K)², denser near the data. Background: log p_t at the played time; arrows (revealed): ∇log p_t, rescaled as above. The readouts give the predictor calls so far and the average data log-density of the final particles, against that of true samples: DDIM and the ODE reach it with a small fraction of DDPM's calls. Lower the DDIM steps to 2–3 and its particles land between the modes.`}
    >
      {/* Three square panels side by side need about 600 px; narrower, they stack. */}
      <div ref={frame} className="h-full w-full">
        <Plots cols={stacked ? 1 : 3} rows={stacked ? 3 : 1} ratiosOf={stacked ? 'equal' : undefined}>
          {panels.map((p, j) => (
            <Plot key={p.name} x={ax[j]} y={ay[j]} title={p.name} legend={false}>
              <Raster
                x={GX}
                y={GX}
                z={z}
                range={[FLOOR, 1.5]}
                colorBar={false}
                fillOpacity={0.6}
                valueLabel="log p_t"
              />
              {p.path && <Curve name="paths" x={p.path.x} y={p.path.y} slot={2} thin />}
              <Points name="particles" x={p.now.x} y={p.now.y} slot={1} thin />
              <Vectors vectors={vectors} />
            </Plot>
          ))}
        </Plots>
      </div>
    </Figure>
  )
}
