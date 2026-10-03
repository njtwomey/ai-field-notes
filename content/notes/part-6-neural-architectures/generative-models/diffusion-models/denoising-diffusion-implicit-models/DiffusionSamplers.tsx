import {
  ddimSampler,
  ddpmSampler,
  gaussianMixtureData,
  linearSchedule,
  mixtureLogDensity,
  mixtureNoisePredictor,
  mixtureScore,
  probabilityFlowSampler,
  vpSde,
  type SamplerState,
} from 'aifn-applied/generative/diffusion'
import { normals, stream } from 'aifn/foundation/random'
import { fromData, linspace, toFlat, type Tensor } from 'aifn/foundation/tensor'
import { trace, type Trace } from 'aifn/foundation/trace'
import { useMemo, useState } from 'react'
import {
  ControlRow,
  Curve,
  Figure,
  Player,
  Plot,
  Plots,
  Points,
  Raster,
  Readout,
  Vectors,
  formatNumber,
  row,
  slider,
  toggle,
  useAxis,
  useElementSize,
  useFigureState,
  type Vector,
} from 'aifn-render'

const f3 = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')
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

const SDE = vpSde()
const T = 200
const SCHEDULE = linearSchedule(T, { betaStart: 0.1 / T, betaEnd: 20 / T })

function densityAt(t: number) {
  const v = toFlat(mixtureLogDensity(MIX, GRID, SDE.meanScale(t), SDE.std(t)))
  return GX.map((_, i) => v.slice(i * GX.length, (i + 1) * GX.length).map((z) => Math.max(FLOOR, z)))
}

const ARROW_SPACING = AX[1] - AX[0]

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

const timeOf = (k: number, frames: number) => (k / frames) ** 2
const PARTICLES = 120
const PATHS = 25
const XT = normals(stream('showcase-diffusion-prior'), [PARTICLES, 2])
const PREDICTOR = mixtureNoisePredictor(MIX)
const REVERSE_FRAMES = 80
const SAMPLERS = ['DDPM', 'DDIM', 'probability flow'] as const

type Run = { tr: Trace<SamplerState>; times: number[] }

function stateAt(run: Run, t: number) {
  let i = 0
  while (i + 1 < run.times.length && run.times[i + 1] >= t - 1e-9) i++
  return i
}

export function DiffusionSamplers() {
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

  const quality = useMemo(
    () =>
      SAMPLERS.map((name) => {
        const v = toFlat(mixtureLogDensity(MIX, runs[name].tr.final.x))
        return v.reduce((a, b) => a + b, 0) / v.length
      }),
    [runs],
  )

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
        </>
      }
      caption={`${PARTICLES} particles drawn once from N(0, I) at t = 1 are carried to t ≈ 0 by each sampler, all driven by the mixture’s exact noise predictor (so no network is trained). DDPM: ancestral sampling over all ${T} steps of the discrete schedule; DDIM (η = 0) on an evenly spaced subsequence; the probability-flow ODE of the VP SDE by RK4 (four predictor calls a step). The discrete step k is drawn at time k/${T}, where its ᾱ matches the SDE's. Paths are drawn for 25 of the particles. The player's frames are spaced as t = (k/K)², denser near the data. Background: log p_t at the played time; arrows (revealed): ∇log p_t, rescaled as above. The readouts give the predictor calls so far and the average data log-density of the final particles. Lower the DDIM steps to 2–3 and its particles land between the modes.`}
    >
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
