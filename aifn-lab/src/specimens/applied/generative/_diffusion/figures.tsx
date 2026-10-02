import {
  alphaBarAt,
  cosineSchedule,
  ddimSampler,
  ddpmSampler,
  denoiser,
  denoiserTraining,
  forwardNoise,
  gaussianMixtureData,
  linearSchedule,
  mixtureNoisePredictor,
  networkNoisePredictor,
  probabilityFlowSampler,
  sampleMixture,
  vpSde,
  type SamplerState,
} from 'aifn-applied/generative/diffusion'
import { stream } from 'aifn/foundation/random'
import { toFlat, toRows, type Tensor } from 'aifn/foundation/tensor'
import { trace, type Algorithm, type Trace } from 'aifn/foundation/trace'
import { useMemo, useState } from 'react'
import { adamRule } from 'aifn/optim/first-order'
import { ControlRow, Figure } from '@lab/layout'
import { Player } from '@lab/controls'
import { choice, row, slider, useFigureState, int } from '@lab/state'
import { Curve, Handle, Plot, Plots, Points, Readout, formatNumber, useAxis } from '@lab/viz'

const fmt = (v: number) => formatNumber(v)

/** A 2-D mixture of three components: two elongated blobs and a small round one. */
const MIX = gaussianMixtureData(
  [0.4, 0.35, 0.25],
  [
    [-1.5, -0.5],
    [1.2, 1.2],
    [1.5, -1.3],
  ],
  [
    [
      [0.35, 0.2],
      [0.2, 0.2],
    ],
    [0.25, 0.45],
    0.2,
  ],
)
const DATA = sampleMixture(stream('diffusion-data'), MIX, 600)
const columns = (x: Tensor) => {
  const rows = toRows(x)
  return { x: rows.map((r) => r[0]), y: rows.map((r) => r[1]) }
}
const RANGE: [number, number] = [-3.5, 3.5]

// ---------------------------------------------------------------------------------------------------------------------
// 1. Forward noising.

const SCHEDULES = { linear: linearSchedule(), cosine: cosineSchedule() }
type ScheduleId = keyof typeof SCHEDULES
const TICKS = Array.from({ length: 101 }, (_, k) => k * 10)

const SCHEDULE_OPTIONS = [
  { value: 'linear' as const, label: 'linear (Ho et al.)' },
  { value: 'cosine' as const, label: 'cosine (Nichol & Dhariwal)' },
]
/** √ᾱ and √(1 − ᾱ) of both schedules over the steps, for the right-hand panel. */
const SCALES = (Object.keys(SCHEDULES) as ScheduleId[]).map((name) => ({
  name,
  signal: TICKS.map((s) => Math.sqrt(alphaBarAt(SCHEDULES[name], s))),
  noise: TICKS.map((s) => Math.sqrt(1 - alphaBarAt(SCHEDULES[name], s))),
}))

export function ForwardNoisingSpecimen() {
  const state = useFigureState({
    setup: row('1 · schedule', { id: choice(SCHEDULE_OPTIONS, 'linear', { label: 'schedule' }) }),
  })
  const id = state.setup.id as ScheduleId
  const [k, setK] = useState(0)
  const schedule = SCHEDULES[id]
  const t = TICKS[k]
  const ab = alphaBarAt(schedule, t)
  // The same noise draw at every t, so each point moves smoothly from its data position towards N(0, I).
  const xt = useMemo(() => columns(forwardNoise(stream('diffusion-eps'), DATA, ab).x), [ab])
  const x1 = useAxis({ label: 'x₁', range: RANGE })
  const x2 = useAxis({ label: 'x₂', range: RANGE, equal: x1 })
  const steps = useAxis({ label: 'step t', range: [0, 1000] })
  const scale = useAxis({ label: 'scale', range: [0, 1] })
  return (
    <Figure
      title="Forward noising of a 2-D mixture"
      purpose="The forward process shrinks the data by √ᾱ_t and adds noise of sd √(1 − ᾱ_t), so the mixture's modes blur into one standard normal; the cosine schedule keeps more signal early."
      state={state}
      defaultSize="L"
      controls={
        <ControlRow label="2 · step">
          <Player value={k} onChange={setK} count={TICKS.length} format={(i) => String(TICKS[i])} label="t" />
        </ControlRow>
      }
      readouts={
        <>
          <Readout label="t" value={String(t)} />
          <Readout label="ᾱ_t" value={fmt(ab)} />
          <Readout label="SNR" value={fmt(ab / (1 - ab))} />
        </>
      }
      caption="aifn forwardNoise of 600 mixture samples with one fixed noise draw, x_t = √ᾱ_t x₀ + √(1 − ᾱ_t) ε, T = 1000. Right: the signal (solid) and noise (dashed) scales of both schedules; drag the step line or play."
    >
      <Plots cols={2} widths={[1, 1.2]}>
        <Plot x={x1} y={x2}>
          <Points name="x_t" x={xt.x} y={xt.y} slot={0} />
        </Plot>
        <Plot x={steps} y={scale}>
          {SCALES.flatMap((c, slot) => [
            <Curve key={`${c.name}-s`} name={`√ᾱ (${c.name})`} x={TICKS} y={c.signal} slot={slot} />,
            <Curve key={`${c.name}-n`} name={`√(1 − ᾱ) (${c.name})`} x={TICKS} y={c.noise} slot={slot} dashed />,
          ])}
          <Handle kind="x" at={t} label="t" onDrag={(x) => setK(Math.max(0, Math.min(100, Math.round(x / 10))))} />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 2. DDPM against DDIM against the probability-flow ODE, with the exact score.

const PREDICTOR = mixtureNoisePredictor(MIX)
const FRAMES = 40
const PATHS = 12

type Run = { label: string; trace: Trace<SamplerState>; steps: number }

function sampled(
  alg: Algorithm<{ n: number; dimension: number }, SamplerState>,
  steps: number,
  label: string,
  n: number,
): Run {
  const every = Math.max(1, Math.ceil(steps / FRAMES))
  return { label, steps, trace: trace(alg, { n, dimension: 2 }, steps, { every, stream: stream('diffusion-start') }) }
}

/** The kept state nearest a fraction f ∈ [0, 1] of a run. */
const at = (run: Run, f: number) => run.trace.steps[Math.round(f * (run.trace.steps.length - 1))]

export function ReverseSamplingSpecimen() {
  const state = useFigureState({
    budgets: row('1 · step budgets', {
      ddimSteps: slider(5, 200, 50, { label: 'DDIM steps', step: 5 }),
      odeSteps: slider(5, 100, 40, { label: 'ODE steps (RK4)', step: 5 }),
    }),
  })
  const { ddimSteps, odeSteps } = state.budgets
  const [frame, setFrame] = useState(0)
  const n = 500
  const schedule = SCHEDULES.linear
  const ddpm = useMemo(() => sampled(ddpmSampler(PREDICTOR, schedule), 1000, 'DDPM (1000 steps)', n), [schedule])
  const ddim = useMemo(
    () =>
      sampled(ddimSampler(PREDICTOR, schedule, { steps: ddimSteps }), ddimSteps, `DDIM, η = 0 (${ddimSteps} steps)`, n),
    [schedule, ddimSteps],
  )
  const ode = useMemo(
    () =>
      sampled(
        probabilityFlowSampler(PREDICTOR, vpSde(), { steps: odeSteps }),
        odeSteps,
        `probability flow, RK4 (${odeSteps} steps)`,
        n,
      ),
    [odeSteps],
  )
  const f = frame / FRAMES
  const data = useMemo(() => columns(DATA), [])
  const runs = [ddpm, ddim, ode]
  const shown = runs.map((run) => {
    const upto = run.trace.steps.slice(0, Math.round(f * (run.trace.steps.length - 1)) + 1)
    return {
      points: columns(at(run, f).x),
      paths: Array.from({ length: PATHS }, (_, i) => ({
        x: upto.map((s) => s.x.data[2 * i] as number),
        y: upto.map((s) => s.x.data[2 * i + 1] as number),
      })),
    }
  })
  // One axis pair per panel: equal-unit panels sharing an axis come out at different sizes (phase5c-3.md request).
  const ax = [
    useAxis({ label: 'x₁', range: RANGE }),
    useAxis({ label: 'x₁', range: RANGE }),
    useAxis({ label: 'x₁', range: RANGE }),
  ]
  const ay = [
    useAxis({ label: 'x₂', range: RANGE, equal: ax[0] }),
    useAxis({ label: 'x₂', range: RANGE, equal: ax[1] }),
    useAxis({ label: 'x₂', range: RANGE, equal: ax[2] }),
  ]
  return (
    <Figure
      title="DDPM against DDIM against the probability-flow ODE"
      purpose="Three samplers turn the same N(0, I) draws into mixture samples with its exact score: DDPM's paths are rough and need a thousand steps, DDIM and the probability-flow ODE follow smooth paths in tens."
      state={state}
      defaultSize="XL"
      controls={
        <ControlRow label="2 · progress">
          <Player
            value={frame}
            onChange={setFrame}
            count={FRAMES + 1}
            format={(i) => `${Math.round((100 * i) / FRAMES)}%`}
            label="progress"
          />
        </ControlRow>
      }
      readouts={runs.map((r) => (
        <Readout
          key={r.label}
          label={`${r.label}: predictor calls`}
          value={String(r.trace.steps.at(-1)!.evaluations)}
        />
      ))}
      caption={`Left to right: ${runs.map((r) => r.label).join('; ')}. aifn samplers driven by mixtureNoisePredictor (no training), 500 particles from the same start, with ${PATHS} paths drawn; grey: data. Play from 0% (the noise) to 100% (the samples). The probability-flow ODE runs on the continuous VP SDE (β from 0.1 to 20), the others on the linear schedule with T = 1000.`}
    >
      <Plots cols={3}>
        {runs.map((r, slot) => (
          <Plot key={r.label} x={ax[slot]} y={ay[slot]} title={r.label} legend={false}>
            <Points name="data" x={data.x} y={data.y} muted thin />
            <Points name={r.label} x={shown[slot].points.x} y={shown[slot].points.y} slot={slot} thin />
            {shown[slot].paths.map((p, i) => (
              <Curve key={i} name="paths" x={p.x} y={p.y} thin slot={slot} />
            ))}
          </Plot>
        ))}
      </Plots>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 3. A learned denoiser.

const SMALL = linearSchedule(100, { betaEnd: 0.2 })

export function LearnedDenoiserSpecimen() {
  const state = useFigureState({
    setup: row('1 · training and sampling', {
      steps: int(800, { ge: 1, suggestions: [200, 800, 2000], label: 'training steps' }),
      ddimSteps: slider(5, 200, 40, { label: 'DDIM steps', step: 5 }),
    }),
  })
  const steps = state.setup.steps
  const { ddimSteps } = state.setup
  const net = useMemo(() => denoiser(2, { hidden: [32, 32], frequencies: 4 }), [])
  const training = useMemo(() => {
    const n = Number(steps)
    // Adam with a learning rate falling linearly to a tenth over the run.
    const optimizer = adamRule({ stepSize: (t: number) => 0.01 * Math.max(0.1, 1 - t / n) })
    const alg = denoiserTraining({ data: DATA, schedule: SMALL, net, batchSize: 128, optimizer })
    return trace(alg, { params: net.layer.init(stream('denoiser')) }, n, { every: 10, record: { loss: (s) => s.loss } })
  }, [net, steps])
  const params = training.steps.at(-1)!.params
  const samples = useMemo(
    () =>
      trace(
        ddimSampler(networkNoisePredictor(net, params), SMALL, { steps: ddimSteps }),
        { n: 500, dimension: 2 },
        ddimSteps,
        { stream: stream('denoiser-samples') },
      ).steps.at(-1)!.x,
    [net, params, ddimSteps],
  )
  const loss = toFlat(training.series.loss)
  const lossCurve = useMemo(() => ({ x: Array.from(training.index), y: toFlat(training.series.loss) }), [training])
  const sampleCols = useMemo(() => columns(samples), [samples])
  const data = useMemo(() => columns(DATA), [])
  const stepAxis = useAxis({ label: 'training step' })
  const lossAxis = useAxis({ label: '‖ε − ε̂‖² per coordinate' })
  const x1 = useAxis({ label: 'x₁', range: RANGE })
  const x2 = useAxis({ label: 'x₂', range: RANGE, equal: x1 })
  return (
    <Figure
      title="A learned noise predictor"
      purpose="A small MLP trained to predict the added noise learns the data's score well enough that DDIM turns Gaussian draws into mixture samples; short training finds the modes but blurs them."
      state={state}
      defaultSize="L"
      readouts={<Readout label="final loss" value={fmt(loss.at(-1)!)} />}
      caption="aifn denoiserTraining (MLP 11 → 32 → 32 → 2 on x and sinusoidal features of the noise level, Adam with a decaying rate, minibatches of 128, T = 100) on the 600 mixture points of the figures above, and ddimSampler with the trained network. Left: the ε-prediction loss; right: 500 samples (colour) over the training data (grey)."
    >
      <Plots cols={2}>
        <Plot x={stepAxis} y={lossAxis} legend={false}>
          <Curve name="loss" x={lossCurve.x} y={lossCurve.y} />
        </Plot>
        <Plot x={x1} y={x2}>
          <Points name="data" x={data.x} y={data.y} muted thin />
          <Points name="samples" x={sampleCols.x} y={sampleCols.y} slot={1} thin />
        </Plot>
      </Plots>
    </Figure>
  )
}
