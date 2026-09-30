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
import { Player, Select, Slider } from '@lab/controls'
import { Panel, Readout, Subplots, XYChart, formatNumber, type XYSeries } from '@lab/viz'

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

export function ForwardNoisingSpecimen() {
  const [id, setId] = useState<ScheduleId>('linear')
  const [k, setK] = useState(30)
  const schedule = SCHEDULES[id]
  const t = TICKS[k]
  const ab = alphaBarAt(schedule, t)
  // The same noise draw at every t, so each point moves smoothly from its data position towards N(0, I).
  const xt = useMemo(() => forwardNoise(stream('diffusion-eps'), DATA, ab).x, [ab])
  const points: XYSeries[] = [{ name: 'x_t', type: 'scatter', ...columns(xt), slot: 0 }]
  const curves = useMemo<XYSeries[]>(
    () =>
      (Object.keys(SCHEDULES) as ScheduleId[]).flatMap((name, slot) => [
        {
          name: `√ᾱ (${name})`,
          type: 'line',
          x: TICKS,
          y: TICKS.map((s) => Math.sqrt(alphaBarAt(SCHEDULES[name], s))),
          slot,
        },
        {
          name: `√(1 − ᾱ) (${name})`,
          type: 'line',
          dashed: true,
          x: TICKS,
          y: TICKS.map((s) => Math.sqrt(1 - alphaBarAt(SCHEDULES[name], s))),
          slot,
        },
      ]),
    [],
  )
  return (
    <Figure
      title="Forward noising of a 2-D mixture"
      description="The forward process shrinks the data by √ᾱ_t and adds noise of standard deviation √(1 − ᾱ_t), so the mixture's modes blur into one standard normal; the cosine schedule keeps more signal early and destroys it more evenly."
      defaultSize="L"
      controls={
        <>
          <ControlRow label="1 · schedule">
            <Select
              label="schedule"
              value={id}
              onChange={setId}
              options={[
                { value: 'linear', label: 'linear (Ho et al.)' },
                { value: 'cosine', label: 'cosine (Nichol & Dhariwal)' },
              ]}
            />
          </ControlRow>
          <ControlRow label="2 · step">
            <Player value={k} onChange={setK} count={TICKS.length} format={(i) => String(TICKS[i])} label="t" />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout label="t" value={String(t)} />
          <Readout label="ᾱ_t" value={fmt(ab)} />
          <Readout label="SNR" value={fmt(ab / (1 - ab))} />
        </>
      }
      caption="aifn/diffusion forwardNoise of 600 mixture samples with one fixed noise draw, x_t = √ᾱ_t x₀ + √(1 − ᾱ_t) ε, T = 1000. Right: the signal and noise scales of both schedules; drag the step."
    >
      <Subplots cols={2} widthRatios={[1, 1.2]}>
        <Panel>
          <XYChart series={points} aspect="equal" xRange={RANGE} yRange={RANGE} xLabel="x₁" yLabel="x₂" />
        </Panel>
        <Panel>
          <XYChart
            series={curves}
            xLabel="step t"
            yLabel="scale"
            yRange={[0, 1]}
            handles={[
              { kind: 'x', at: t, label: 't', onDrag: (x) => setK(Math.max(0, Math.min(100, Math.round(x / 10)))) },
            ]}
          />
        </Panel>
      </Subplots>
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
  const [ddimSteps, setDdimSteps] = useState(50)
  const [odeSteps, setOdeSteps] = useState(40)
  const [frame, setFrame] = useState(FRAMES)
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
  const data = useMemo(() => ({ name: 'data', type: 'scatter' as const, muted: true, ...columns(DATA) }), [])
  const panel = (run: Run, slot: number): XYSeries[] => {
    const upto = run.trace.steps.slice(0, Math.round(f * (run.trace.steps.length - 1)) + 1)
    const paths: XYSeries[] = Array.from({ length: PATHS }, (_, i) => ({
      name: 'paths',
      type: 'line',
      thin: true,
      slot,
      x: upto.map((s) => s.x.data[2 * i] as number),
      y: upto.map((s) => s.x.data[2 * i + 1] as number),
    }))
    return [data, { name: run.label, type: 'scatter', slot, ...columns(at(run, f).x) }, ...paths]
  }
  const runs = [ddpm, ddim, ode]
  return (
    <Figure
      title="DDPM against DDIM against the probability-flow ODE"
      description="All three samplers turn the same N(0, I) draws into samples of the mixture using its exact score; DDPM's paths are rough and need a thousand steps, while DDIM and the probability-flow ODE follow smooth deterministic paths in tens of steps."
      defaultSize="XL"
      controls={
        <>
          <ControlRow label="1 · step budgets">
            <Slider label="DDIM steps" value={ddimSteps} min={5} max={200} step={5} onChange={setDdimSteps} />
            <Slider label="ODE steps (RK4)" value={odeSteps} min={5} max={100} step={5} onChange={setOdeSteps} />
          </ControlRow>
          <ControlRow label="2 · progress">
            <Player
              value={frame}
              onChange={setFrame}
              count={FRAMES + 1}
              format={(i) => `${Math.round((100 * i) / FRAMES)}%`}
              label="progress"
            />
          </ControlRow>
        </>
      }
      readouts={runs.map((r) => (
        <Readout
          key={r.label}
          label={`${r.label}: predictor calls`}
          value={String(r.trace.steps.at(-1)!.evaluations)}
        />
      ))}
      caption={`aifn/diffusion samplers driven by mixtureNoisePredictor (no training), 500 particles from the same start, with ${PATHS} paths drawn; grey: data. The probability-flow ODE runs on the continuous VP SDE (β from 0.1 to 20), the others on the linear schedule with T = 1000.`}
    >
      <Subplots cols={3} sharex sharey>
        {runs.map((r, slot) => (
          <Panel key={r.label}>
            <XYChart
              series={panel(r, slot)}
              legend={false}
              xRange={RANGE}
              yRange={RANGE}
              xLabel={r.label}
              yLabel="x₂"
            />
          </Panel>
        ))}
      </Subplots>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 3. A learned denoiser.

const SMALL = linearSchedule(100, { betaEnd: 0.2 })

export function LearnedDenoiserSpecimen() {
  const [steps, setSteps] = useState('800')
  const [ddimSteps, setDdimSteps] = useState(40)
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
  return (
    <Figure
      title="A learned noise predictor"
      description="A small MLP trained to predict the added noise learns the score of the data well enough that DDIM turns Gaussian draws into samples of the mixture; with a few hundred steps of training the modes are found but blurred, where the exact score gives sharp ones."
      defaultSize="L"
      controls={
        <ControlRow label="1 · training and sampling">
          <Select label="training steps" value={steps} onChange={setSteps} options={['200', '800', '2000']} />
          <Slider label="DDIM steps" value={ddimSteps} min={5} max={200} step={5} onChange={setDdimSteps} />
        </ControlRow>
      }
      readouts={<Readout label="final loss" value={fmt(loss.at(-1)!)} />}
      caption="aifn/diffusion denoiserTraining (Mlp 11 → 32 → 32 → 2 on x and sinusoidal features of the noise level, Adam with a decaying rate, minibatches of 128, T = 100) on the 600 mixture points of the figures above, and ddimSampler with the trained network. Left: the ε-prediction loss; right: 500 samples (colour) over the training data (grey)."
    >
      <Subplots cols={2}>
        <Panel>
          <XYChart
            series={[{ name: 'loss', type: 'line', x: Array.from(training.index), y: loss }]}
            xLabel="training step"
            yLabel="‖ε − ε̂‖² per coordinate"
          />
        </Panel>
        <Panel>
          <XYChart
            series={[
              { name: 'data', type: 'scatter', muted: true, ...columns(DATA) },
              { name: 'samples', type: 'scatter', slot: 1, ...columns(samples) },
            ]}
            aspect="equal"
            xRange={RANGE}
            yRange={RANGE}
            xLabel="x₁"
            yLabel="x₂"
          />
        </Panel>
      </Subplots>
    </Figure>
  )
}
