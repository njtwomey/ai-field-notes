import { useMemo } from 'react'
import {
  Bars,
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { normal, stream, uniform } from 'aifn-compute/foundation/random'

/**
 * A one-dimensional energy-based model: E_θ(x) = x²/18 + Σ_k θ_k φ_k(x), with nine Gaussian bumps φ_k of width 0.6
 * centred on −3, …, 3. The fixed quadratic keeps the density normalisable for every θ.
 */
const CENTRES = toFlat(linspace(-3, 3, 9))
const BW = 0.6
const CONF = 1 / 18
const K = CENTRES.length
const LO = -4.5
const HI = 4.5
const GRID = toFlat(linspace(LO, HI, 181))
const DX = GRID[1] - GRID[0]
const ITERATIONS = 200
const NEG = 64
const LR = 0.5

const features = (x: number) => CENTRES.map((c) => Math.exp(-((x - c) ** 2) / (2 * BW * BW)))
const energy = (x: number, th: number[]) => {
  const f = features(x)
  return CONF * x * x + f.reduce((s, v, k) => s + v * th[k], 0)
}
const energyGrad = (x: number, th: number[]) =>
  2 * CONF * x +
  CENTRES.reduce((s, c, k) => s - th[k] * Math.exp(-((x - c) ** 2) / (2 * BW * BW)) * ((x - c) / (BW * BW)), 0)

// Training data: 600 draws from a two-component mixture.
const DATA = (() => {
  const g = stream(11)
  return Array.from({ length: 600 }, (_, i) => (i % 2 === 0 ? -1.5 + 0.45 * normal(g) : 1.2 + 0.6 * normal(g)))
})()
const DATA_FEATURES = DATA.map(features)
const DATA_MEAN_FEATURES = CENTRES.map((_, k) => DATA_FEATURES.reduce((s, f) => s + f[k], 0) / DATA.length)

/** Normalised model density on the grid and the average negative log-likelihood of the data. */
function densityAndNll(th: number[]) {
  const e = GRID.map((x) => energy(x, th))
  const lo = Math.min(...e)
  const w = e.map((v) => Math.exp(-(v - lo)))
  const z = w.reduce((s, v) => s + v, 0) * DX
  const logZ = Math.log(z) - lo
  const meanE = DATA.reduce((s, x) => s + energy(x, th), 0) / DATA.length
  return { density: w.map((v) => v / z), nll: meanE + logZ }
}

/** The maximum-likelihood θ in this family, by gradient ascent with exact expectations on the grid. */
const EXACT_NLL = (() => {
  const th = new Array<number>(K).fill(0)
  const gridFeatures = GRID.map(features)
  for (let it = 0; it < 1500; it++) {
    const { density } = densityAndNll(th)
    for (let k = 0; k < K; k++) {
      const model = gridFeatures.reduce((s, f, i) => s + f[k] * density[i] * DX, 0)
      th[k] += 1.5 * (model - DATA_MEAN_FEATURES[k])
    }
  }
  return densityAndNll(th).nll
})()

type Sampler = 'short-run' | 'persistent' | 'cd'

/**
 * Stochastic maximum likelihood. Each iteration draws negative samples by `steps` Langevin updates
 * x ← x − h∇E(x) + s√(2h)ξ, then moves θ by LR × (mean features of negatives − mean features of a data batch), which is
 * the gradient of the average log-likelihood with the model expectation replaced by the sample average.
 */
function train(sampler: Sampler, steps: number, h: number, s: number, seed: number) {
  const g = stream(seed)
  const th = new Array<number>(K).fill(0)
  let buffer = Array.from({ length: NEG }, () => LO + (HI - LO) * uniform(g))
  const thetas: number[][] = [th.slice()]
  const negatives: number[][] = [buffer.slice()]
  const phases: { pos: number; neg: number }[] = []
  const sd = s * Math.sqrt(2 * h)
  for (let it = 0; it < ITERATIONS; it++) {
    let x: number[]
    if (sampler === 'short-run') x = Array.from({ length: NEG }, () => LO + (HI - LO) * uniform(g))
    else if (sampler === 'persistent') x = buffer
    else x = Array.from({ length: NEG }, () => DATA[Math.floor(uniform(g) * DATA.length)])
    x = x.map((v) => {
      let u = v
      for (let t = 0; t < steps; t++) u = u - h * energyGrad(u, th) + sd * normal(g)
      // Keep diverged chains finite so that one bad sample shows up as a bad update rather than NaN.
      return Math.max(-50, Math.min(50, u))
    })
    buffer = x
    const pos = Array.from({ length: NEG }, () => Math.floor(uniform(g) * DATA.length))
    const negF = x.map(features)
    phases.push({
      pos: pos.reduce((acc, i) => acc + energy(DATA[i], th), 0) / NEG,
      neg: x.reduce((acc, v) => acc + energy(v, th), 0) / NEG,
    })
    for (let k = 0; k < K; k++) {
      const model = negF.reduce((acc, f) => acc + f[k], 0) / NEG
      const data = pos.reduce((acc, i) => acc + DATA_FEATURES[i][k], 0) / NEG
      th[k] += LR * (model - data)
    }
    thetas.push(th.slice())
    negatives.push(x.slice())
  }
  const nll = thetas.map((t) => densityAndNll(t).nll)
  return { thetas, negatives, phases, nll }
}

const BINS = toFlat(linspace(LO, HI, 37))
const BIN_CENTRES = BINS.slice(0, -1).map((b, i) => (b + BINS[i + 1]) / 2)
function histogram(xs: number[]) {
  const width = BINS[1] - BINS[0]
  const counts = new Array<number>(BIN_CENTRES.length).fill(0)
  for (const x of xs) {
    const b = Math.floor((x - LO) / width)
    if (b >= 0 && b < counts.length) counts[b]++
  }
  return counts.map((c) => c / (xs.length * width))
}
const DATA_HISTOGRAM = histogram(DATA)
const ITERATION_AXIS = Array.from({ length: ITERATIONS + 1 }, (_, i) => i)

export function EbmTraining() {
  const state = useFigureState({
    sampler: choice<Sampler>(
      [
        { value: 'short-run', label: 'short-run from noise' },
        { value: 'persistent', label: 'persistent (PCD)' },
        { value: 'cd', label: 'from data (CD)' },
      ],
      'persistent',
      { label: 'negative samples' },
    ),
    steps: int(20, { min: 1, max: 60, step: 1, label: 'Langevin steps per iteration', format: (v) => String(v) }),
    h: float(0.05, { min: 0.01, max: 0.2, step: 0.01, label: 'Langevin step h' }),
    s: float(1, { min: 0, max: 1, step: 0.05, label: 'noise scale s' }),
    iteration: float(ITERATIONS, {
      min: 0,
      max: ITERATIONS,
      step: 1,
      label: 'training iteration',
      format: (v) => String(v),
    }),
    seed: int(1, { min: 1, max: 20, step: 1, label: 'seed', format: (v) => String(v) }),
  })

  const run = useMemo(
    () => train(state.sampler, state.steps, state.h, state.s, state.seed),
    [state.sampler, state.steps, state.h, state.s, state.seed],
  )
  const it = state.iteration

  const densitySeries = useMemo(() => {
    const { density } = densityAndNll(run.thetas[it])
    return [
      { name: 'data', x: BIN_CENTRES, y: DATA_HISTOGRAM, muted: true },
      {
        name: 'Langevin samples',
        x: BIN_CENTRES,
        y: histogram(run.negatives[it]),
        slot: 1,
        dashed: true,
      },
      { name: 'model density', x: GRID, y: density, slot: 0 },
    ] as const
  }, [run, it])

  const lossSeries = useMemo(
    () =>
      [
        {
          name: 'data negative log-likelihood',
          x: ITERATION_AXIS,
          y: run.nll.map((v) => Math.min(v, 3)),
          slot: 0,
        },
        {
          name: 'maximum-likelihood optimum',
          x: [0, ITERATIONS],
          y: [EXACT_NLL, EXACT_NLL],
          emphasis: true,
          dashed: true,
        },
      ] as const,
    [run],
  )

  const phase = run.phases[Math.max(0, it - 1)]

  const xAxis = useAxis({ label: 'x', range: [LO, HI] })
  const yAxis = useAxis({ label: 'density', range: [0, 0.8] })
  const xAxis2 = useAxis({ label: 'training iteration', range: [0, ITERATIONS] })
  const yAxis2 = useAxis({ label: 'negative log-likelihood', range: [1.3, 3] })
  return (
    <Figure
      title="Training a one-dimensional energy-based model with Langevin samples"
      state={state}
      caption="The energy is a fixed quadratic plus nine learned Gaussian bumps. Each training iteration runs Langevin chains to draw negative samples, then lowers the energy where the data are and raises it where the samples are. Grey bars are the data (positive phase), the dashed line the Langevin samples (negative phase) and the solid line the model density. Drag the iteration line on the right or use the slider. Persistent chains with s = 1 fit the data and approach the maximum-likelihood optimum. Short-run chains restarted from uniform noise learn a model whose 20-step samples match the data closely while the density p_θ itself fits worse than with persistent chains; with only a few steps the density drifts far from the data. Lowering the noise scale s makes the samples cluster at the modes of the energy, so the model is pushed to spread its density out and the fit degrades. Contrastive divergence starts its chains at the data; with few steps they stay near the data, so the energy far from the data is never corrected."

      readouts={
        <>
          <Readout label="negative log-likelihood" value={formatNumber(run.nll[it])} />
          <Readout label="maximum-likelihood optimum" value={formatNumber(EXACT_NLL)} />
          <Readout label="mean energy of data" value={phase ? formatNumber(phase.pos) : '—'} />
          <Readout label="mean energy of samples" value={phase ? formatNumber(phase.neg) : '—'} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={340}>
          <Bars {...densitySeries[0]} />
          <Curve {...densitySeries[1]} />
          <Curve {...densitySeries[2]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={340}>
          <Curve {...lossSeries[0]} />
          <Curve {...lossSeries[1]} />
          <Handle kind="x" at={it} onDrag={(x) => state.set('iteration', x)} />
        </Plot>
      </div>
    </Figure>
  )
}
