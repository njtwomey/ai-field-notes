import { useMemo } from 'react'
import {
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { normal, stream, uniform } from 'aifn/foundation/random'

// Sine-wave regression tasks as in Finn et al. (2017): y = A sin(x − φ), A ∈ [0.1, 5], φ ∈ [0, π], x ∈ [−5, 5].
// The model is a network with one tanh hidden layer. Three initialisations are adapted to a new task by the same
// full-batch gradient steps on K support points.
const HIDDEN = 20
const INNER_LR = 0.01
const MAX_STEPS = 10
const X_GRID = toFlat(linspace(-5, 5, 101))

type Net = { w1: Float64Array; b1: Float64Array; w2: Float64Array; b2: number }

const clone = (p: Net): Net => ({ w1: p.w1.slice(), b1: p.b1.slice(), w2: p.w2.slice(), b2: p.b2 })

function initNet(seed: number): Net {
  const r = stream(seed)
  const w1 = Float64Array.from({ length: HIDDEN }, () => normal(r))
  const w2 = Float64Array.from({ length: HIDDEN }, () => normal(r) / Math.sqrt(HIDDEN))
  return { w1, b1: new Float64Array(HIDDEN), w2, b2: 0 }
}

function predict(p: Net, x: number) {
  let out = p.b2
  for (let j = 0; j < HIDDEN; j++) out += p.w2[j] * Math.tanh(p.w1[j] * x + p.b1[j])
  return out
}

/** One full-batch gradient step on the mean squared error over (xs, ys), in place. */
function step(p: Net, xs: number[], ys: number[], lr: number) {
  const g1 = new Float64Array(HIDDEN)
  const gb1 = new Float64Array(HIDDEN)
  const g2 = new Float64Array(HIDDEN)
  let gb2 = 0
  const z = new Float64Array(HIDDEN)
  for (let i = 0; i < xs.length; i++) {
    let out = p.b2
    for (let j = 0; j < HIDDEN; j++) {
      z[j] = Math.tanh(p.w1[j] * xs[i] + p.b1[j])
      out += p.w2[j] * z[j]
    }
    const e = (2 * (out - ys[i])) / xs.length
    gb2 += e
    for (let j = 0; j < HIDDEN; j++) {
      g2[j] += e * z[j]
      const dz = e * p.w2[j] * (1 - z[j] * z[j])
      g1[j] += dz * xs[i]
      gb1[j] += dz
    }
  }
  for (let j = 0; j < HIDDEN; j++) {
    p.w1[j] -= lr * g1[j]
    p.b1[j] -= lr * gb1[j]
    p.w2[j] -= lr * g2[j]
  }
  p.b2 -= lr * gb2
}

/**
 * Reptile (Nichol et al., 2018): adapt a copy with `inner` steps on a sampled task, then move the initialisation a
 * fraction ε of the way towards the adapted weights. With one inner step and ε = 1 this is plain SGD on the pooled
 * tasks, i.e. ordinary pretraining.
 */
function metaTrain(inner: number, epsilon: number, iterations: number, seed: number): Net {
  const r = stream(seed)
  const p = initNet(seed + 1)
  for (let it = 0; it < iterations; it++) {
    const A = 0.1 + 4.9 * uniform(r)
    const phase = Math.PI * uniform(r)
    const xs = Array.from({ length: 10 }, () => -5 + 10 * uniform(r))
    const ys = xs.map((x) => A * Math.sin(x - phase))
    const q = clone(p)
    for (let s = 0; s < inner; s++) step(q, xs, ys, INNER_LR)
    const eps = epsilon * (1 - it / iterations)
    for (let j = 0; j < HIDDEN; j++) {
      p.w1[j] += eps * (q.w1[j] - p.w1[j])
      p.b1[j] += eps * (q.b1[j] - p.b1[j])
      p.w2[j] += eps * (q.w2[j] - p.w2[j])
    }
    p.b2 += eps * (q.b2 - p.b2)
  }
  return p
}

let cache: { meta: Net; pretrained: Net; scratch: Net } | null = null
function initialisations() {
  cache ??= {
    meta: metaTrain(10, 0.5, 6000, 11),
    pretrained: metaTrain(1, 1, 6000, 11),
    scratch: initNet(7),
  }
  return cache
}

const NAMES = ['meta-learned (Reptile)', 'pretrained on pooled tasks', 'random initialisation'] as const

export function MetaSine() {
  const state = useFigureState({
    amplitude: float(3, { min: 0.1, max: 5, step: 0.1, label: 'task amplitude A' }),
    phase: slider(0, 3.1, 1, { step: 0.05, label: 'task phase φ' }),
    k: int(10, { min: 2, max: 20, step: 1, label: 'support points K' }),
    steps: int(1, { min: 0, max: MAX_STEPS, step: 1, label: 'gradient steps', format: (v) => String(v) }),
    draw: int(0, { ge: 0, label: 'seed' }),
  })

  const support = useMemo(() => {
    const r = stream(100 + state.draw)
    const xs = Array.from({ length: state.k }, () => -5 + 10 * uniform(r))
    return { xs, ys: xs.map((x) => state.amplitude * Math.sin(x - state.phase)) }
  }, [state.k, state.draw, state.amplitude, state.phase])

  // Adapt each initialisation for MAX_STEPS steps once; the step slider only chooses which snapshot to show.
  const paths = useMemo(() => {
    const inits = initialisations()
    const truth = X_GRID.map((x) => state.amplitude * Math.sin(x - state.phase))
    return [inits.meta, inits.pretrained, inits.scratch].map((p0) => {
      const p = clone(p0)
      const curves: number[][] = []
      const mse: number[] = []
      for (let s = 0; s <= MAX_STEPS; s++) {
        const curve = X_GRID.map((x) => predict(p, x))
        curves.push(curve)
        mse.push(curve.reduce((a, v, i) => a + (v - truth[i]) ** 2, 0) / X_GRID.length)
        step(p, support.xs, support.ys, INNER_LR)
      }
      return { curves, mse }
    })
  }, [support, state.amplitude, state.phase])

  const s = state.steps
  const fitSeries = useMemo<SeriesSpec[]>(
    () => [
      {
        name: 'task',
        type: 'line',
        x: X_GRID,
        y: X_GRID.map((x) => state.amplitude * Math.sin(x - state.phase)),
        emphasis: true,
      },
      ...paths.map((p, i) => ({ name: NAMES[i], type: 'line' as const, x: X_GRID, y: p.curves[s], slot: i })),
      { name: `${state.k} support points`, type: 'scatter', x: support.xs, y: support.ys, emphasis: true },
    ],
    [paths, s, state.amplitude, state.phase, support, state.k],
  )
  const lossSeries = useMemo<SeriesSpec[]>(() => {
    const t = Array.from({ length: MAX_STEPS + 1 }, (_, i) => i)
    return paths.map((p, i) => ({ name: NAMES[i], type: 'line' as const, x: t, y: p.mse, slot: i }))
  }, [paths])

  const xAxis = useAxis({ label: 'x', range: [-5, 5] })
  const yAxis = useAxis({ label: 'y', range: [-6, 6] })
  const xAxis2 = useAxis({ label: 'gradient steps', range: [0, MAX_STEPS] })
  const yAxis2 = useAxis({ label: 'mean squared error', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="Adapting to a new sine wave from three initialisations"
      state={state}
      caption="Each initialisation takes the same gradient steps (learning rate 0.01) on the support points of a new task. The meta-learned initialisation was trained by Reptile, a first-order relative of MAML, over 6000 random sine tasks; the pretrained one by plain SGD on the same tasks, which learns their average and adapts slowly. Step through the updates with the arrows or drag the line on the loss chart."

      readouts={
        <>
          {paths.map((p, i) => (
            <Readout key={NAMES[i]} label={`MSE, ${NAMES[i]}`} value={formatNumber(p.mse[s])} />
          ))}
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={300}>
          {seriesLayers(fitSeries)}
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={300}>
          {seriesLayers(lossSeries)}
          <Handle kind="x" at={s} label="step" onDrag={(v: number) => state.set('steps', v)} />
        </Plot>
      </div>
    </Figure>
  )
}
