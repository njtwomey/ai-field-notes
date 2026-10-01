import { stream, uniform } from 'aifn/foundation/random'
import { toFlat } from 'aifn/foundation/tensor'
import {
  extend,
  now,
  profile,
  run,
  seek,
  timeSliced,
  trace,
  type Algorithm,
  type Status,
  type Trace,
} from 'aifn/foundation/trace'
import { useEffect, useMemo, useRef, useState } from 'react'
import { Slider } from '@lab/controls'
import { Equation, Figure, live, tex } from '@lab/layout'
import { slider, useFigureState, type FigureStateApi } from '@lab/state'
import { Contours, Curve, Handle, Plot, Points, Readout, useAxis } from '@lab/viz'
import { TracePanel } from '@lab/views'

// ---------------------------------------------------------------------------------------------------------------------
// Toy algorithms for the specimens.

type Quadratic = { a: [number, number]; x0: [number, number]; lr: number; momentum: number }
type MomentumState = Status & {
  x: [number, number]
  velocity: [number, number]
  gradient: [number, number]
  loss: number
}

/** Heavy-ball gradient descent on f(x) = ½(a₀x₀² + a₁x₁²), with the gradient and update timed as phases. */
const heavyBall = (a: [number, number], lr: number, momentum: number): Algorithm<Quadratic, MomentumState> => ({
  name: 'heavy-ball',
  init: ({ x0 }) => {
    const gradient: [number, number] = [a[0] * x0[0], a[1] * x0[1]]
    return { t: 0, x: x0, velocity: [0, 0], gradient, loss: 0.5 * (a[0] * x0[0] ** 2 + a[1] * x0[1] ** 2) }
  },
  step: (s) => {
    const gradient = profile('gradient', (): [number, number] => [a[0] * s.x[0], a[1] * s.x[1]])
    return profile('update', () => {
      const velocity: [number, number] = [
        momentum * s.velocity[0] - lr * gradient[0],
        momentum * s.velocity[1] - lr * gradient[1],
      ]
      const x: [number, number] = [s.x[0] + velocity[0], s.x[1] + velocity[1]]
      const loss = 0.5 * (a[0] * x[0] ** 2 + a[1] * x[1] ** 2)
      return { t: s.t + 1, x, velocity, gradient, loss, converged: loss < 1e-10, diverged: !(loss < 1e12) }
    })
  },
})

type Walkers = Status & { x: Float64Array }
/** Eight independent ±1 random walks; step t draws only from its context's stream, `child(root, 'step', t)`. */
const walkers: Algorithm<{ count: number }, Walkers> = {
  name: 'random-walks',
  init: ({ count }) => ({ t: 0, x: new Float64Array(count) }),
  step: (s, ctx) => ({ t: s.t + 1, x: s.x.map((v) => v + (uniform(ctx.stream) < 0.5 ? -1 : 1)) }),
}

type Newton = Status & { x: number; target: number; residual: number }
/** Newton's method for √a (Heron's iteration), converged when the residual x² − a is tiny. */
const heron: Algorithm<{ a: number; x0: number }, Newton> = {
  name: 'heron-square-root',
  init: ({ a, x0 }) => ({ t: 0, x: x0, target: a, residual: x0 * x0 - a }),
  step: (s) => {
    const x = 0.5 * (s.x + s.target / s.x)
    const residual = x * x - s.target
    return { ...s, t: s.t + 1, x, residual, converged: Math.abs(residual) < 1e-12 * s.target }
  },
}

type Logistic = Status & { x: number }
const logisticMap = (r: number): Algorithm<{ x0: number }, Logistic> => ({
  name: 'logistic-map',
  init: ({ x0 }) => ({ x: x0, t: 0 }),
  step: (s) => ({ x: r * s.x * (1 - s.x), t: s.t + 1 }),
})

// ---------------------------------------------------------------------------------------------------------------------

/** The quadratic's curvatures: condition number 12. */
const A: [number, number] = [1, 12]

/** The quadratic's level sets on the drawn window, for the path's backdrop. */
const LEVEL_X = Array.from({ length: 81 }, (_, k) => -5 + k / 8)
const LEVEL_Y = Array.from({ length: 41 }, (_, k) => -2.5 + k / 8)
const LEVEL_Z = LEVEL_Y.map((y) => LEVEL_X.map((x) => 0.5 * (A[0] * x * x + A[1] * y * y)))
const LEVELS = [0.25, 1, 2.5, 5, 10, 20]

const MOMENTUM_SCHEMA = {
  lr: slider(0.005, 0.2, 0.05, { label: 'learning rate η' }),
  momentum: slider(0, 0.98, 0.8, { label: 'momentum β' }),
  sx: slider(-5, 5, -4, { onChart: true, label: 'start x₀' }),
  sy: slider(-2.5, 2.5, 1.5, { onChart: true, label: 'start x₁' }),
}

export function MomentumSpecimen() {
  const state = useFigureState(MOMENTUM_SCHEMA)
  const { lr, momentum, sx, sy } = state
  const t = useMemo(
    () =>
      trace(heavyBall(A, lr, momentum), { a: A, x0: [sx, sy], lr, momentum }, 400, {
        record: {
          loss: (s) => s.loss,
          x: (s) => s.x,
          'gradient norm': (s) => Math.hypot(...s.gradient),
          velocity: (s) => s.velocity,
        },
        checkpointEvery: 50,
      }),
    [lr, momentum, sx, sy],
  )
  return (
    <Figure
      title="Heavy-ball descent on ½(x₀² + 12 x₁²)"
      purpose="Momentum carries velocity across steps: it speeds progress along the shallow x₀ direction and overshoots across the steep x₁ one."
      state={state}
      caption="Drag the start point on the path chart; play or scrub the steps. The path up to the current step is drawn over the whole path (muted) and the level sets of the quadratic. Set β = 0 for plain gradient descent: it zigzags across the valley and crawls along it."
    >
      <TracePanel
        trace={t}
        show={['loss', 'x']}
        renderState={(_, { position, trace: tr }) => <PathSoFar trace={tr} position={position} state={state} />}
      />
    </Figure>
  )
}

/** The iterate's path up to the current kept step, over the whole path muted, with its start draggable. */
function PathSoFar({
  trace: tr,
  position,
  state,
}: {
  trace: Trace<MomentumState>
  position: number
  state: FigureStateApi<typeof MOMENTUM_SCHEMA>
}) {
  const data = useMemo(() => toFlat(tr.series.x), [tr.series.x])
  const kept = tr.index.length
  const xs = useMemo(() => Array.from({ length: kept }, (_, k) => data[2 * k]), [data, kept])
  const ys = useMemo(() => Array.from({ length: kept }, (_, k) => data[2 * k + 1]), [data, kept])
  const soFarX = useMemo(() => xs.slice(0, position + 1), [xs, position])
  const soFarY = useMemo(() => ys.slice(0, position + 1), [ys, position])
  const x = useAxis({ label: 'x₀', range: [-5, 5] })
  const y = useAxis({ label: 'x₁', range: [-2.5, 2.5], equal: x })
  return (
    <Plot x={x} y={y}>
      <Contours x={LEVEL_X} y={LEVEL_Y} z={LEVEL_Z} levels={LEVELS} />
      <Curve name="whole path" x={xs} y={ys} muted />
      <Curve name="so far" x={soFarX} y={soFarY} slot={0} />
      <Points name="current" x={[xs[position]]} y={[ys[position]]} emphasis />
      <Handle {...state.handle(['sx', 'sy'], { label: 'start' })} />
    </Plot>
  )
}

const WALK_RECORD = {
  positions: (s: Walkers) => s.x,
  mean: (s: Walkers) => s.x.reduce((a, b) => a + b, 0) / s.x.length,
  'mean square': (s: Walkers) => s.x.reduce((a, b) => a + b * b, 0) / s.x.length,
}

export function TimeSlicedSpecimen() {
  const state = useFigureState({
    // Drawn by hand below: raising it extends the trace in the slider's own handler.
    steps: slider(500, 20000, 2000, { step: 500, onChart: true, label: 'steps (raising it extends the trace)' }),
    seed: slider(1, 20, 1, { step: 1, label: 'seed' }),
  })
  const { steps, seed } = state
  const [current, setCurrent] = useState<Trace<Walkers> | null>(null)
  const [running, setRunning] = useState(false)
  // The latest requested step count, read when a run finishes (set by the slider, not during render).
  const target = useRef(steps)
  // A new seed restarts, time-sliced over animation frames so the page stays responsive. State is only set after the
  // first slice, asynchronously.
  useEffect(() => {
    let cancelled = false
    void (async () => {
      let last: Trace<Walkers> | null = null
      const slices = timeSliced(walkers, { count: 8 }, target.current, 4, {
        every: 10,
        record: WALK_RECORD,
        stream: stream(seed),
        schedule: (resume) => requestAnimationFrame(() => resume()),
      })
      for await (const partial of slices) {
        if (cancelled) return
        last = partial
        setRunning(true)
        setCurrent(partial)
      }
      // The slider may have been raised during the run: extend to the latest target.
      if (last && last.meta.steps < target.current) setCurrent(extend(last, walkers, target.current - last.meta.steps))
      setRunning(false)
    })()
    return () => {
      cancelled = true
    }
  }, [seed])
  // Raising the step count extends the finished trace by only the new steps; lowering it keeps what was computed.
  const changeSteps = (v: number) => {
    state.set('steps', v)
    target.current = v
    if (!running && current && current.meta.stopped === 'limit' && current.meta.steps < v)
      setCurrent(extend(current, walkers, v - current.meta.steps))
  }
  return (
    <Figure
      title="Eight random walks"
      purpose="A long run is cut into slices that yield to the browser, and a finished trace grows by computing only the new steps."
      state={state}
      controls={
        <Slider
          label="steps (raising it extends the trace)"
          value={steps}
          min={500}
          max={20000}
          step={500}
          onChange={changeSteps}
        />
      }
      readouts={<Readout label="computed" value={`${current?.meta.steps ?? 0} steps${running ? ' (running)' : ''}`} />}
      hoverReadout={!!current}
      caption="Eight independent ±1 walks from 0. The mean square grows like the step count t (each walk's variance is t). Change the seed to restart the time-sliced run; raise the steps and watch 'computed' jump without a restart."
    >
      {current ? (
        <TracePanel trace={current} show={['positions', 'mean square']} />
      ) : (
        <p className="text-xs text-muted-foreground">Starting the first slice…</p>
      )}
    </Figure>
  )
}

export function DivergenceSpecimen() {
  const state = useFigureState({
    lr: slider(0.1, 0.2, 0.175, { step: 0.005, label: 'learning rate η (stable below 2/12 ≈ 0.167)' }),
  })
  const lr = state.lr
  // Plain gradient descent is stable for lr < 2 / max(a) = 1/6.
  const t = useMemo(
    () =>
      trace(heavyBall(A, lr, 0), { a: A, x0: [-4, 1.5], lr, momentum: 0 }, 300, {
        record: { loss: (s) => s.loss, x: (s) => s.x },
      }),
    [lr],
  )
  const factor = Math.abs(1 - lr * A[1])
  return (
    <Figure
      title="Gradient descent near the stability limit"
      purpose="Each step multiplies the steep coordinate by 1 − η·12; once that factor's size passes 1 the loss grows without bound, and the trace stops with stopped = diverged."
      state={state}
      equation={
        <Equation>
          {tex`x_1 \leftarrow (1 - \eta a_1)\, x_1, \quad |1 - ${live(lr, { digits: 3 })} \cdot 12| = ${live(factor, { digits: 3, strong: true })}`}
        </Equation>
      }
      readouts={
        <>
          <Readout label="stopped" value={t.meta.stopped} />
          <Readout label="steps run" value={t.meta.steps} />
        </>
      }
      caption="Play to the end: with the default η the factor is above 1, the loss rises geometrically, and the run halts once it passes 10¹² (diverged). Lower η below 1/6 and the same run converges."
    >
      <TracePanel trace={t} show={['loss']} />
    </Figure>
  )
}

export function HeronSpecimen() {
  const t = useMemo(
    () => trace(heron, { a: 2, x0: 10 }, 100, { record: { x: (s) => s.x, '|residual|': (s) => Math.abs(s.residual) } }),
    [],
  )
  return (
    <Figure
      title="Heron's iteration for √2 from x₀ = 10"
      purpose="A trace stops as soon as its algorithm reports done: Newton's iteration for √2 converges in a handful of steps of the 100 allowed."
      readouts={
        <>
          <Readout label="stopped" value={t.meta.stopped} />
          <Readout label="steps run" value={`${t.meta.steps} of 100`} />
        </>
      }
      caption="x ← (x + 2/x)/2. The residual |x² − 2| shrinks quadratically once x is near √2: the digits double each step. Switch the residual to a log scale to see it."
    >
      <TracePanel trace={t} />
    </Figure>
  )
}

const SEEK_N = 200000

export function SeekSpecimen() {
  const r = 3.9
  const alg = useMemo(() => logisticMap(r), [])
  const state = useFigureState({ i: slider(0, SEEK_N, 123456, { step: 1, label: 'step i' }) })
  const i = state.i
  const stored = useMemo(() => trace(alg, { x0: 0.2 }, SEEK_N, { every: SEEK_N, checkpointEvery: 1000 }), [alg])
  const result = useMemo(() => {
    const t0 = now()
    const fromCheckpoints = seek(alg, { x0: 0.2 }, i, { checkpoints: stored })
    const t1 = now()
    const fromScratch = run(alg, { x0: 0.2 }, i)
    const t2 = now()
    return { fromCheckpoints, fromScratch, seekMs: t1 - t0, runMs: t2 - t1 }
  }, [alg, i, stored])
  // A window of the logistic map around step i, from the checkpoints, with i as a draggable cursor.
  const orbit = useMemo(() => {
    const from = Math.max(0, Math.min(i - 60, SEEK_N - 120))
    const start = seek(alg, { x0: 0.2 }, from, { checkpoints: stored })
    // The map's state is x alone, so a trace started at x_from continues the orbit exactly.
    const local = trace(alg, { x0: start.x }, 120, { record: { x: (s) => s.x } })
    return { x: Array.from(local.index, (k) => k + from), y: toFlat(local.series.x) }
  }, [alg, i, stored])
  const x = useAxis({ label: 'step', format: (v) => v.toFixed(0) })
  const y = useAxis({ label: 'x', range: [0, 1] })
  return (
    <Figure
      title="The logistic map around step i"
      purpose="seek(i) restarts from the nearest stored checkpoint instead of step 0, and returns exactly what run(i) returns, in a fraction of the time."
      state={state}
      readouts={{
        'state at step i': (
          <>
            <Readout label="seek(i) from checkpoints" value={result.fromCheckpoints.x.toPrecision(12)} />
            <Readout label="run(i)" value={result.fromScratch.x.toPrecision(12)} />
            <Readout label="equal" value={String(result.fromCheckpoints.x === result.fromScratch.x)} />
          </>
        ),
        cost: (
          <>
            <Readout label="seek" value={`${result.seekMs.toFixed(3)} ms`} />
            <Readout label="run" value={`${result.runMs.toFixed(3)} ms`} />
            <Readout label="checkpoints" value={stored.checkpoints.index.length} />
          </>
        ),
      }}
      caption="The logistic map x ← 3.9 x(1 − x) is chaotic, so any error would show at once: the two answers agree to every digit. Drag the vertical line to move i within the window (the window recentres on release), or type i."
    >
      <Plot x={x} y={y}>
        <Curve name="x" x={orbit.x} y={orbit.y} slot={0} showPoints />
        <Handle {...state.handle('i', { label: 'i' })} />
      </Plot>
    </Figure>
  )
}
