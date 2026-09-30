import { stream, type Stream } from 'aifn/random'
import { toFlat } from 'aifn/tensor'
import { extend, now, profile, run, seek, timeSliced, trace, type Algorithm, type Trace } from 'aifn/trace'
import { useEffect, useMemo, useRef, useState } from 'react'
import { Slider } from '@lab/controls'
import { Figure } from '@lab/layout'
import { Readout, XYChart } from '@lab/viz'
import { TraceView } from '@lab/views'

// ---------------------------------------------------------------------------------------------------------------------
// Toy algorithms for the specimens.

type Quadratic = { a: [number, number]; x0: [number, number]; lr: number; momentum: number }
type MomentumState = {
  x: [number, number]
  velocity: [number, number]
  gradient: [number, number]
  loss: number
  diverged: boolean
}

/** Heavy-ball gradient descent on f(x) = ½(a₀x₀² + a₁x₁²), with the gradient and update timed as phases. */
const heavyBall = (a: [number, number], lr: number, momentum: number): Algorithm<Quadratic, MomentumState> => ({
  name: 'heavy-ball',
  init: ({ x0 }) => {
    const gradient: [number, number] = [a[0] * x0[0], a[1] * x0[1]]
    return { x: x0, velocity: [0, 0], gradient, loss: 0.5 * (a[0] * x0[0] ** 2 + a[1] * x0[1] ** 2), diverged: false }
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
      return { x, velocity, gradient, loss, diverged: !(loss < 1e12) }
    })
  },
  done: (s) => s.loss < 1e-10,
})

type Walkers = { x: Float64Array; t: number; stream: Stream }
/** Eight independent ±1 random walks; step t draws from the substream keyed by t, so steps are pure. */
const walkers: Algorithm<{ count: number }, Walkers> = {
  name: 'random-walks',
  init: ({ count }, s) => ({ x: new Float64Array(count), t: 0, stream: s ?? stream(0) }),
  step: (s) => {
    const draws = s.stream.child('step', s.t)
    return { ...s, x: s.x.map((v) => v + (draws.uniform() < 0.5 ? -1 : 1)), t: s.t + 1 }
  },
}

type Newton = { x: number; target: number; residual: number }
/** Newton's method for √a (Heron's iteration), done when the residual x² − a is tiny. */
const heron: Algorithm<{ a: number; x0: number }, Newton> = {
  name: 'heron-square-root',
  init: ({ a, x0 }) => ({ x: x0, target: a, residual: x0 * x0 - a }),
  step: (s) => {
    const x = 0.5 * (s.x + s.target / s.x)
    return { ...s, x, residual: x * x - s.target }
  },
  done: (s) => Math.abs(s.residual) < 1e-12 * s.target,
}

type Logistic = { x: number; t: number }
const logisticMap = (r: number): Algorithm<{ x0: number }, Logistic> => ({
  name: 'logistic-map',
  init: ({ x0 }) => ({ x: x0, t: 0 }),
  step: (s) => ({ x: r * s.x * (1 - s.x), t: s.t + 1 }),
})

// ---------------------------------------------------------------------------------------------------------------------

/** The quadratic's curvatures: condition number 12. */
const A: [number, number] = [1, 12]

export function MomentumSpecimen() {
  const [momentum, setMomentum] = useState(0.8)
  const [lr, setLr] = useState(0.05)
  const t = useMemo(
    () =>
      trace(heavyBall(A, lr, momentum), { a: A, x0: [-4, 1.5], lr, momentum }, 400, {
        record: {
          loss: (s) => s.loss,
          x: (s) => s.x,
          'gradient norm': (s) => Math.hypot(...s.gradient),
          velocity: (s) => s.velocity,
        },
        checkpointEvery: 50,
      }),
    [lr, momentum],
  )
  return (
    <TraceView
      title="Heavy-ball descent on ½(x₀² + 12 x₁²)"
      trace={t}
      show={['loss', 'x']}
      controls={
        <>
          <Slider label="learning rate" value={lr} min={0.005} max={0.2} onChange={setLr} />
          <Slider label="momentum" value={momentum} min={0} max={0.98} onChange={setMomentum} />
        </>
      }
      caption="Drag the step cursor on any panel to scrub; hover one panel to read every panel at that step."
      renderState={(_, { position, trace: tr }) => <PathSoFar trace={tr} position={position} />}
    />
  )
}

/** The iterate's path up to the current kept step, over the whole path muted. */
function PathSoFar({ trace: tr, position }: { trace: Trace<MomentumState>; position: number }) {
  const data = useMemo(() => toFlat(tr.series.x), [tr.series.x])
  const kept = tr.index.length
  const xs = useMemo(() => Array.from({ length: kept }, (_, k) => data[2 * k]), [data, kept])
  const ys = useMemo(() => Array.from({ length: kept }, (_, k) => data[2 * k + 1]), [data, kept])
  const whole = useMemo(() => ({ name: 'whole path', type: 'line' as const, x: xs, y: ys, muted: true }), [xs, ys])
  return (
    <XYChart
      equalAspect
      xRange={[-5, 5]}
      yRange={[-2.5, 2.5]}
      xLabel="x₀"
      yLabel="x₁"
      series={[
        whole,
        { name: 'so far', type: 'line', x: xs.slice(0, position + 1), y: ys.slice(0, position + 1), slot: 0 },
        { name: 'current', type: 'scatter', x: [xs[position]], y: [ys[position]], emphasis: true },
      ]}
    />
  )
}

const WALK_RECORD = {
  positions: (s: Walkers) => s.x,
  mean: (s: Walkers) => s.x.reduce((a, b) => a + b, 0) / s.x.length,
  'mean square': (s: Walkers) => s.x.reduce((a, b) => a + b * b, 0) / s.x.length,
}

export function TimeSlicedSpecimen() {
  const [steps, setSteps] = useState(2000)
  const [seed, setSeed] = useState(1)
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
      if (last && last.meta.steps < target.current)
        setCurrent(extend(last, walkers, { count: 8 }, target.current - last.meta.steps))
      setRunning(false)
    })()
    return () => {
      cancelled = true
    }
  }, [seed])
  // Raising the step count extends the finished trace by only the new steps; lowering it keeps what was computed.
  const changeSteps = (v: number) => {
    setSteps(v)
    target.current = v
    if (!running && current && current.meta.stopped === 'limit' && current.meta.steps < v)
      setCurrent(extend(current, walkers, { count: 8 }, v - current.meta.steps))
  }
  const controls = (
    <>
      <Slider
        label="steps (raising it extends the trace)"
        value={steps}
        min={500}
        max={20000}
        step={500}
        onChange={changeSteps}
      />
      <Slider label="seed" value={seed} min={1} max={20} step={1} onChange={setSeed} />
    </>
  )
  const computed = (
    <Readout label="computed" value={`${current?.meta.steps ?? 0} steps${running ? ' (running)' : ''}`} />
  )
  return current ? (
    <TraceView
      title="Eight random walks"
      trace={current}
      show={['positions', 'mean square']}
      controls={controls}
      readouts={computed}
    />
  ) : (
    <Figure title="Eight random walks" controls={controls} readouts={computed} hoverReadout={false}>
      <p className="text-xs text-muted-foreground">Starting the first slice…</p>
    </Figure>
  )
}

export function DivergenceSpecimen() {
  const [lr, setLr] = useState(0.16)
  // Plain gradient descent is stable for lr < 2 / max(a) = 1/6.
  const t = useMemo(
    () =>
      trace(heavyBall(A, lr, 0), { a: A, x0: [-4, 1.5], lr, momentum: 0 }, 300, {
        record: { loss: (s) => s.loss, x: (s) => s.x },
      }),
    [lr],
  )
  return (
    <TraceView
      title="Gradient descent near the stability limit"
      trace={t}
      show={['loss']}
      controls={
        <Slider
          label="learning rate (stable below 2/12 ≈ 0.167)"
          value={lr}
          min={0.1}
          max={0.2}

          onChange={setLr}
        />
      }
    />
  )
}

export function HeronSpecimen() {
  const t = useMemo(
    () => trace(heron, { a: 2, x0: 10 }, 100, { record: { x: (s) => s.x, '|residual|': (s) => Math.abs(s.residual) } }),
    [],
  )
  return <TraceView title="Heron's iteration for √2 from x₀ = 10" trace={t} startAtFirst />
}

export function SeekSpecimen() {
  const r = 3.9
  const n = 200000
  const alg = useMemo(() => logisticMap(r), [])
  const [i, setI] = useState(123456)
  const stored = useMemo(() => trace(alg, { x0: 0.2 }, n, { every: n, checkpointEvery: 1000 }), [alg])
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
    const from = Math.max(0, Math.min(i - 60, n - 120))
    const start = seek(alg, { x0: 0.2 }, from, { checkpoints: stored })
    // The map's state is x alone, so a trace started at x_from continues the orbit exactly.
    const local = trace(alg, { x0: start.x }, 120, { record: { x: (s) => s.x } })
    return { x: local.index.map((k) => k + from), y: Array.from(toFlat(local.series.x)) }
  }, [alg, i, stored])
  return (
    <Figure
      title="The logistic map around step i"
      controls={<Slider label="step i" value={i} min={0} max={n} step={1} onChange={setI} />}
      readouts={
        <>
          <Readout label="seek(i) from checkpoints" value={result.fromCheckpoints.x.toPrecision(12)} />
          <Readout label="run(i)" value={result.fromScratch.x.toPrecision(12)} />
          <Readout label="equal" value={String(result.fromCheckpoints.x === result.fromScratch.x)} />
          <Readout label="seek" value={`${result.seekMs.toFixed(3)} ms`} />
          <Readout label="run" value={`${result.runMs.toFixed(3)} ms`} />
          <Readout label="checkpoints" value={stored.checkpoints.index.length} />
        </>
      }
      caption="Drag the vertical line to move i within the window."
    >
      <XYChart
        xLabel="step"
        yLabel="x"
        yRange={[0, 1]}
        series={[{ name: 'x', type: 'line', x: orbit.x, y: orbit.y, slot: 0, showPoints: true }]}
        handles={[{ kind: 'x', at: i, label: 'i', onDrag: (v) => setI(Math.round(Math.min(Math.max(v, 0), n))) }]}
      />
    </Figure>
  )
}
