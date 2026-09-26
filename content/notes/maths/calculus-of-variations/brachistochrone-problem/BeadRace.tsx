import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Interactive,
  ParamButton,
  ParamSlider,
  Readout,
  XYChart,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'

const G = 9.81
/** Segments per curve for the time integral; the substitution below makes this accurate to about 1e-6 s. */
const STEPS = 600
/** Real seconds per simulated second when playing: a descent of 0.6 s plays over about 2.4 s. */
const SLOW = 4

type Vec = [number, number]
/**
 * A curve from the origin, y measured downwards, as pos(τ) for τ in [0, 1]. Near τ = 0 the time integrand
 * ds / √(2gy) is singular; sampling τ = w^m on a uniform w grid with the right m removes the singularity, so a plain
 * midpoint rule is accurate.
 */
type Curve = { name: string; slot: number; pos: (tau: number) => Vec; m: number }
type Trace = { points: Vec[]; times: number[]; total: number }

function trace(c: Curve): Trace {
  const points: Vec[] = [c.pos(0)]
  const times = [0]
  for (let k = 0; k < STEPS; k++) {
    const next = c.pos(((k + 1) / STEPS) ** c.m)
    const mid = c.pos(((k + 0.5) / STEPS) ** c.m)
    const prev = points[k]
    // Time on a segment is its length over the speed at its midpoint, v = √(2gy) from conservation of energy.
    const dt = Math.hypot(next[0] - prev[0], next[1] - prev[1]) / Math.sqrt(2 * G * Math.max(mid[1], 1e-12))
    points.push(next)
    times.push(times[k] + dt)
  }
  return { points, times, total: times[STEPS] }
}

/** Where the bead is at time t: interpolated along the trace, and parked at the end once it has arrived. */
function beadAt(tr: Trace, t: number): Vec {
  if (t >= tr.total) return tr.points[STEPS]
  let lo = 0
  let hi = STEPS
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1
    if (tr.times[mid] <= t) lo = mid
    else hi = mid
  }
  const f = (t - tr.times[lo]) / (tr.times[hi] - tr.times[lo])
  const [a, b] = [tr.points[lo], tr.points[hi]]
  return [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])]
}

/** The cycloid through (L, H): solve (θ − sin θ)/(1 − cos θ) = L/H for θ by bisection, then r = H/(1 − cos θ). */
function cycloidFit(L: number, H: number): { theta: number; r: number } {
  const ratio = (t: number) => (t - Math.sin(t)) / (1 - Math.cos(t))
  let lo = 1e-6
  let hi = 2 * Math.PI - 1e-6
  for (let i = 0; i < 80; i++) {
    const mid = (lo + hi) / 2
    if (ratio(mid) < L / H) lo = mid
    else hi = mid
  }
  const theta = (lo + hi) / 2
  return { theta, r: H / (1 - Math.cos(theta)) }
}

/** A race of frictionless beads from rest at the origin to (L, H) along four curves; the cycloid always wins. */
export function BeadRace() {
  const [end, setEnd] = useState<Vec>([1, 0.5])
  const p = useParam(0.5, { min: 0.3, max: 1.5, step: 0.05 })
  const t = useParam(0, { min: 0, max: 2.5, step: 0.01 })
  const [playing, setPlaying] = useState(false)
  const [L, H] = end

  const r = useMemo(() => {
    const { theta, r } = cycloidFit(L, H)
    // The circle through both ends that leaves the origin vertically: centre (R, 0), R = (L² + H²)/2L.
    const R = (L * L + H * H) / (2 * L)
    const phi = Math.atan2(H / R, (R - L) / R)
    const pw = p.value
    const curves: Curve[] = [
      { name: 'straight line', slot: 0, pos: (s) => [L * s, H * s], m: 2 },
      { name: 'circular arc', slot: 1, pos: (s) => [R - R * Math.cos(phi * s), R * Math.sin(phi * s)], m: 2 },
      {
        name: `power curve, p = ${pw.toFixed(2)}`,
        slot: 2,
        pos: (s) => [L * s, H * s ** pw],
        m: pw <= 1 ? 2 / pw : 2 / (2 - pw),
      },
      // Along the cycloid the integrand is exactly √(r/g), so no substitution is needed.
      {
        name: 'cycloid',
        slot: 3,
        pos: (s) => [r * (theta * s - Math.sin(theta * s)), r * (1 - Math.cos(theta * s))],
        m: 1,
      },
    ]
    const traces = curves.map(trace)
    const depth = Math.max(...traces.flatMap((tr) => tr.points.map((q) => q[1])))
    return { curves, traces, theta, radius: r, depth, cycloidTime: theta * Math.sqrt(r / G) }
  }, [L, H, p.value])

  const longest = Math.max(...r.traces.map((tr) => tr.total))

  // Play at 1/SLOW speed from the current time until every bead has arrived.
  const setTime = useRef(t.set)
  setTime.current = t.set
  useEffect(() => {
    if (!playing) return
    let frame = 0
    const start = performance.now()
    const from = t.value >= longest ? 0 : t.value
    const tick = (now: number) => {
      const next = from + (now - start) / 1000 / SLOW
      setTime.current(Math.min(next, longest))
      if (next < longest) frame = requestAnimationFrame(tick)
      else setPlaying(false)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
    // Restart only when play is pressed or the race changes; the time itself is driven by the loop.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, longest])

  // Plot with y pointing up, so the curves descend on the page.
  const series: XYSeries[] = [
    ...r.curves.map((c, i): XYSeries => ({
      name: c.name,
      type: 'line',
      x: r.traces[i].points.map((q) => q[0]),
      y: r.traces[i].points.map((q) => -q[1]),
      slot: c.slot,
    })),
    ...r.curves.map((c, i): XYSeries => {
      const [bx, by] = beadAt(r.traces[i], t.value)
      return { name: c.name, type: 'scatter', x: [bx], y: [-by], slot: c.slot }
    }),
  ]

  const handles: Handle[] = [
    {
      kind: 'point',
      at: [L, -H],
      label: 'end',
      // Keep L/H ≤ 4 so the cycloid, which dips below the end point for flat ends, stays in view.
      onDrag: ([x, y]) => {
        const nx = Math.min(2, Math.max(0.3, x))
        setEnd([nx, Math.min(1.2, Math.max(0.2, nx / 4, -y))])
        setPlaying(false)
      },
    },
  ]

  const fmt = (s: number) => `${s.toFixed(3)} s`
  const pad = 0.05
  return (
    <Interactive
      title="The race to the bottom"
      caption="Four frictionless beads start from rest at the origin and slide to the same end point: along the straight line, a circular arc that leaves vertically, a power curve y = H(x/L)^p, and the cycloid. The cycloid is always fastest. It drops steeply at first, so its bead builds speed early, and that beats the shorter but shallower paths. Press Play (quarter speed) or step the clock with the arrows; drag the end point or change p."
      controls={
        <>
          <ParamSlider label="time t (s)" param={t} format={(v) => `${v.toFixed(2)} s`} withArrows />
          <ParamSlider label="power-curve exponent p" param={p} />
          <ParamButton onClick={() => setPlaying((v) => !v)}>{playing ? 'Pause' : 'Play'}</ParamButton>
        </>
      }
      readout={
        <>
          {r.curves.map((c, i) => (
            <Readout key={c.slot} label={c.name} value={fmt(r.traces[i].total)} />
          ))}
          <Readout label="cycloid θ_end √(r/g)" value={fmt(r.cycloidTime)} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-xl">
        <XYChart
          series={series}
          xLabel="x (m)"
          yLabel="height (m)"
          xRange={[-pad, L + pad]}
          yRange={[-(r.depth + pad), pad]}
          equalAspect
          handles={handles}
        />
      </div>
    </Interactive>
  )
}
