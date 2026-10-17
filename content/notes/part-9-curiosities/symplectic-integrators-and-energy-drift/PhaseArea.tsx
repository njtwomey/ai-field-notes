import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Player,
  Plot,
  Readout,
  Shapes,
  slider,
  useAxis,
  useComputed,
  useFigureState,
  usePlayhead,
} from 'aifn-render'
import { flowSet, polygonArea, type AreaMethod } from './physics'

const RADIUS = 0.3
const POINTS = 800
const METHODS: { method: AreaMethod; name: string; slot: number }[] = [
  { method: 'rk4', name: 'RK4', slot: 0 },
  { method: 'leapfrog', name: 'leapfrog', slot: 1 },
  { method: 'euler', name: 'explicit Euler', slot: 2 },
]
/** Energy levels of the pendulum drawn behind the sets: H = p²/2 − cos q. */
const LEVELS = [-0.5, 0, 0.5, 1, 1.5, 2.5]

/** The level set H = c as curves over q ∈ [−π, π]: the upper and lower branches p = ±√(2(c + cos q)). */
function levelCurve(c: number) {
  const q = Array.from({ length: 241 }, (_, i) => -Math.PI + (2 * Math.PI * i) / 240)
  const branch = (sign: number) => q.map((v) => (c + Math.cos(v) >= 0 ? sign * Math.sqrt(2 * (c + Math.cos(v))) : NaN))
  return { x: [...q, NaN, ...q], y: [...branch(1), NaN, ...branch(-1)] }
}

export function PhaseArea() {
  const state = useFigureState({
    q0: slider(-2.5, 2.5, 0, { onChart: true, label: 'centre q' }),
    p0: slider(-2, 2, 1.2, { onChart: true, label: 'centre p' }),
    h: float(0.3, { gt: 0, le: 1, label: 'step size h', suggestions: [0.1, 0.2, 0.3, 0.5] }),
    steps: int(20, { ge: 1, le: 200, label: 'steps', suggestions: [10, 20, 50, 100] }),
  })
  const { q0, p0, h, steps } = state

  const computed = useComputed(() => {
    const starts = Array.from({ length: POINTS }, (_, j) => {
      const a = (2 * Math.PI * j) / POINTS
      return [q0 + RADIUS * Math.cos(a), p0 + RADIUS * Math.sin(a)] as const
    })
    return METHODS.map((m) => ({ ...m, sets: flowSet(m.method, starts, h, steps) }))
  }, [q0, p0, h, steps])
  const runs = computed.value
  const [k, setK] = usePlayhead(steps + 1)
  const levels = useMemo(() => LEVELS.map(levelCurve), [])
  const area0 = Math.PI * RADIUS * RADIUS

  const qa = useAxis({ label: 'angle q', range: [-Math.PI, Math.PI] })
  const pa = useAxis({ label: 'momentum p', range: [-2.6, 2.6], equal: qa })

  return (
    <Figure
      title="Phase-space area under three integrators"
      state={state}
      defaultSize="L"
      controls={
        <Player
          value={k}
          onChange={setK}
          count={steps + 1}
          label="step"
          format={(i) => `step ${i}, t = ${formatNumber(i * h)}`}
        />
      }
      readouts={{
        'area / initial area': runs.map((r) => {
          const set = r.sets[Math.min(k, r.sets.length - 1)]
          return <Readout key={r.name} label={r.name} value={(polygonArea(set) / area0).toFixed(4)} />
        }),
      }}
      caption="A disc of 800 starting states of the pendulum H = p²/2 − cos q, each moved by the same integrator with step h. Drag the disc's centre to start elsewhere; play to take the steps. The grey curves are levels of the exact energy. Explicit Euler stretches the set's area by about 1 + h² cos q per step, so the set grows where cos q > 0. Leapfrog keeps the area exactly; the set shears into a thin arc but its area stays that of the disc. RK4 changes the area by a tiny amount per step, because it is not symplectic. The readouts compute each area from the polygon through the 800 points, which loses a little when the arc becomes very thin."
    >
      <Plot x={qa} y={pa}>
        {levels.map((l, i) => (
          <Curve key={i} name="energy levels" x={l.x} y={l.y} muted width={0.5} silent />
        ))}
        {runs.map((r) => {
          const set = r.sets[Math.min(k, r.sets.length - 1)]
          return (
            <Shapes
              key={r.name}
              name={r.name}
              shapes={[{ contours: [set], tone: r.slot, opacity: 0.25 }]}
              stale={computed.stale}
            />
          )
        })}
        {runs.map((r) => {
          const set = r.sets[Math.min(k, r.sets.length - 1)]
          return (
            <Curve
              key={r.name}
              name={r.name}
              x={[...set.map((s) => s[0]), set[0][0]]}
              y={[...set.map((s) => s[1]), set[0][1]]}
              slot={r.slot}
              width={1}
              silent
            />
          )
        })}
        <Handle
          kind="point"
          at={[q0, p0]}
          label="centre"
          onDrag={([q, p]) => {
            state.set('q0', q)
            state.set('p0', p)
          }}
        />
      </Plot>
    </Figure>
  )
}
