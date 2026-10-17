import { useMemo } from 'react'
import {
  Curve,
  Figure,
  formatNumber,
  Handle,
  Player,
  Plot,
  Plots,
  Points,
  Readout,
  Segments,
  slider,
  useAxis,
  useFigureState,
  usePlayhead,
} from 'aifn-render'
import { meanFromEccentric, newtonIterates, stepsToConverge } from './anomalies'

const fmt = (v: number) => formatNumber(v)
const TAU = 2 * Math.PI
/** Errors below this are drawn at it: a log axis cannot show zero. */
const FLOOR = 1e-16
/** Iteration counts are capped here on the sweep over M. */
const CAP = 100
const GRID = 600
const STARTS = [
  { key: 'M', label: 'E₀ = M', slot: 0, start: (M: number) => M },
  { key: 'pi', label: 'E₀ = π', slot: 1, start: () => Math.PI },
] as const

export function NewtonOnKepler() {
  const state = useFigureState({
    e: slider(0, 0.99, 0.99, { label: 'eccentricity e', step: 0.001 }),
    M: slider(0, TAU, 0.1, { onChart: true, label: 'mean anomaly M', step: 0.001 }),
  })
  const { e, M } = state

  const runs = useMemo(
    () =>
      STARTS.map((s) => {
        const E = newtonIterates(e, M, s.start(M))
        return { ...s, E, converged: stepsToConverge(e, M, s.start(M), CAP) }
      }),
    [e, M],
  )
  // The converged root, from the run that converged (E₀ = π always does for e < 1).
  const root = runs[1].E[runs[1].E.length - 1]
  const count = Math.max(...runs.map((r) => r.E.length))
  const [k, setK] = usePlayhead(count)

  const curve = useMemo(() => {
    const x = Array.from({ length: 801 }, (_, i) => -Math.PI + (4 * Math.PI * i) / 800)
    return { x, y: x.map((E) => meanFromEccentric(e, E)) }
  }, [e])

  const sweep = useMemo(() => {
    const Ms = Array.from({ length: GRID }, (_, i) => (TAU * (i + 0.5)) / GRID)
    return {
      M: Ms,
      counts: STARTS.map((s) => Ms.map((m) => Math.max(1, stepsToConverge(e, m, s.start(m), CAP)))),
    }
  }, [e])

  const shown = runs.map((r) => {
    const upto = Math.min(k, r.E.length - 1)
    const segments: { from: [number, number]; to: [number, number] }[] = []
    for (let i = 0; i < upto; i++) {
      const g = meanFromEccentric(e, r.E[i])
      segments.push({ from: [r.E[i], M], to: [r.E[i], g] })
      segments.push({ from: [r.E[i], g], to: [r.E[i + 1], M] })
    }
    const errors = r.E.slice(0, upto + 1).map((E) => Math.max(FLOOR, Math.abs(E - root)))
    return { ...r, upto, segments, errors }
  })

  const ex = useAxis({ label: 'eccentric anomaly E', range: [-Math.PI, 3 * Math.PI] })
  const ey = useAxis({ label: 'E − e sin E', range: [-Math.PI - 1, 3 * Math.PI + 1], equal: ex })
  const kx = useAxis({ label: 'Newton step k', range: [0, Math.max(10, count - 1)], integer: true })
  const ky = useAxis({ label: '|E_k − E*|', log: true, range: [FLOOR, 1e4] })
  const mx = useAxis({ label: 'mean anomaly M', range: [0, TAU] })
  const my = useAxis({ label: `steps to converge (capped at ${CAP})`, log: true, range: [1, CAP] })

  return (
    <Figure
      title="Newton's method on Kepler's equation"
      state={state}
      defaultSize="XL"
      controls={<Player value={k} onChange={setK} count={count} label="Newton step" format={(i) => `k = ${i}`} />}
      readouts={{
        ...Object.fromEntries(
          shown.map((r) => [
            r.label,
            <>
              <Readout label="E_k" value={fmt(r.E[r.upto])} />
              <Readout label="|E_k − E*|" value={fmt(Math.abs(r.E[r.upto] - root))} />
              <Readout label="steps to converge" value={r.converged >= CAP ? `more than ${CAP}` : r.converged} />
            </>,
          ]),
        ),
        root: <Readout label="E* (radians)" value={fmt(root)} />,
      }}
      caption="Left: the curve y = E − e sin E and the level y = M (drag the line to change M). Each Newton step runs up the vertical to the curve, then down its tangent to the level, from the start E₀ = M and from E₀ = π. Play to take the steps one at a time. Middle: the error of each iterate on a log scale; once close, the number of correct digits doubles every step. Right: the number of steps for every M at this eccentricity (drag the vertical line to choose M). At high eccentricity the curve is nearly flat near E = 0, so the first tangent from E₀ = M overshoots by many radians and the iterates wander before they settle; from E₀ = π the iterates fall monotonically onto the root."
    >
      <Plots cols={3} widths={[1.3, 1, 1]}>
        <Plot x={ex} y={ey}>
          <Curve name="E − e sin E" x={curve.x} y={curve.y} emphasis />
          {shown.map((r) => (
            <Segments key={r.key} name={r.label} segments={r.segments} slot={r.slot} width={1.5} />
          ))}
          {shown.map((r) => (
            <Points
              key={r.key}
              name={r.label}
              x={r.E.slice(0, r.upto + 1)}
              y={r.E.slice(0, r.upto + 1).map(() => M)}
              slot={r.slot}
              size={6}
            />
          ))}
          <Points name="root E*" x={[root]} y={[M]} emphasis size={9} />
          <Handle kind="y" at={M} label="M" onDrag={(v) => state.set('M', Math.min(TAU, Math.max(0, v)))} />
        </Plot>
        <Plot x={kx} y={ky}>
          {shown.map((r) => (
            <Curve key={r.key} name={r.label} x={r.errors.map((_, i) => i)} y={r.errors} slot={r.slot} showPoints />
          ))}
        </Plot>
        <Plot x={mx} y={my}>
          {STARTS.map((s, i) => (
            <Curve key={s.key} name={s.label} x={sweep.M} y={sweep.counts[i]} slot={s.slot} />
          ))}
          <Handle kind="x" at={M} label="M" onDrag={(v) => state.set('M', Math.min(TAU, Math.max(0, v)))} />
        </Plot>
      </Plots>
    </Figure>
  )
}
