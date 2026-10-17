import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  Player,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useComputed,
  useFigureState,
  usePlayhead,
} from 'aifn-render'
import { frameSteps, MASSES, pathSteps, runPythagorean, START } from './pythagorean'

/** Frame spacing in time units: 80 time units play in about 27 s at 60 frames per second. */
const DT = 0.05
const SIZES = [9, 11, 13]

const fmtE = (v: number) => (v > 0 ? v.toExponential(1) : '0')

export function PythagoreanProblem() {
  const state = useFigureState({
    tEnd: float(80, { min: 1, max: 150, label: 'end time t', suggestions: [20, 80, 120] }),
    rtol: float(1e-11, {
      min: 1e-14,
      max: 1e-4,
      label: 'relative tolerance',
      suggestions: [1e-6, 1e-8, 1e-10, 1e-11, 1e-12],
    }),
    delta: float(1e-6, { gt: 0, max: 0.1, label: 'offset δ of copy B', suggestions: [1e-9, 1e-6, 1e-3] }),
    softening: float(0, { min: 0, max: 0.5, label: 'softening length ε', suggestions: [0, 0.01, 0.1] }),
    x3: slider(-4, 4, START[0], { step: 0.01, onChart: true, label: 'start x of mass 3' }),
    y3: slider(-4, 4, START[1], { step: 0.01, onChart: true, label: 'start y of mass 3' }),
    x4: slider(-4, 4, START[2], { step: 0.01, onChart: true, label: 'start x of mass 4' }),
    y4: slider(-4, 4, START[3], { step: 0.01, onChart: true, label: 'start y of mass 4' }),
    x5: slider(-4, 4, START[4], { step: 0.01, onChart: true, label: 'start x of mass 5' }),
    y5: slider(-4, 4, START[5], { step: 0.01, onChart: true, label: 'start y of mass 5' }),
  })
  const { tEnd, rtol, delta, softening, x3, y3, x4, y4, x5, y5 } = state

  const sim = useComputed(
    () => runPythagorean({ tEnd, rtol, delta, softening, start: [x3, y3, x4, y4, x5, y5] }),
    [tEnd, rtol, delta, softening, x3, y3, x4, y4, x5, y5],
    { mode: 'release' },
  )
  const run = sim.value

  const view = useMemo(() => {
    const frames = frameSteps(run.time, DT)
    const pathA = pathSteps(run.qa, 0.03)
    const pathB = pathSteps(run.qb, 0.05)
    // One point per frame for the log panel: separation at the frame's step, worst energy error since the last frame.
    const ft: number[] = []
    const fsep: number[] = []
    const ferr: number[] = []
    let from = 0
    frames.forEach((i, k) => {
      let worst = 0
      for (let j = from; j <= i; j++) worst = Math.max(worst, run.energyError[j])
      from = i + 1
      ft.push(k * DT)
      fsep.push(run.separation[i])
      ferr.push(worst)
    })
    return { frames, pathA, pathB, ft, fsep, ferr }
  }, [run])

  const [frame, setFrame] = usePlayhead(view.frames.length)
  const step = view.frames[Math.min(frame, view.frames.length - 1)]
  const now = frame * DT

  const trail = (path: number[], q: number[][], b: number) => {
    const xs: number[] = []
    const ys: number[] = []
    for (const i of path) {
      if (i > step) break
      xs.push(q[i][2 * b])
      ys.push(q[i][2 * b + 1])
    }
    xs.push(q[step][2 * b])
    ys.push(q[step][2 * b + 1])
    return { x: xs, y: ys }
  }

  let closest = Infinity
  let closestAt = 0
  let worstE = 0
  for (let j = 0; j <= step; j++) {
    if (run.closest[j] < closest) {
      closest = run.closest[j]
      closestAt = run.time[j]
    }
    worstE = Math.max(worstE, run.energyError[j])
  }

  // The log panel cannot show zeros: the energy error is exactly 0 at t = 0.
  const errShown = view.ft.flatMap((t, k) => (view.ferr[k] > 0 ? [[t, view.ferr[k]]] : []))
  const sepShown = view.ft.flatMap((t, k) => (view.fsep[k] > 0 ? [[t, view.fsep[k]]] : []))

  const xa = useAxis({ label: 'x', range: [-5, 5] })
  const ya = useAxis({ label: 'y', range: [-5, 5], equal: xa })
  const ta = useAxis({ label: 'time t', range: [0, tEnd] })
  const la = useAxis({ label: 'size (log scale)', range: [1e-14, 1e2], log: true })

  const starts: [string, string][] = [
    ['x3', 'y3'],
    ['x4', 'y4'],
    ['x5', 'y5'],
  ]

  return (
    <Figure
      title="Burrau's Pythagorean problem and two copies 10⁻⁶ apart"
      state={state}
      defaultSize="L"
      caption={`Masses 3, 4 and 5 start at rest on a 3-4-5 triangle (G = 1). Solid lines are copy A; dashed lines are copy B, identical except that mass 3 starts δ further along x. Both are integrated as one system by the Dormand–Prince 5(4) method, so they share every step. Right, on a log scale: the distance between the copies (ink) and the relative energy error of copy A (grey). The distance jumps at each close encounter and reaches the size of the system near t = 57; the energy error stays near the tolerance. At step 0, drag a body to change its start. Drag the time line on the right to scrub. Zoom out on the left to follow the ejection.${run.failure ? ` The run ${run.failure}.` : ''}`}
      controls={
        <Player
          value={frame}
          onChange={setFrame}
          count={view.frames.length}
          label="time"
          format={(k) => `t = ${(k * DT).toFixed(2)}`}
        />
      }
      readouts={
        <>
          <Readout label="time t" value={now.toFixed(2)} />
          <Readout label="energy error |E/E₀ − 1| now" value={fmtE(run.energyError[step])} />
          <Readout label="largest energy error so far" value={fmtE(worstE)} />
          <Readout label="distance between copies" value={fmtE(run.separation[step])} />
          <Readout
            label="closest approach so far"
            value={`${closest.toExponential(2)} at t = ${closestAt.toFixed(3)}`}
          />
          <Readout
            label="accepted steps (both copies)"
            value={`${run.time.length} in ${formatNumber(sim.ms / 1000)} s`}
          />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-[3fr_2fr]">
        <Plot x={xa} y={ya} height={460}>
          {MASSES.map((m, b) => {
            const a = trail(view.pathA, run.qa, b)
            return <Curve key={`a${m}`} id={`a${m}`} name={`mass ${m}`} x={a.x} y={a.y} slot={b} thin live />
          })}
          {MASSES.map((m, b) => {
            const t = trail(view.pathB, run.qb, b)
            return (
              <Curve key={`b${m}`} id={`b${m}`} name={`mass ${m}, copy B`} x={t.x} y={t.y} slot={b} thin dashed live />
            )
          })}
          {MASSES.map((m, b) => (
            <Points
              key={`p${m}`}
              id={`p${m}`}
              name={`mass ${m}`}
              x={[run.qa[step][2 * b]]}
              y={[run.qa[step][2 * b + 1]]}
              slot={b}
              size={SIZES[b]}
              live
            />
          ))}
          {frame === 0 &&
            starts.map(([kx, ky], b) => (
              <Handle key={kx} {...state.handle([kx, ky], { label: `start of mass ${MASSES[b]}` })} />
            ))}
        </Plot>
        <Plot x={ta} y={la} height={460}>
          <Curve name="distance between copies" x={sepShown.map((p) => p[0])} y={sepShown.map((p) => p[1])} emphasis />
          <Curve
            name="energy error of copy A"
            x={errShown.map((p) => p[0])}
            y={errShown.map((p) => p[1])}
            muted
            dashed
          />
          <Handle
            kind="x"
            at={now}
            label="t"
            onDrag={(t) => setFrame(Math.min(view.frames.length - 1, Math.round(Math.max(0, t) / DT)))}
          />
        </Plot>
      </div>
    </Figure>
  )
}
