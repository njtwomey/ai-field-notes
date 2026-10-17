import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Player,
  Plot,
  Points,
  Readout,
  useAxis,
  useComputed,
  useFigureState,
  usePlayhead,
} from 'aifn-render'
import { ORBITS, runOrbit, type OrbitName } from './orbits'

const NAMES = Object.keys(ORBITS) as OrbitName[]
const fmtE = (v: number) => (v > 0 ? v.toExponential(1) : '0')

export function KickedOrbits() {
  const state = useFigureState({
    orbit: choice(
      NAMES.map((value) => ({ value, label: ORBITS[value].label })),
      'eight',
      { label: 'periodic solution' },
    ),
    kick: float(1e-3, {
      min: 0,
      max: 0.1,
      label: 'kick δ to the x-velocity of body 1',
      suggestions: [0, 1e-6, 1e-3, 1e-2],
    }),
    periods: int(10, { min: 1, max: 200, label: 'periods to run', suggestions: [3, 10, 50] }),
    perPeriod: int(1000, { min: 50, max: 10000, label: 'leapfrog steps per period', suggestions: [200, 1000, 5000] }),
  })
  const { orbit, kick, periods, perPeriod } = state

  const sim = useComputed(() => runOrbit(orbit, kick, periods, perPeriod), [orbit, kick, periods, perPeriod], {
    mode: 'release',
  })
  const run = sim.value

  const [frame, setFrame] = usePlayhead(run.q.length)
  const k = Math.min(frame, run.q.length - 1)

  const trails = useMemo(() => {
    // Each body's path up to every frame is a prefix of its whole path, so build the whole paths once.
    return [0, 1, 2].map((b) => ({ x: run.q.map((s) => s[2 * b]), y: run.q.map((s) => s[2 * b + 1]) }))
  }, [run])
  // Draw at most the last 300 frames (three periods) of each trail.
  const from = Math.max(0, k - 300)

  let worst = 0
  for (let j = 0; j <= k; j++) worst = Math.max(worst, run.shapeDistance[j])
  const logT: number[] = []
  const logShape: number[] = []
  const logErr: number[] = []
  const logErrT: number[] = []
  run.periods.forEach((t, j) => {
    if (run.shapeDistance[j] > 0) {
      logT.push(t)
      logShape.push(run.shapeDistance[j])
    }
    if (run.energyError[j] > 0) {
      logErrT.push(t)
      logErr.push(run.energyError[j])
    }
  })

  const xa = useAxis({ label: 'x', range: [-1.6, 1.6] })
  const ya = useAxis({ label: 'y', range: [-1.2, 1.2], equal: xa })
  const ta = useAxis({ label: 'time in periods', range: [0, periods] })
  const la = useAxis({ label: 'size (log scale)', range: [1e-12, 1e2], log: true })

  return (
    <Figure
      title="Three periodic orbits, kicked"
      state={state}
      defaultSize="L"
      caption="Three equal masses (G = 1) on one of three periodic solutions, integrated by leapfrog after body 1's x-velocity is changed by δ. Left: the unperturbed orbit over one period (grey, dashed) and the last three periods of the kicked bodies. Right, on a log scale: how far the kicked triangle's shape (its three side lengths) is from every shape the unperturbed orbit passes through (ink), and the relative energy error (grey). The figure-eight's shape error stays at the size of the kick. Lagrange's triangle and Euler's line amplify any kick, even the rounding error at δ = 0 for the triangle, until the bodies fly apart; the leapfrog's fixed step then fails in the close encounters, which shows as a jump in the energy error. Drag the time line on the right to scrub."
      controls={
        <Player
          value={frame}
          onChange={setFrame}
          count={run.q.length}
          label="time"
          format={(j) => `${formatNumber(run.periods[Math.min(j, run.periods.length - 1)])} periods`}
        />
      }
      readouts={
        <>
          <Readout label="time (periods)" value={formatNumber(run.periods[k])} />
          <Readout label="shape distance now" value={fmtE(run.shapeDistance[k])} />
          <Readout label="largest shape distance so far" value={fmtE(worst)} />
          <Readout label="energy error |E/E₀ − 1|" value={fmtE(run.energyError[k])} />
          <Readout label="angular momentum L" value={formatNumber(run.angularMomentum)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-[3fr_2fr]">
        <Plot x={xa} y={ya} height={440}>
          {run.reference.map((r, b) => (
            <Curve key={`r${b}`} name="unperturbed, one period" x={r.x} y={r.y} muted dashed thin />
          ))}
          {trails.map((t, b) => (
            <Curve
              key={`t${b}`}
              id={`t${b}`}
              name={`body ${b + 1}`}
              x={t.x.slice(from, k + 1)}
              y={t.y.slice(from, k + 1)}
              slot={b}
              thin
              live
            />
          ))}
          {trails.map((t, b) => (
            <Points
              key={`p${b}`}
              id={`p${b}`}
              name={`body ${b + 1}`}
              x={[t.x[k]]}
              y={[t.y[k]]}
              slot={b}
              size={11}
              live
            />
          ))}
        </Plot>
        <Plot x={ta} y={la} height={440}>
          <Curve name="shape distance" x={logT} y={logShape} emphasis />
          <Curve name="energy error" x={logErrT} y={logErr} muted dashed />
          <Handle
            kind="x"
            at={run.periods[k]}
            label="t"
            onDrag={(t) =>
              setFrame(Math.min(run.q.length - 1, Math.max(0, Math.round((t / periods) * (run.q.length - 1)))))
            }
          />
        </Plot>
      </div>
    </Figure>
  )
}
