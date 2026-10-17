import { useMemo } from 'react'
import { rungeKutta } from 'aifn-compute/dynamics/ode'
import { toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import {
  Curve,
  Dashboard,
  DashboardCell,
  DashboardRow,
  Figure,
  float,
  Handle,
  int,
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
import { MAX_GROWTH, growthRate, initialPair, measuredRate, pairRhs, separation } from './springs'

// Units: Ω = 1, so one orbit takes 2π; lengths in units of the initial separation δ.
const DELTA = 1
const FRAMES = 200
const K_MAX = 4
const K_GRID = Array.from({ length: 401 }, (_, i) => (K_MAX * i) / 400)
const RATE_CURVE = K_GRID.map((k) => growthRate(k))

export function SpringInstability() {
  const state = useFigureState({
    k: slider(0, K_MAX, 0.5, { label: 'spring constant k / Ω²', step: 0.01, onChart: true }),
    orbits: int(4, { label: 'orbits to integrate', ge: 1, le: 12, suggestions: [2, 4, 6, 8] }),
    stepSize: float(0.01, { label: 'RK4 step Ωh', ge: 0.002, le: 0.2, suggestions: [0.005, 0.01, 0.02, 0.05] }),
  })
  const { k, orbits, stepSize } = state
  const [frame, setFrame] = usePlayhead(FRAMES + 1)

  // One RK4 run of the two masses in Hill's frame, recorded at every step.
  const run = useMemo(() => {
    const tEnd = 2 * Math.PI * orbits
    const steps = Math.ceil(tEnd / stepSize) + 2
    const tr = trace(rungeKutta(pairRhs(k), 'rk4', { stepSize, tEnd }), { x0: initialPair(DELTA), t0: 0 }, steps, {
      record: { state: (s) => toFlat(s.x), time: (s) => s.time },
    })
    const rows = toRows(tr.series.state)
    const time = toFlat(tr.series.time)
    const sep = rows.map(separation)
    return { rows, time, sep, tEnd, rate: measuredRate(time, sep) }
  }, [k, orbits, stepSize])

  const theory = growthRate(k)
  const last = run.rows.length - 1
  const at = Math.round((frame / FRAMES) * last)
  const now = run.rows[at]

  // Trails up to the current step, thinned to at most about 600 points per mass.
  const trails = useMemo(() => {
    const stride = Math.max(1, Math.ceil(at / 600))
    const idx: number[] = []
    for (let i = 0; i < at; i += stride) idx.push(i)
    idx.push(at)
    const pick = (c: number) => idx.map((i) => run.rows[i][c])
    return { x1: pick(0), y1: pick(1), x2: pick(2), y2: pick(3) }
  }, [run, at])
  // A square window around the origin that holds both paths so far.
  const extent = useMemo(() => {
    let m = DELTA / 2
    for (const a of [trails.x1, trails.y1, trails.x2, trails.y2]) for (const v of a) m = Math.max(m, Math.abs(v))
    return 1.15 * m
  }, [trails])

  // The theoretical slope e^{st}, drawn through the separation at the end of the run.
  const reference = useMemo(() => {
    if (!(theory > 0)) return null
    const end = run.sep[last]
    const ts = [run.time[0], run.tEnd]
    return { x: ts, y: ts.map((t) => end * Math.exp(theory * (t - run.tEnd))) }
  }, [run, theory, last])

  // δℓ / R₀ = ẏ + 2Ωx: each mass's angular momentum relative to a circular orbit at R₀ (Hill's approximation).
  const momentum = (x: number, v: number) => v + 2 * x
  const inner = momentum(now[0], now[5]) - momentum(run.rows[0][0], run.rows[0][5])
  const outer = momentum(now[2], now[7]) - momentum(run.rows[0][2], run.rows[0][7])

  const xAxis = useAxis({ label: 'radial x / δ', range: [-extent, extent], nice: false })
  const yAxis = useAxis({ label: 'azimuthal y / δ', range: [-extent, extent], nice: false, equal: xAxis })
  const tAxis = useAxis({ label: 'time Ωt', range: [0, run.tEnd] })
  const sAxis = useAxis({ label: 'separation / δ', log: true })
  const kAxis = useAxis({ label: 'spring constant k / Ω²', range: [0, K_MAX] })
  const rAxis = useAxis({ label: 'growth rate s / Ω', range: [0, 0.8] })

  return (
    <Figure
      title="Two masses on a spring in orbit"
      purpose="In a Keplerian disc a weak spring between two masses on neighbouring orbits drives them apart exponentially: the spring moves angular momentum from the inner mass to the outer one."
      state={state}
      defaultSize="XL"
      controls={
        <Player
          value={frame}
          onChange={setFrame}
          count={FRAMES + 1}
          label="time"
          format={(i) => `Ωt = ${run.time[Math.round((i / FRAMES) * last)].toFixed(2)}`}
        />
      }
      readouts={{
        'growth rate': (
          <>
            <Readout label="dispersion relation s / Ω" value={theory.toFixed(4)} />
            <Readout label="measured from the run" value={run.rate.toFixed(4)} />
            <Readout label="growth factor per orbit, e^{2πs/Ω}" value={Math.exp(2 * Math.PI * theory).toPrecision(3)} />
          </>
        ),
        now: (
          <>
            <Readout label="Ωt" value={run.time[at].toFixed(2)} />
            <Readout label="separation / δ" value={run.sep[at].toPrecision(3)} />
            <Readout label="δℓ inner mass / (R₀Ωδ)" value={inner.toPrecision(3)} />
            <Readout label="δℓ outer mass / (R₀Ωδ)" value={outer.toPrecision(3)} />
          </>
        ),
      }}
      caption="Left: the two masses in the frame rotating at Ω, x outward and y along the orbit, with their paths so far and the spring between them; the axes rescale to follow the masses. Top right: the separation on a log axis, with the dispersion relation's e^{st} dashed through the final point; a straight line means exponential growth. Bottom right: the growth rate from the dispersion relation against k, with the rate measured from the run as a point and the maximum (3/4)Ω at k = (15/16)Ω² marked. Drag the vertical k line to change the spring. The inner mass loses angular momentum and the outer mass gains the same amount."
    >
      <Dashboard>
        <DashboardRow minHeight={360}>
          <DashboardCell aspect="square">
            <Plot x={xAxis} y={yAxis}>
              <Curve name="inner mass path" x={trails.x1} y={trails.y1} slot={0} thin />
              <Curve name="outer mass path" x={trails.x2} y={trails.y2} slot={1} thin />
              <Segments name="spring" segments={[{ from: [now[0], now[1]], to: [now[2], now[3]] }]} muted />
              <Points name="inner mass" x={[now[0]]} y={[now[1]]} slot={0} live />
              <Points name="outer mass" x={[now[2]]} y={[now[3]]} slot={1} live />
            </Plot>
          </DashboardCell>
          <DashboardCell>
            <Plots rows={2}>
              <Plot x={tAxis} y={sAxis}>
                <Curve name="separation" x={run.time} y={run.sep} slot={2} />
                {reference && <Curve name="e^{st}, dispersion relation" x={reference.x} y={reference.y} muted dashed />}
                <Points name="now" x={[run.time[at]]} y={[run.sep[at]]} emphasis live />
              </Plot>
              <Plot x={kAxis} y={rAxis}>
                <Curve name="dispersion relation" x={K_GRID} y={RATE_CURVE} slot={2} />
                <Curve name="k = 3Ω², edge of instability" x={[3, 3]} y={[0, 0.8]} muted dashed />
                <Points name="maximum" x={[MAX_GROWTH.k]} y={[MAX_GROWTH.rate]} muted />
                <Points name="measured" x={[k]} y={[Math.max(0, run.rate)]} emphasis live />
                <Handle {...state.handle('k', { label: 'k' })} />
              </Plot>
            </Plots>
          </DashboardCell>
        </DashboardRow>
      </Dashboard>
    </Figure>
  )
}
