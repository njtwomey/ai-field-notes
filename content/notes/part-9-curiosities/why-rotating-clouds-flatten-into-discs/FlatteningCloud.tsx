import { useMemo, useState } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Player,
  Plot,
  Plots,
  Points,
  Readout,
  slider,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { simulateCloud, type CloudFrame } from './cloud'

const BOX = 2.5
/** Steps between kept frames. */
const EVERY = 2
const NEIGHBOURS = 8
const REFRESH = 4
const DISPERSION = 0.3

const norm = (v: readonly number[]) => Math.hypot(v[0], v[1], v[2])

/**
 * A rotating cloud of test particles around a central mass, with a torque-free pairwise drag between neighbours in
 * place of inelastic collisions. The run is computed once per setting and scrubbed with the player: top and edge-on
 * views, the energy, the axis ratio and the angular momentum in time.
 */
export function FlatteningCloud() {
  const state = useFigureState({
    spin: slider(0, 1, 0.8, { label: 'initial spin s (fraction of circular speed)', step: 0.05 }),
    dissipation: slider(0, 1, 0.3, { label: 'drag rate γ per pair', step: 0.05 }),
    n: int(400, { min: 50, max: 600, label: 'particles n', suggestions: [200, 300, 400, 500] }),
    steps: int(800, { min: 50, max: 4000, label: 'RK4 steps', suggestions: [400, 800, 1600] }),
    stepSize: float(0.025, { gt: 0, le: 0.1, label: 'step size h', suggestions: [0.0125, 0.025, 0.05] }),
    seed: int(1, { min: 0, max: 999, label: 'seed' }),
  })

  const run = useComputed(
    () => {
      const frames = simulateCloud({
        n: state.n,
        spin: state.spin,
        dissipation: state.dissipation,
        dispersion: DISPERSION,
        neighbours: NEIGHBOURS,
        stepSize: state.stepSize,
        steps: state.steps,
        every: EVERY,
        refresh: REFRESH,
        seed: state.seed,
      })
      const t = frames.map((f) => f.time)
      const energy = frames.map((f) => f.energy)
      const L0 = norm(frames[0].angularMomentum)
      const lRatio = frames.map((f) => norm(f.angularMomentum) / L0)
      const axis = frames.map((f) => f.axisRatio)
      const pos = frames.map((f) => split(f, state.n))
      return { frames, t, energy, L0, lRatio, axis, pos }
    },
    [state.n, state.spin, state.dissipation, state.stepSize, state.steps, state.seed],
    { mode: 'release' },
  )
  const r = run.value

  // The player restarts at step 0 whenever the run changes.
  const [position, setPosition] = useState({ r, k: 0 })
  const k = Math.min(position.r === r ? position.k : 0, r.frames.length - 1)
  const f = r.frames[k]
  const p = r.pos[k]

  const eRange = useMemo((): [number, number] => {
    const lo = Math.min(...r.energy)
    const hi = Math.max(...r.energy)
    const pad = 0.05 * (hi - lo || 1)
    return [lo - pad, hi + pad]
  }, [r])

  const xTop = useAxis({ label: 'x', range: [-BOX, BOX] })
  const yTop = useAxis({ label: 'y', range: [-BOX, BOX], equal: xTop })
  const xEdge = useAxis({ label: 'x', range: [-BOX, BOX] })
  const zEdge = useAxis({ label: 'z (along the spin axis)', range: [-BOX, BOX], equal: xEdge })
  const tE = useAxis({ label: 't', range: [0, r.t[r.t.length - 1]] })
  const yE = useAxis({ label: 'energy per particle E', range: eRange })
  const tR = useAxis({ label: 't', range: [0, r.t[r.t.length - 1]] })
  const yR = useAxis({ label: 'ratio', range: [0, 1.1] })

  return (
    <Figure
      title="A spinning cloud settles into a disc"
      defaultSize="L"
      state={state}
      controls={
        <Player
          value={k}
          onChange={(v) => setPosition({ r, k: v })}
          count={r.frames.length}
          label="time"
          format={(i) => `t = ${r.t[Math.min(i, r.t.length - 1)].toFixed(2)}`}
        />
      }
      readouts={{
        [`at t = ${f.time.toFixed(2)}`]: (
          <>
            <Readout label="energy per particle E" value={formatNumber(f.energy)} />
            <Readout label="axis ratio c/a" value={formatNumber(f.axisRatio)} />
            <Readout label="rms radius in the plane" value={formatNumber(f.radius)} />
            <Readout label="rms height" value={formatNumber(f.height)} />
          </>
        ),
        'angular momentum': (
          <>
            <Readout label="|L| per particle" value={formatNumber(norm(f.angularMomentum))} />
            <Readout label="relative change in |L| since t = 0" value={formatNumber(r.lRatio[k] - 1)} />
            <Readout label="net torque of the drag |Σ r × F| / n" value={formatNumber(f.dragTorque)} />
          </>
        ),
      }}
      caption={
        <>
          {state.n} test particles around a central mass (G = M = 1, softening 0.2) start in a shell between radii 0.5
          and 1.5, rotating about z at s times the circular speed, with random velocities of 0.3 times it. Each
          neighbour pair feels a drag along the line joining the two, at rate γ. Play to watch the vertical motion die
          first: the edge-on view (top right) thins to a sheet while the top view (top left) stays round. Bottom left:
          the energy per particle falls. Bottom right: the axis ratio c/a falls towards 0 and |L|/|L₀| stays at 1. The
          drag&apos;s net torque is zero to rounding; the small change in |L| is the RK4 truncation error, which falls
          as h⁴. Set s = 0 for the control: the ball shrinks into the centre. Set γ = 0 and nothing settles.
        </>
      }
    >
      <Plots rows={2} cols={2} heights={[3, 2]}>
        <Plot x={xTop} y={yTop}>
          <Points name="particles (top view)" x={p.x} y={p.y} slot={0} size={3} stale={run.stale} />
        </Plot>
        <Plot x={xEdge} y={zEdge}>
          <Points name="particles (edge-on)" x={p.x} y={p.z} slot={0} size={3} stale={run.stale} />
        </Plot>
        <Plot x={tE} y={yE}>
          <Curve name="E (whole run)" x={r.t} y={r.energy} muted />
          <Curve name="E" x={r.t.slice(0, k + 1)} y={r.energy.slice(0, k + 1)} slot={1} stale={run.stale} />
          <Points name="now" x={[f.time]} y={[f.energy]} emphasis live />
        </Plot>
        <Plot x={tR} y={yR}>
          <Curve name="c/a (whole run)" x={r.t} y={r.axis} muted />
          <Curve name="|L| / |L₀| (whole run)" x={r.t} y={r.lRatio} muted />
          <Curve name="axis ratio c/a" x={r.t.slice(0, k + 1)} y={r.axis.slice(0, k + 1)} slot={2} />
          <Curve name="|L| / |L₀|" x={r.t.slice(0, k + 1)} y={r.lRatio.slice(0, k + 1)} slot={0} />
          <Points name="now" x={[f.time, f.time]} y={[f.axisRatio, r.lRatio[k]]} emphasis live />
        </Plot>
      </Plots>
    </Figure>
  )
}

/** The x, y and z coordinates of a frame's particles. */
function split(f: CloudFrame, n: number) {
  const x = new Float64Array(n)
  const y = new Float64Array(n)
  const z = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    x[i] = f.positions[3 * i]
    y[i] = f.positions[3 * i + 1]
    z[i] = f.positions[3 * i + 2]
  }
  return { x, y, z }
}
