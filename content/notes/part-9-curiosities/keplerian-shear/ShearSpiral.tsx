import { useMemo, useState } from 'react'
import {
  Curve,
  Figure,
  formatNumber,
  int,
  Player,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { omega, pitchAngle, windTracers } from './shear'

const R_IN = 1
const R_OUT = 3
const TRACERS = 31
const FRAMES = 400
const BOX = 3.3

const tracerRadii = Array.from({ length: TRACERS }, (_, i) => R_IN + ((R_OUT - R_IN) * i) / (TRACERS - 1))
const lineRadii = Array.from({ length: 301 }, (_, i) => R_IN + ((R_OUT - R_IN) * i) / 300)
const circle = (r: number) => {
  const a = Array.from({ length: 181 }, (_, i) => (2 * Math.PI * i) / 180)
  return { x: a.map((v) => r * Math.cos(v)), y: a.map((v) => r * Math.sin(v)) }
}
const INNER = circle(R_IN)
const OUTER = circle(R_OUT)

/**
 * Tracers that start on one radial line, each on its own circular orbit with Ω(r) = r^(−q): the line winds into a
 * trailing spiral. Closed form, scrubbed in time with the player.
 */
export function ShearSpiral() {
  const state = useFigureState({
    q: slider(0, 2, 1.5, { label: 'shear exponent q in Ω ∝ r^(−q)', step: 0.05 }),
    orbits: int(4, { min: 1, max: 20, label: 'orbits of the inner edge', suggestions: [1, 4, 10] }),
  })
  const tEnd = 2 * Math.PI * state.orbits
  const run = useMemo(() => ({ q: state.q, tEnd }), [state.q, tEnd])
  const [position, setPosition] = useState({ run, k: 0 })
  const k = position.run === run ? position.k : 0
  const t = (tEnd * k) / FRAMES

  const tracers = windTracers(tracerRadii, state.q, t)
  const line = windTracers(lineRadii, state.q, t)
  const windings = ((omega(R_IN, state.q) - omega(R_OUT, state.q)) * t) / (2 * Math.PI)

  const xa = useAxis({ label: 'x', range: [-BOX, BOX] })
  const ya = useAxis({ label: 'y', range: [-BOX, BOX], equal: xa })
  return (
    <Figure
      title="Shear winds a line into a spiral"
      state={state}
      aspect={1}
      controls={
        <Player
          value={k}
          onChange={(v) => setPosition({ run, k: v })}
          count={FRAMES + 1}
          label="time"
          format={(i) => `t = ${((tEnd * i) / FRAMES).toFixed(2)}`}
        />
      }
      readouts={{
        [`at t = ${t.toFixed(2)}`]: (
          <>
            <Readout label="orbits at r = 1" value={formatNumber((omega(R_IN, state.q) * t) / (2 * Math.PI))} />
            <Readout label="orbits at r = 3" value={formatNumber((omega(R_OUT, state.q) * t) / (2 * Math.PI))} />
            <Readout label="turns of the spiral, (Ω(1) − Ω(3)) t / 2π" value={formatNumber(windings)} />
            <Readout label="pitch angle at r = 2" value={`${pitchAngle(2, state.q, t).toFixed(1)}°`} />
          </>
        ),
        'shear at r = 2': (
          <>
            <Readout label="Ω(2)" value={formatNumber(omega(2, state.q))} />
            <Readout label="r dΩ/dr = −qΩ" value={formatNumber(-state.q * omega(2, state.q))} />
          </>
        ),
      }}
      caption={
        <>
          Units G = M = 1, so Ω(1) = 1 and the inner edge orbits once every 2π. At t = 0, {TRACERS} tracers sit on the
          line from r = 1 to r = 3 at angle 0, and each then moves on its own circular orbit, counter-clockwise, at Ω(r)
          = r^(−q). Play to wind the line. At q = 1.5 (Kepler) the inner tracers lap the outer ones and the line becomes
          a trailing spiral, its outer end lagging behind the rotation; its pitch angle p obeys tan p = 1/(qΩt). At q =
          1 (a flat rotation curve) it winds more slowly, and at q = 0 (rigid rotation) the line turns without bending.
        </>
      }
    >
      <Plot x={xa} y={ya}>
        <Curve name="r = 1" x={INNER.x} y={INNER.y} muted />
        <Curve name="r = 3" x={OUTER.x} y={OUTER.y} muted />
        <Curve name="the starting line, wound" x={line.x} y={line.y} slot={0} />
        <Points name="tracers" x={tracers.x} y={tracers.y} slot={0} size={5} />
        <Points name="inner end" x={[tracers.x[0]]} y={[tracers.y[0]]} emphasis />
      </Plot>
    </Figure>
  )
}
