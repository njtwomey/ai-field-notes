import { useMemo } from 'react'
import {
  Curve,
  Figure,
  formatNumber,
  Handle,
  Player,
  Plot,
  Points,
  Readout,
  Segments,
  Shapes,
  slider,
  useAxis,
  useFigureState,
  usePlayhead,
} from 'aifn-render'
import { eccentricAnomaly, trueFromEccentric } from './anomalies'

const fmt = (v: number) => formatNumber(v)
const deg = (v: number) => `${fmt(v)} rad (${((v * 180) / Math.PI).toFixed(1)}°)`
const TAU = 2 * Math.PI
/** Player frames over one orbit: the mean anomaly runs from 0 to 2π in equal steps of time. */
const FRAMES = 360
const ARC = 240

export function AnomalyDiagram() {
  const state = useFigureState({ e: slider(0, 0.95, 0.6, { label: 'eccentricity e', step: 0.01 }) })
  const { e } = state
  const [frame, setFrame] = usePlayhead(FRAMES + 1)
  const M = (TAU * frame) / FRAMES
  const { E, steps } = useMemo(() => eccentricAnomaly(e, M), [e, M])
  const nu = trueFromEccentric(e, E)
  const b = Math.sqrt(1 - e * e)
  const focus: [number, number] = [e, 0]

  const ellipse = useMemo(() => {
    const t = Array.from({ length: ARC + 1 }, (_, i) => (TAU * i) / ARC)
    return { x: t.map(Math.cos), y: t.map((s) => b * Math.sin(s)), cx: t.map(Math.cos), cy: t.map(Math.sin) }
  }, [b])

  // The area swept from the focus since pericentre: the focus, then the ellipse from E = 0 to the current E.
  const swept = useMemo(() => {
    const n = Math.max(2, Math.ceil((ARC * E) / TAU))
    const pts: (readonly [number, number])[] = [[e, 0]]
    for (let i = 0; i <= n; i++) {
      const s = (E * i) / n
      pts.push([Math.cos(s), b * Math.sin(s)])
    }
    return pts
  }, [E, b, e])
  const sweptFraction = (E - e * Math.sin(E)) / TAU

  const P: [number, number] = [Math.cos(E), b * Math.sin(E)]
  const Q: [number, number] = [Math.cos(E), Math.sin(E)]
  const mean: [number, number] = [Math.cos(M), Math.sin(M)]
  const r = 1 - e * Math.cos(E)

  const xa = useAxis({ label: 'x / a', range: [-1.25, 1.25] })
  const ya = useAxis({ label: 'y / a', range: [-1.25, 1.25], equal: xa })

  return (
    <Figure
      title="Mean, eccentric and true anomaly"
      state={state}
      defaultSize="L"
      aspect={1}
      controls={
        <Player
          value={frame}
          onChange={setFrame}
          count={FRAMES + 1}
          label="time since pericentre"
          format={(i) => `t/T = ${(i / FRAMES).toFixed(3)}`}
        />
      }
      readouts={{
        anomalies: (
          <>
            <Readout label="mean anomaly M = 2π t/T" value={deg(M)} />
            <Readout label="eccentric anomaly E" value={deg(E)} />
            <Readout label="true anomaly ν" value={deg(nu)} />
          </>
        ),
        checks: (
          <>
            <Readout label="distance r/a = 1 − e cos E" value={fmt(r)} />
            <Readout label="swept area / ellipse area" value={fmt(sweptFraction)} />
            <Readout label="M / 2π" value={fmt(M / TAU)} />
            <Readout label="Newton steps from E₀ = π" value={steps} />
          </>
        ),
      }}
      caption="An ellipse with semi-major axis a = 1 and eccentricity e, its auxiliary circle (dashed) and the attracting mass at the focus F. Play to move time on; the mean point M runs round the circle at a constant rate, and dragging it sets the time. The eccentric point Q is on the circle directly above or below the body P. The true anomaly ν is the angle at the focus from pericentre (the right end of the major axis) to P. The shaded area swept from F grows in proportion to time, so its share of the ellipse equals M/2π, which is Kepler's equation."
    >
      <Plot x={xa} y={ya}>
        <Shapes name="area swept since pericentre" shapes={[{ contours: [swept], tone: 0, opacity: 0.18 }]} />
        <Curve name="auxiliary circle" x={ellipse.cx} y={ellipse.cy} muted dashed />
        <Curve name="orbit" x={ellipse.x} y={ellipse.y} slot={0} />
        <Segments
          name="construction"
          segments={[
            { from: [0, 0], to: Q },
            { from: Q, to: [Q[0], 0] },
            { from: [0, 0], to: mean },
            { from: [-1, 0], to: [1, 0] },
          ]}
        />
        <Segments name="radius from the focus" segments={[{ from: focus, to: P }]} slot={0} width={2} />
        <Points name="centre and focus" x={[0, e]} y={[0, 0]} labels={['C', 'F']} emphasis size={7} />
        <Points name="eccentric point Q" x={[Q[0]]} y={[Q[1]]} labels={['Q']} slot={1} size={8} />
        <Points name="body P" x={[P[0]]} y={[P[1]]} labels={['P']} slot={0} size={10} />
        <Handle
          kind="point"
          at={mean}
          label="M"
          onDrag={([px, py]) => {
            const angle = (Math.atan2(py, px) + TAU) % TAU
            setFrame(Math.round((angle / TAU) * FRAMES))
          }}
        />
      </Plot>
    </Figure>
  )
}
