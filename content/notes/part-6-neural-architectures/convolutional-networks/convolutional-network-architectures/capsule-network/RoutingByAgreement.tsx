import { useContext, useMemo, useState, type ReactNode } from 'react'
import {
  Button,
  Curve,
  Figure,
  formatNumber,
  FrameContext,
  Handle,
  int,
  Player,
  Plot,
  Readout,
  slider,
  useAxis,
  useFigureState,
  Vectors,
  type Vector,
} from 'aifn-render'
import { norm, tensor } from 'aifn-compute/foundation/tensor'
import { capsulePredictions, dynamicRoutingTrace } from 'aifn-methods/neural/convolutional'

const DEG = Math.PI / 180
const PANEL = 280

/** Plots inside a Figure take the frame's height; this gives the three panels a fixed height instead. */
function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}
/** Gain of every part-to-object transformation: a part of length 0.9 predicts an object of length 1.44. */
const GAIN = 1.6
/** The orientation each part has in object A, and in object B (the same four part types, arranged differently). */
const ALPHA = [0, 90, 180, 270].map((d) => d * DEG)
const BETA = [0, 210, 60, 330].map((d) => d * DEG)
const PART_LENGTH = 0.9
const VIEW = 20 * DEG

const rotation = (a: number, gain = 1): number[][] => [
  [gain * Math.cos(a), -gain * Math.sin(a)],
  [gain * Math.sin(a), gain * Math.cos(a)],
]
/** W[i][j]: part i's pose rotated back from its orientation in object j, so a matching part predicts the object's pose. */
const WEIGHTS = ALPHA.map((a, i) => [rotation(-a, GAIN), rotation(-BETA[i], GAIN)])

const circle = (r: number) => {
  const t = Array.from({ length: 97 }, (_, k) => (2 * Math.PI * k) / 96)
  return { x: t.map((a) => r * Math.cos(a)), y: t.map((a) => r * Math.sin(a)) }
}
const UNIT = circle(1)

/** Part poses arranged as an object with orientations `angles`, viewed at angle `view`. */
const arrangement = (angles: number[], view: number) =>
  angles.map((a) => [PART_LENGTH * Math.cos(a + view), PART_LENGTH * Math.sin(a + view)])

const KEYS = [0, 1, 2, 3].map((i) => [`p${i + 1}x`, `p${i + 1}y`] as const)
const START = arrangement(ALPHA, VIEW)
const coord = (v: number) => slider(-1, 1, Math.round(v * 100) / 100, { step: 0.01, onChart: true })

/** Four part capsules route their predictions to two object capsules by agreement. */
export function RoutingByAgreement() {
  const state = useFigureState({
    iterations: int(3, { ge: 1, le: 10, label: 'routing iterations r', suggestions: [1, 2, 3, 5, 10] }),
    p1x: coord(START[0][0]),
    p1y: coord(START[0][1]),
    p2x: coord(START[1][0]),
    p2y: coord(START[1][1]),
    p3x: coord(START[2][0]),
    p3y: coord(START[2][1]),
    p4x: coord(START[3][0]),
    p4y: coord(START[3][1]),
  })
  const { iterations, p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y } = state
  const [picked, setPicked] = useState(0)
  const step = Math.min(picked, iterations - 1)

  const routed = useMemo(() => {
    const raw = [
      [p1x, p1y],
      [p2x, p2y],
      [p3x, p3y],
      [p4x, p4y],
    ]
    // A capsule's length is a probability: cap the dragged poses below 1.
    const poses = raw.map((u) => {
      const n = Math.hypot(u[0], u[1])
      return n > 0.99 ? u.map((x) => (0.99 * x) / n) : u
    })
    const predictions = capsulePredictions(WEIGHTS, poses)
    return { poses, predictions, trace: dynamicRoutingTrace(predictions, iterations) }
  }, [p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, iterations])
  const it = routed.trace[step]

  const place = (poses: number[][]) =>
    poses.forEach((u, i) => {
      state.set(KEYS[i][0], Math.round(u[0] * 100) / 100)
      state.set(KEYS[i][1], Math.round(u[1] * 100) / 100)
    })

  const parts: Vector[] = routed.poses.map((u, i): Vector => ({ from: [0, 0], to: [u[0], u[1]], label: `${i + 1}` }))
  const objectVectors = (j: number): Vector[] => [
    ...routed.predictions.map((row, i): Vector => ({
      from: [0, 0],
      to: [row[j][0], row[j][1]],
      label: `${i + 1}`,
      width: 1 + 5 * it.couplings[i][j],
      muted: it.couplings[i][j] < 0.3,
      head: 7,
    })),
    { from: [0, 0], to: [it.outputs[j][0], it.outputs[j][1]], slot: j, width: 4, label: `v${j === 0 ? 'A' : 'B'}` },
  ]

  const xp = useAxis({ label: 'pose, axis 1', range: [-1.1, 1.1] })
  const yp = useAxis({ label: 'pose, axis 2', range: [-1.1, 1.1], equal: xp })
  const xa = useAxis({ label: 'axis 1', range: [-1.8, 1.8] })
  const ya = useAxis({ label: 'axis 2', range: [-1.8, 1.8], equal: xa })
  const xb = useAxis({ label: 'axis 1', range: [-1.8, 1.8] })
  const yb = useAxis({ label: 'axis 2', range: [-1.8, 1.8], equal: xb })
  const couplingsTo = (j: number) => it.couplings.map((c) => formatNumber(c[j])).join(', ')
  return (
    <Figure
      title="Routing by agreement"
      state={state}
      defaultSize="L"
      controls={
        <>
          <Player
            value={step}
            onChange={setPicked}
            count={iterations}
            label="iteration"
            format={(k) => `${k + 1} of ${iterations}`}
          />
          <Button variant="outline" size="sm" onClick={() => place(arrangement(ALPHA, VIEW))}>
            parts as object A
          </Button>
          <Button variant="outline" size="sm" onClick={() => place(arrangement(BETA, VIEW))}>
            parts as object B
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => place(routed.poses.map((u) => rotation(30 * DEG).map((r) => r[0] * u[0] + r[1] * u[1])))}
          >
            turn all parts 30°
          </Button>
        </>
      }
      caption="Left: the outputs of four part capsules, as two-dimensional pose vectors whose length is the probability that the part is present; drag their tips. Middle and right: each part's prediction of the pose of object A and of object B, made by a fixed transformation per part and object, and the object capsule's output v (coloured) at this routing iteration. A prediction's thickness is its coupling coefficient; faint predictions have coupling below 0.3. Step through the iterations: the parts start by sending half of their output to each object, then move it towards the object whose output agrees with their prediction. With the parts arranged as object A, A's predictions coincide, its output grows towards length 1, and B's output shrinks towards 0. Turning all parts together turns v_A by the same angle and leaves its length and the couplings unchanged. Turn one part half a turn: its vote moves to object B, and that single long prediction keeps v_B near length 0.7, because routing has no rule that an object needs several agreeing parts."
      readouts={
        <>
          <Readout label="|v_A|, presence of object A" value={formatNumber(norm(tensor(it.outputs[0])))} />
          <Readout label="|v_B|, presence of object B" value={formatNumber(norm(tensor(it.outputs[1])))} />
          <Readout label="couplings of parts 1-4 to A" value={couplingsTo(0)} />
          <Readout label="couplings of parts 1-4 to B" value={couplingsTo(1)} />
        </>
      }
    >
      <FixedHeight height={PANEL}>
        <div className="grid gap-3 md:grid-cols-3">
          <Plot x={xp} y={yp} title="part capsules u_i" legend={false}>
            <Curve name="length 1" x={UNIT.x} y={UNIT.y} muted dashed silent />
            <Vectors vectors={parts} />
            {KEYS.map(([kx, ky], i) => (
              <Handle key={kx} {...state.handle([kx, ky], { label: `part ${i + 1}` })} />
            ))}
          </Plot>
          <Plot x={xa} y={ya} title="predictions of object A" legend={false}>
            <Curve name="length 1" x={UNIT.x} y={UNIT.y} muted dashed silent />
            <Vectors vectors={objectVectors(0)} />
          </Plot>
          <Plot x={xb} y={yb} title="predictions of object B" legend={false}>
            <Curve name="length 1" x={UNIT.x} y={UNIT.y} muted dashed silent />
            <Vectors vectors={objectVectors(1)} />
          </Plot>
        </div>
      </FixedHeight>
    </Figure>
  )
}
