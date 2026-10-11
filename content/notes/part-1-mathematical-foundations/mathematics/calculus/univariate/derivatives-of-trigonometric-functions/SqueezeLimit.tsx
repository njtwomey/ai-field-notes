import { useContext, useMemo, type ReactNode } from 'react'
import {
  Curve,
  Figure,
  formatNumber,
  FrameContext,
  Handle,
  Plot,
  Points,
  Readout,
  Shapes,
  slider,
  useAxis,
  useElementSize,
  useFigureState,
  type FilledShape,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'

const H_MIN = 0.05
const H_MAX = 1.1
const TANGENT_LINE = { x: [1, 1], y: [-0.2, 2.1] }
const CIRCLE_T = toFlat(linspace(0, 2 * Math.PI, 241))
const CIRCLE = { x: CIRCLE_T.map(Math.cos), y: CIRCLE_T.map(Math.sin) }
const HS = toFlat(linspace(0.001, H_MAX, 281))
const RATIO = HS.map((h) => Math.sin(h) / h)
const LOWER = HS.map(Math.cos)
const ONE = HS.map(() => 1)

/** Plots inside a Figure take the frame's height; this gives the plots inside it a fixed height instead. */
function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

/**
 * The squeeze behind sin h / h → 1. For 0 < h < π/2 the triangle OAP lies inside the sector OAP, which lies inside the
 * triangle OAT, so ½ sin h ≤ ½ h ≤ ½ tan h, and dividing by ½ sin h traps sin h / h between cos h and 1.
 */
export function SqueezeLimit() {
  const state = useFigureState({
    h: slider(H_MIN, H_MAX, 0.9, { step: 0.01, label: 'angle h (rad)' }),
  })
  const { h } = state
  const c = Math.cos(h)
  const s = Math.sin(h)
  const t = Math.tan(h)

  const shapes = useMemo((): FilledShape[] => {
    const arc = toFlat(linspace(0, h, 64)).map((u): [number, number] => [Math.cos(u), Math.sin(u)])
    return [
      {
        contours: [
          [
            [0, 0],
            [1, 0],
            [1, t],
          ],
        ],
        tone: 2,
        opacity: 0.18,
      },
      { contours: [[[0, 0], ...arc]], tone: 1, opacity: 0.3 },
      {
        contours: [
          [
            [0, 0],
            [1, 0],
            [c, s],
          ],
        ],
        tone: 0,
        opacity: 0.45,
      },
    ]
  }, [h, c, s, t])
  const edges = useMemo(() => ({ x: [0, 1, 1, 0, c], y: [0, 0, t, 0, s] }), [c, s, t])

  const [cell, size] = useElementSize<HTMLDivElement>()
  const height = Math.max(240, Math.round(size.width))

  const x = useAxis({ label: 'x', range: [-0.2, 1.25] })
  const y = useAxis({ label: 'y', range: [-0.2, 2.1], equal: x })
  const hx = useAxis({ label: 'h (rad)', range: [0, H_MAX] })
  const ry = useAxis({ label: 'value', range: [0, 1.1] })

  return (
    <Figure
      title="Squeezing sin h / h"
      state={state}
      caption="Drag the point P = (cos h, sin h) along the unit circle. A = (1, 0), and T = (1, tan h) is where the ray OP meets the tangent line x = 1. Three nested regions share the vertex O: the triangle OAP (blue, area ½ sin h), the sector OAP (orange, area ½ h, because the angle is in radians) and the triangle OAT (green, area ½ tan h). Their areas are in increasing order for every h in (0, π/2). Dividing by ½ sin h and taking reciprocals gives cos h ≤ sin h / h ≤ 1, drawn on the right. As h → 0 the lower bound cos h rises to 1, so sin h / h is squeezed to 1."
      readouts={
        <>
          <Readout label="½ sin h (triangle OAP)" value={formatNumber(s / 2)} />
          <Readout label="½ h (sector)" value={formatNumber(h / 2)} />
          <Readout label="½ tan h (triangle OAT)" value={formatNumber(t / 2)} />
          <Readout label="cos h" value={formatNumber(c)} />
          <Readout label="sin h / h" value={formatNumber(s / h)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <div ref={cell}>
          <Plot x={x} y={y} fitHeight legend={false} title="three nested areas">
            <Shapes name="areas" shapes={shapes} />
            <Curve name="unit circle" x={CIRCLE.x} y={CIRCLE.y} muted silent />
            <Curve name="tangent line x = 1" x={TANGENT_LINE.x} y={TANGENT_LINE.y} muted dashed silent />
            <Curve name="edges" x={edges.x} y={edges.y} emphasis width={1} silent />
            <Handle
              kind="point"
              at={[c, s]}
              label="P"
              onDrag={([a, b]) => state.set('h', Math.min(H_MAX, Math.max(H_MIN, Math.atan2(b, a))))}
            />
          </Plot>
        </div>
        <FixedHeight height={height}>
          <Plot x={hx} y={ry} title="the squeeze">
            <Curve name="1" x={HS} y={ONE} muted dashed />
            <Curve name="sin h / h" x={HS} y={RATIO} slot={1} />
            <Curve name="cos h" x={HS} y={LOWER} slot={0} />
            <Points name="at h" x={[h, h]} y={[s / h, c]} emphasis size={7} live />
            <Handle {...state.handle('h', { label: 'h' })} />
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
