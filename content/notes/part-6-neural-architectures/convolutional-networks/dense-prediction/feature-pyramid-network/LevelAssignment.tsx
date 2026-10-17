import { useContext, useMemo, type ReactNode } from 'react'
import {
  Bars,
  Figure,
  formatNumber,
  FrameContext,
  Handle,
  Plot,
  Readout,
  Segments,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { boxIou } from 'aifn-methods/evaluation/detection'
import { anchorGrid, pyramidLevel, type Box } from 'aifn-methods/vision/detection'

const W = 1024
const H = 800
const LEVELS = [2, 3, 4, 5, 6]
const RATIOS = [0.5, 1, 2]
const PANEL = 330

/** Plots inside a Figure take the frame's height; this gives the plots inside it a fixed height instead. */
function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

type Seg = { from: [number, number]; to: [number, number] }
const outline = ([x0, y0, x1, y1]: Box): Seg[] => [
  { from: [x0, y0], to: [x1, y0] },
  { from: [x1, y0], to: [x1, y1] },
  { from: [x1, y1], to: [x0, y1] },
  { from: [x0, y1], to: [x0, y0] },
]

/** The anchors of level k (area (32 · 2^(k−2))², ratios 1:2, 1:1, 2:1) at the cell containing (cx, cy). */
function anchorsAt(k: number, cx: number, cy: number): Box[] {
  const stride = 2 ** k
  const col = Math.floor(cx / stride)
  const row = Math.floor(cy / stride)
  const { anchors } = anchorGrid({ featureShape: [1, 1], stride, sizes: [32 * 2 ** (k - 2)], ratios: RATIOS })
  return anchors.map(([x0, y0, x1, y1]) => [x0 + col * stride, y0 + row * stride, x1 + col * stride, y1 + row * stride])
}

/** Which pyramid level handles a box: FPN's equation for region features, and the anchor of best IoU for proposals. */
export function LevelAssignment() {
  const state = useFigureState({
    cx: slider(0, W, 520, { onChart: true }),
    cy: slider(0, H, 380, { onChart: true }),
    w: slider(8, W, 120, { onChart: true }),
    h: slider(8, H, 300, { onChart: true }),
  })
  const box: Box = [state.cx - state.w / 2, state.cy - state.h / 2, state.cx + state.w / 2, state.cy + state.h / 2]
  const r = useMemo(() => {
    const k = pyramidLevel(state.w, state.h)
    const stride = 2 ** k
    // Cell edges of level k over the box and one cell around it.
    const grid: Seg[] = []
    const [x0, y0, x1, y1] = box
    const gx0 = Math.max(0, Math.floor(x0 / stride - 1) * stride)
    const gx1 = Math.min(W, Math.ceil(x1 / stride + 1) * stride)
    const gy0 = Math.max(0, Math.floor(y0 / stride - 1) * stride)
    const gy1 = Math.min(H, Math.ceil(y1 / stride + 1) * stride)
    for (let gx = gx0; gx <= gx1; gx += stride) grid.push({ from: [gx, gy0], to: [gx, gy1] })
    for (let gy = gy0; gy <= gy1; gy += stride) grid.push({ from: [gx0, gy], to: [gx1, gy] })
    // The best anchor of each level at the box's centre cell.
    const best = LEVELS.map((lv) => {
      const as = anchorsAt(lv, state.cx, state.cy)
      const ious = as.map((a) => boxIou(box, a))
      const i = ious.indexOf(Math.max(...ious))
      return { level: lv, iou: ious[i], anchor: as[i] }
    })
    const top = best.reduce((a, b) => (b.iou > a.iou ? b : a))
    return { k, stride, grid, best, top }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.cx, state.cy, state.w, state.h])

  const x = useAxis({ label: 'x (pixels)', range: [0, W] })
  const y = useAxis({ label: 'y (pixels)', range: [0, H], inverse: true, equal: x })
  const lx = useAxis({ label: 'pyramid level', categories: LEVELS.map((k) => `P${k}`) })
  const ly = useAxis({ label: 'IoU of best anchor at the centre cell', range: [0, 1] })
  const clampW = (v: number) => Math.min(W, Math.max(8, v))
  const clampH = (v: number) => Math.min(H, Math.max(8, v))
  return (
    <Figure
      title="Which pyramid level handles an object"
      state={state}
      caption="Left: an 800 × 1024 image with one object box. Drag the centre point to move the box and the corner point to resize it. The grid is the cell lattice of the level that FPN's equation assigns to the box for region features, k = ⌊4 + log₂(√(wh)/224)⌋ clamped to P2 to P5: an unclamped box spans 14 to 28 cells per side of its level. The dashed box is the anchor of best IoU at the box's centre cell. Right: that best IoU on each level, for anchors of side 32 on P2 doubling to 512 on P6, in three aspect ratios. An anchor spans 8 cells per side of its level, so proposals, which are matched to anchors by IoU, sit one or two levels above the level the equation gives the same box."
      readouts={
        <>
          <Readout
            label="box"
            value={`${Math.round(state.w)} × ${Math.round(state.h)}, √(wh) = ${formatNumber(Math.sqrt(state.w * state.h))}`}
          />
          <Readout label="level by the equation" value={`P${r.k}, stride ${r.stride}`} />
          <Readout
            label="cells covered on that level"
            value={`${formatNumber(state.w / r.stride)} × ${formatNumber(state.h / r.stride)}`}
          />
          <Readout label="level of the best anchor" value={`P${r.top.level}, IoU ${formatNumber(r.top.iou)}`} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-[3fr_2fr]">
        <FixedHeight height={PANEL}>
          <Plot x={x} y={y} ariaLabel="Image with one box, the cell grid of its pyramid level and the best anchor">
            <Segments name={`P${r.k} cells`} segments={r.grid} muted />
            <Segments name="best anchor" segments={outline(r.top.anchor)} slot={1} dashed width={1.5} />
            <Segments name="object box" segments={outline(box)} slot={0} width={2} />
            <Handle {...state.handle(['cx', 'cy'], { label: 'centre' })} />
            <Handle
              kind="point"
              at={[box[2], box[3]]}
              label="corner"
              onDrag={([px, py]) => {
                state.set('w', clampW(2 * (px - state.cx)))
                state.set('h', clampH(2 * (py - state.cy)))
              }}
            />
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={lx} y={ly} ariaLabel="Best anchor IoU on each pyramid level">
            <Bars name="best anchor IoU" x={LEVELS.map((_, i) => i)} y={r.best.map((b) => b.iou)} slot={1} />
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
