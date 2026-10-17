import { useMemo } from 'react'
import {
  choice,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Points,
  Readout,
  row,
  Segments,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { boxIou } from 'aifn-methods/evaluation/detection'
import {
  anchorFreeTargets,
  anchorGrid,
  decodeBoxes,
  encodeBoxes,
  matchAnchors,
  type Box,
} from 'aifn-methods/vision/detection'

const SIDE = 256
const STRIDE = 16
const CELLS = SIDE / STRIDE
/** Nine anchors per cell: sides 32, 64, 128 by aspect ratios h/w = 1/2, 1, 2. */
const SIZES = [32, 64, 128]
const RATIOS = [0.5, 1, 2]
const GRID = anchorGrid({ featureShape: [CELLS, CELLS], stride: STRIDE, sizes: SIZES, ratios: RATIOS })
const POINTS = Array.from(
  { length: CELLS * CELLS },
  (_, i) => [((i % CELLS) + 0.5) * STRIDE, (Math.floor(i / CELLS) + 0.5) * STRIDE] as const,
)
const RULES = {
  retinanet: { positive: 0.5, negative: 0.4 },
  fasterrcnn: { positive: 0.7, negative: 0.3 },
} as const

type Seg = { from: [number, number]; to: [number, number] }
const outline = ([x0, y0, x1, y1]: Box): Seg[] => [
  { from: [x0, y0], to: [x1, y0] },
  { from: [x1, y0], to: [x1, y1] },
  { from: [x1, y1], to: [x0, y1] },
  { from: [x0, y1], to: [x0, y0] },
]
const CELL_EDGES: Seg[] = Array.from({ length: CELLS + 1 }, (_, i) => [
  { from: [i * STRIDE, 0] as [number, number], to: [i * STRIDE, SIDE] as [number, number] },
  { from: [0, i * STRIDE] as [number, number], to: [SIDE, i * STRIDE] as [number, number] },
]).flat()
const f2 = (v: number) => formatNumber(Math.round(v * 1000) / 1000)

/** Anchors on a stride-16 grid matched to a ground-truth box, the box's offset targets, and an anchor-free alternative. */
export function AnchorMatching() {
  const state = useFigureState({
    rule: choice(
      [
        { value: 'retinanet', label: 'anchors, IoU ≥ 0.5 positive, < 0.4 negative (RetinaNet)' },
        { value: 'fasterrcnn', label: 'anchors, IoU ≥ 0.7 positive, < 0.3 negative (RPN)' },
        { value: 'fcos', label: 'anchor-free: cells inside the box (FCOS)' },
      ],
      'retinanet',
      { label: 'assignment' },
    ),
    predicted: row(
      'predicted offsets for the best anchor',
      {
        tx: slider(-1, 1, 0.1, { step: 0.05, label: 't_x' }),
        ty: slider(-1, 1, -0.2, { step: 0.05, label: 't_y' }),
        tw: slider(-1, 1, 0.3, { step: 0.05, label: 't_w' }),
        th: slider(-1, 1, 0, { step: 0.05, label: 't_h' }),
      },
      { when: (v) => v.rule !== 'fcos' },
    ),
    cx: slider(0, SIDE, 130, { onChart: true }),
    cy: slider(0, SIDE, 80, { onChart: true }),
    w: slider(8, SIDE, 80, { onChart: true }),
    h: slider(8, SIDE, 60, { onChart: true }),
  })
  const truth: Box = [state.cx - state.w / 2, state.cy - state.h / 2, state.cx + state.w / 2, state.cy + state.h / 2]
  const { tx, ty, tw, th } = state.predicted

  const r = useMemo(() => {
    if (state.rule === 'fcos') {
      const targets = anchorFreeTargets(POINTS, [truth], [0, Infinity])
      const pos = targets.map((t, i) => (t.positive ? i : -1)).filter((i) => i >= 0)
      const best = pos.reduce((a, i) => (a < 0 || targets[i].centreness > targets[a].centreness ? i : a), -1)
      return { kind: 'fcos' as const, targets, pos, best }
    }
    const m = matchAnchors(GRID.anchors, [truth], RULES[state.rule])
    const positives = m.labels.map((l, i) => (l === 1 ? i : -1)).filter((i) => i >= 0)
    const ignored = m.labels.filter((l) => l === -1).length
    const best = m.bestAnchor[0]
    const anchor = GRID.anchors[best]
    const [target] = encodeBoxes([anchor], [truth])
    return { kind: 'anchors' as const, positives, ignored, best, bestIou: m.iou[best], anchor, target }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.rule, state.cx, state.cy, state.w, state.h])

  const decoded = r.kind === 'anchors' ? decodeBoxes([r.anchor], [[tx, ty, tw, th]])[0] : null

  const x = useAxis({ label: 'x (pixels)', range: [0, SIDE] })
  const y = useAxis({ label: 'y (pixels)', range: [0, SIDE], inverse: true, equal: x })
  const clamp = (v: number, hi: number) => Math.min(hi, Math.max(8, v))
  const bestCell = r.kind === 'anchors' ? GRID.index[r.best] : null
  const fcosBest = r.kind === 'fcos' && r.best >= 0 ? r.targets[r.best] : null

  return (
    <Figure
      title="Anchors, matching and box offsets"
      state={state}
      aspect={1}
      caption="A 256 × 256 image under a stride-16 feature map (16 × 16 cells), with nine anchors per cell: sides 32, 64 and 128 in aspect ratios h/w of ½, 1 and 2. Drag the centre and the corner of the ground-truth box (ink). Under an anchor rule, every anchor with IoU at or above the positive threshold is a positive (coloured outlines); the anchor of highest IoU (heavy outline) is made positive even below it, so no object goes unmatched. The readouts give that anchor's regression targets; the offset sliders decode a prediction from it (dashed). Under the anchor-free rule, every cell centre inside the box is a positive point, with distances to the four edges as its target and centre-ness as its quality. Small boxes find few or no anchors above 0.7; the anchor-free rule gives any box a positive for every cell it contains."
      readouts={
        r.kind === 'anchors'
          ? {
              matching: (
                <>
                  <Readout label="positive anchors" value={String(r.positives.length)} />
                  <Readout label="ignored anchors" value={String(r.ignored)} />
                  <Readout
                    label="best anchor"
                    value={`cell (${bestCell!.row}, ${bestCell!.col}), side ${SIZES[bestCell!.size]}, h/w ${RATIOS[bestCell!.ratio]}, IoU ${f2(r.bestIou)}`}
                  />
                  <Readout label="targets (t_x, t_y, t_w, t_h)" value={r.target.map(f2).join(', ')} />
                </>
              ),
              decoding: (
                <>
                  <Readout
                    label="decoded box (x₀, y₀, x₁, y₁)"
                    value={decoded!.map((v) => formatNumber(Math.round(v * 10) / 10)).join(', ')}
                  />
                  <Readout label="IoU of decoded box with the truth" value={f2(boxIou(truth, decoded!))} />
                </>
              ),
            }
          : {
              matching: (
                <>
                  <Readout label="positive points" value={String(r.pos.length)} />
                  <Readout
                    label="most central point: (l, t, r, b)"
                    value={
                      fcosBest ? fcosBest.ltrb.map((v) => formatNumber(Math.round(v * 10) / 10)).join(', ') : 'none'
                    }
                  />
                  <Readout label="its centre-ness" value={fcosBest ? f2(fcosBest.centreness) : 'none'} />
                </>
              ),
            }
      }
    >
      <Plot x={x} y={y} ariaLabel="Image grid with anchors or points matched to a ground-truth box">
        <Segments name="cells" segments={CELL_EDGES} muted width={0.5} />
        {r.kind === 'anchors' && r.positives.length > 0 && (
          <Segments
            name="positive anchors"
            segments={r.positives.flatMap((i) => outline(GRID.anchors[i]))}
            slot={0}
            width={1}
          />
        )}
        {r.kind === 'anchors' && <Segments name="best anchor" segments={outline(r.anchor)} slot={0} width={2.5} />}
        {r.kind === 'anchors' && decoded && (
          <Segments name="decoded prediction" segments={outline(decoded)} slot={1} dashed width={2} />
        )}
        {r.kind === 'fcos' && r.pos.length > 0 && (
          <Points
            name="positive points"
            x={r.pos.map((i) => POINTS[i][0])}
            y={r.pos.map((i) => POINTS[i][1])}
            slot={0}
            size={7}
          />
        )}
        {r.kind === 'fcos' && fcosBest && (
          <Segments
            name="edge distances of the most central point"
            segments={(() => {
              const [px, py] = POINTS[r.best]
              const [l, t, rr, b] = fcosBest.ltrb
              return [
                { from: [px, py] as [number, number], to: [px - l, py] as [number, number] },
                { from: [px, py] as [number, number], to: [px, py - t] as [number, number] },
                { from: [px, py] as [number, number], to: [px + rr, py] as [number, number] },
                { from: [px, py] as [number, number], to: [px, py + b] as [number, number] },
              ]
            })()}
            slot={1}
            width={1.5}
          />
        )}
        <Segments name="ground-truth box" segments={outline(truth)} emphasis width={2} />
        <Handle {...state.handle(['cx', 'cy'], { label: 'centre' })} />
        <Handle
          kind="point"
          at={[truth[2], truth[3]]}
          label="corner"
          onDrag={([px, py]) => {
            state.set('w', clamp(2 * (px - state.cx), SIDE))
            state.set('h', clamp(2 * (py - state.cy), SIDE))
          }}
        />
      </Plot>
    </Figure>
  )
}
