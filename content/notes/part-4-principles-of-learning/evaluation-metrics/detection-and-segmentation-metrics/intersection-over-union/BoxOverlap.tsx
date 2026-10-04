import { useMemo, useState } from 'react'
import {
  Figure,
  formatNumber,
  Handle,
  Plot,
  Readout,
  type Segment,
  Segments,
  seriesLayers,
  type SeriesSpec,
  useAxis,
} from 'aifn-render'
import { overlap, type Box } from '../_shared/boxes'
import { linspace, toFlat } from 'aifn/foundation/tensor'

type Vec = [number, number]
const VIEW: [number, number] = [0, 12]
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))

const outline = (name: string, b: Box, style: Partial<SeriesSpec>): SeriesSpec => ({
  name,
  type: 'line',
  x: [b[0], b[2], b[2], b[0], b[0]],
  y: [b[1], b[1], b[3], b[3], b[1]],
  ...style,
})

/**
 * A ground-truth box and a predicted box, both draggable. The readouts compare IoU with GIoU, DIoU and CIoU; the lower
 * chart sweeps the prediction horizontally to show that IoU is flat at 0 once the boxes separate, while the variants
 * still change and so still give a gradient.
 */
export function BoxOverlap() {
  const [truthCentre, setTruthCentre] = useState<Vec>([4.5, 5])
  const truthSize: Vec = [4, 4]
  const [predCentre, setPredCentre] = useState<Vec>([6.5, 6.5])
  const [predSize, setPredSize] = useState<Vec>([4, 5])

  const box = (c: Vec, s: Vec): Box => [c[0] - s[0] / 2, c[1] - s[1] / 2, c[0] + s[0] / 2, c[1] + s[1] / 2]
  const truth = box(truthCentre, truthSize)
  const pred = box(predCentre, predSize)
  const o = overlap(pred, truth)

  const sweep = useMemo(() => {
    const dx = toFlat(linspace(-8, 8, 161))
    const rows = dx.map((d) => overlap(box([truthCentre[0] + d, predCentre[1]], predSize), truth))
    const line = (name: string, key: 'iou' | 'giou' | 'diou' | 'ciou', slot: number): SeriesSpec => ({
      name,
      type: 'line',
      x: dx,
      y: rows.map((r) => r[key]),
      slot,
    })
    return [line('IoU', 'iou', 0), line('GIoU', 'giou', 1), line('DIoU', 'diou', 2), line('CIoU', 'ciou', 3)] as const
    // truth depends only on truthCentre and the fixed size.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [truthCentre, predCentre[1], predSize])

  const scene: SeriesSpec[] = [
    outline('ground truth', truth, { slot: 0 }),
    outline('prediction', pred, { slot: 1 }),
    outline('enclosing box C', o.enclosing, { muted: true, dashed: true }),
  ]
  const centres: Segment[] = [{ from: truthCentre, to: predCentre }]

  const offset = predCentre[0] - truthCentre[0]

  const xAxis = useAxis({ label: 'x', range: VIEW })
  const yAxis = useAxis({ label: 'y', range: VIEW, equal: xAxis })
  const xAxis2 = useAxis({ label: 'horizontal offset of the prediction', hold: 'union' })
  const yAxis2 = useAxis({ label: 'score', range: [-1, 1] })
  return (
    <Figure
      title="IoU and its variants"
      caption="Drag the prediction's centre or its top-right corner, or the ground truth's centre. The dashed box is C, the smallest box enclosing both, and the line joins the two centres. The lower chart moves the prediction horizontally with its current size and height: IoU is flat at 0 once the boxes stop overlapping, so it gives no gradient, while GIoU, DIoU and CIoU keep falling with distance."
      readouts={
        <>
          <Readout label="IoU" value={formatNumber(o.iou)} />
          <Readout label="GIoU" value={formatNumber(o.giou)} />
          <Readout label="DIoU" value={formatNumber(o.diou)} />
          <Readout label="CIoU" value={formatNumber(o.ciou)} />
          <Readout label="intersection / union" value={`${formatNumber(o.intersection)} / ${formatNumber(o.union)}`} />
          <Readout label="aspect term αv" value={formatNumber(o.alpha * o.v)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <div className="mx-auto w-full max-w-md">
          <Plot x={xAxis} y={yAxis}>
            {seriesLayers(scene)}
            <Segments segments={centres} />
            <Handle
              kind="point"
              at={predCentre}
              label="prediction centre"
              onDrag={([x, y]) => setPredCentre([clamp(x, 1, 11), clamp(y, 1, 11)])}
            />
            <Handle
              kind="point"
              at={[pred[2], pred[3]]}
              label="prediction corner"
              onDrag={([x, y]) =>
                setPredSize([clamp(2 * (x - predCentre[0]), 0.5, 10), clamp(2 * (y - predCentre[1]), 0.5, 10)])
              }
            />
            <Handle
              kind="point"
              at={truthCentre}
              label="ground-truth centre"
              onDrag={([x, y]) => setTruthCentre([clamp(x, 2, 10), clamp(y, 2, 10)])}
            />
          </Plot>
        </div>
        <Plot x={xAxis2} y={yAxis2} height={340}>
          {seriesLayers(sweep)}
          <Handle
            kind="x"
            at={offset}
            label="offset"
            onDrag={(d) => setPredCentre([clamp(truthCentre[0] + d, 1, 11), predCentre[1]])}
          />
        </Plot>
      </div>
    </Figure>
  )
}
