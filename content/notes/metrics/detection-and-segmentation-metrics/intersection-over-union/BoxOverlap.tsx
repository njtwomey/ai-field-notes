import { useMemo, useState } from 'react'
import { Interactive, Readout, XYChart, formatNumber, type Handle, type Segment, type XYSeries } from '@/components/viz'
import { linspace } from '@/lib/math'
import { overlap, type Box } from '../_shared/boxes'

type Vec = [number, number]
const VIEW: [number, number] = [0, 12]
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))

const outline = (name: string, b: Box, style: Partial<XYSeries>): XYSeries => ({
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
    const dx = linspace(-8, 8, 161)
    const rows = dx.map((d) => overlap(box([truthCentre[0] + d, predCentre[1]], predSize), truth))
    const line = (name: string, key: 'iou' | 'giou' | 'diou' | 'ciou', slot: number): XYSeries => ({
      name,
      type: 'line',
      x: dx,
      y: rows.map((r) => r[key]),
      slot,
    })
    return [line('IoU', 'iou', 0), line('GIoU', 'giou', 1), line('DIoU', 'diou', 2), line('CIoU', 'ciou', 3)]
    // truth depends only on truthCentre and the fixed size.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [truthCentre, predCentre[1], predSize])

  const scene: XYSeries[] = [
    outline('ground truth', truth, { slot: 0 }),
    outline('prediction', pred, { slot: 1 }),
    outline('enclosing box C', o.enclosing, { muted: true, dashed: true }),
  ]
  const centres: Segment[] = [{ from: truthCentre, to: predCentre }]

  const handles: Handle[] = [
    {
      kind: 'point',
      at: predCentre,
      label: 'prediction centre',
      onDrag: ([x, y]) => setPredCentre([clamp(x, 1, 11), clamp(y, 1, 11)]),
    },
    {
      kind: 'point',
      at: [pred[2], pred[3]],
      label: 'prediction corner',
      onDrag: ([x, y]) =>
        setPredSize([clamp(2 * (x - predCentre[0]), 0.5, 10), clamp(2 * (y - predCentre[1]), 0.5, 10)]),
    },
    {
      kind: 'point',
      at: truthCentre,
      label: 'ground-truth centre',
      onDrag: ([x, y]) => setTruthCentre([clamp(x, 2, 10), clamp(y, 2, 10)]),
    },
  ]
  const offset = predCentre[0] - truthCentre[0]
  const offsetHandle: Handle[] = [
    {
      kind: 'x',
      at: offset,
      label: 'offset',
      onDrag: (d) => setPredCentre([clamp(truthCentre[0] + d, 1, 11), predCentre[1]]),
    },
  ]

  return (
    <Interactive
      title="IoU and its variants"
      caption="Drag the prediction's centre or its top-right corner, or the ground truth's centre. The dashed box is C, the smallest box enclosing both, and the line joins the two centres. The lower chart moves the prediction horizontally with its current size and height: IoU is flat at 0 once the boxes stop overlapping, so it gives no gradient, while GIoU, DIoU and CIoU keep falling with distance."
      readout={
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
          <XYChart
            series={scene}
            segments={centres}
            xRange={VIEW}
            yRange={VIEW}
            equalAspect
            handles={handles}
            xLabel="x"
            yLabel="y"
          />
        </div>
        <XYChart
          series={sweep}
          xLabel="horizontal offset of the prediction"
          yLabel="score"
          yRange={[-1, 1]}
          height={340}
          handles={offsetHandle}
        />
      </div>
    </Interactive>
  )
}
