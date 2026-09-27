import { useMemo, useState } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, type XYSeries } from '@/components/viz'

type Box = { object: string; box: [number, number, number, number]; score: number }

// Ten candidate detections of three objects. A and B are two people standing close together; C stands alone.
const BOXES: Box[] = [
  { object: 'A', box: [1, 1, 4, 6], score: 0.95 },
  { object: 'A', box: [1.3, 1.2, 4.2, 6.3], score: 0.8 },
  { object: 'A', box: [0.8, 0.7, 3.7, 5.6], score: 0.7 },
  { object: 'A', box: [1.5, 1.8, 4.4, 6.6], score: 0.4 },
  { object: 'B', box: [2.4, 1.1, 5.4, 6.1], score: 0.9 },
  { object: 'B', box: [2.7, 0.9, 5.7, 5.9], score: 0.75 },
  { object: 'B', box: [2.2, 1.5, 5.2, 6.6], score: 0.5 },
  { object: 'C', box: [7.5, 2, 10, 4.5], score: 0.85 },
  { object: 'C', box: [7.7, 2.2, 10.3, 4.8], score: 0.6 },
  { object: 'C', box: [7.2, 1.8, 9.8, 4.3], score: 0.3 },
]

const area = ([x1, y1, x2, y2]: Box['box']) => (x2 - x1) * (y2 - y1)

function iou(a: Box['box'], b: Box['box']): number {
  const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]))
  const h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]))
  const inter = w * h
  return inter / (area(a) + area(b) - inter)
}

/** Greedy non-maximum suppression: returns a kept flag per box. */
function nms(boxes: Box[], iouThreshold: number, scoreThreshold: number): boolean[] {
  const order = boxes
    .map((_, i) => i)
    .filter((i) => boxes[i].score >= scoreThreshold)
    .sort((i, j) => boxes[j].score - boxes[i].score)
  const kept = boxes.map(() => false)
  const removed = boxes.map(() => false)
  for (const i of order) {
    if (removed[i]) continue
    kept[i] = true
    for (const j of order)
      if (!kept[j] && !removed[j] && iou(boxes[i].box, boxes[j].box) > iouThreshold) removed[j] = true
  }
  return kept
}

/** Box outlines joined into one series, separated by NaN so that ECharts breaks the line between boxes. */
function outlines(boxes: Box[]): { x: number[]; y: number[] } {
  const x: number[] = []
  const y: number[] = []
  for (const { box } of boxes) {
    const [x1, y1, x2, y2] = box
    x.push(x1, x2, x2, x1, x1, NaN)
    y.push(y1, y1, y2, y2, y1, NaN)
  }
  return { x, y }
}

/** Greedy non-maximum suppression on ten candidate boxes around three objects. */
export function NmsExplorer() {
  const [iouThreshold, setIouThreshold] = useState(0.5)
  const [scoreThreshold, setScoreThreshold] = useState(0.35)
  const kept = useMemo(() => nms(BOXES, iouThreshold, scoreThreshold), [iouThreshold, scoreThreshold])

  const series = useMemo<XYSeries[]>(() => {
    const keptBoxes = BOXES.filter((_, i) => kept[i])
    const dropped = BOXES.filter((_, i) => !kept[i])
    return [
      { name: 'suppressed or below score', type: 'line', ...outlines(dropped), muted: true, dashed: true },
      { name: 'kept', type: 'line', ...outlines(keptBoxes), slot: 0 },
    ]
  }, [kept])

  const keptList = BOXES.filter((_, i) => kept[i])
    .map((b) => `${b.object} ${b.score.toFixed(2)}`)
    .join(', ')

  return (
    <Interactive
      title="Greedy non-maximum suppression"
      caption="Ten candidate boxes around three objects. People A and B stand close together, so their best boxes overlap with IoU 0.35. NMS keeps the highest-scoring box, deletes every remaining box whose IoU with it exceeds the threshold, and repeats. At a threshold of 0.3 the best box for B is deleted by A's box, and a worse box for B survives instead. At 0.7 duplicates of A and C survive."
      controls={
        <>
          <ParamSlider
            label="IoU threshold"
            value={iouThreshold}
            onChange={setIouThreshold}
            min={0.1}
            max={0.9}
            step={0.05}
            format={(v) => v.toFixed(2)}
          />
          <ParamSlider
            label="score threshold"
            value={scoreThreshold}
            onChange={setScoreThreshold}
            min={0}
            max={0.9}
            step={0.05}
            format={(v) => v.toFixed(2)}
          />
        </>
      }
      readout={
        <>
          <Readout label="boxes kept" value={String(keptList ? keptList.split(', ').length : 0)} />
          <Readout label="kept (object, score)" value={keptList || 'none'} />
        </>
      }
    >
      <XYChart
        series={series}
        xRange={[0, 11]}
        yRange={[0, 7.5]}
        equalAspect
        xLabel="x"
        yLabel="y"
        ariaLabel="Candidate boxes, kept and suppressed"
      />
    </Interactive>
  )
}
