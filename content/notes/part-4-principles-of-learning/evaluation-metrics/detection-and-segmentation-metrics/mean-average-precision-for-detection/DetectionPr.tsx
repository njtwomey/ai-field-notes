import { useMemo } from 'react'
import { Curve, Figure, formatNumber, int, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { apAllPoint, apSampled, interpolated, prCurve } from '../_shared/ap'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { normal, stream, uniform } from 'aifn-compute/foundation/random'

const OBJECTS = 12
const BACKGROUND = 6
const COCO_THRESHOLDS = toFlat(linspace(0.5, 0.95, 10))

type Detection = { score: number; object: number; iou: number }

/** Simulated detections: most objects are found with some IoU, some twice, and a few false alarms fire on background. */
function simulate(seed: number): Detection[] {
  const g = stream(seed)
  const out: Detection[] = []
  const sigmoid = (z: number) => 1 / (1 + Math.exp(-z))
  for (let o = 0; o < OBJECTS; o++) {
    if (uniform(g) < 0.9) {
      const iou = 0.35 + 0.6 * uniform(g) ** 0.6
      out.push({ object: o, iou, score: sigmoid(6 * (iou - 0.55) + 0.8 * normal(g)) })
    }
    if (uniform(g) < 0.25) {
      const iou = 0.3 + 0.4 * uniform(g)
      out.push({ object: o, iou, score: sigmoid(4 * (iou - 0.7) + 0.8 * normal(g)) })
    }
  }
  for (let b = 0; b < BACKGROUND; b++) out.push({ object: -1, iou: 0, score: sigmoid(-1 + 1.2 * normal(g)) })
  return out.sort((a, b) => b.score - a.score)
}

/** Greedy matching in score order: a detection is a true positive if it overlaps an unmatched object by at least t. */
function match(dets: Detection[], t: number): boolean[] {
  const taken = new Set<number>()
  return dets.map((d) => {
    if (d.object < 0 || d.iou < t || taken.has(d.object)) return false
    taken.add(d.object)
    return true
  })
}

/**
 * One class of simulated detections, matched to ground truth at an IoU threshold. The precision–recall curve zigzags;
 * AP integrates its monotone envelope. COCO's mAP averages the 101-point AP over IoU thresholds 0.50 to 0.95.
 */
export function DetectionPr() {
  const state = useFigureState({
    threshold: slider(0.5, 0.95, 0.5, { step: 0.05, label: 'IoU threshold' }),
    seed: int(3, { min: 1, max: 40, step: 1, label: 'seed', format: (v) => String(v) }),
  })

  const r = useMemo(() => {
    const dets = simulate(state.seed)
    const tp = match(dets, state.threshold)
    const curve = prCurve(tp, OBJECTS)
    const envelope = curve.recall.map((rv) => interpolated(curve, rv))
    const coco =
      COCO_THRESHOLDS.reduce((s, t) => s + apSampled(prCurve(match(dets, t), OBJECTS), 101), 0) / COCO_THRESHOLDS.length
    return {
      curve,
      envelope,
      tp: tp.filter(Boolean).length,
      fp: tp.length - tp.filter(Boolean).length,
      all: apAllPoint(curve),
      eleven: apSampled(curve, 11),
      hundred: apSampled(curve, 101),
      coco,
    }
  }, [state.threshold, state.seed])

  const series = [
    { name: 'precision at each detection', x: r.curve.recall, y: r.curve.precision, slot: 0 },
    { name: 'interpolated envelope', x: r.curve.recall, y: r.envelope, slot: 1, dashed: true },
    { name: 'detections', x: r.curve.recall, y: r.curve.precision, slot: 0 },
  ] as const

  const xAxis = useAxis({ label: 'recall', range: [0, 1] })
  const yAxis = useAxis({ label: 'precision', range: [0, 1.05] })
  return (
    <Figure
      title="Average precision for one class"
      state={state}
      caption={`A detector's outputs for ${OBJECTS} objects, sorted by confidence and matched to ground truth at the IoU threshold: each object can be claimed once, so duplicates and background detections are false positives. Precision zigzags as recall grows; AP is the area under the dashed envelope, the best precision at each recall or higher. Raise the threshold to see strict localisation turn true positives into false ones. COCO's mAP averages the 101-point AP over thresholds 0.50 to 0.95.`}

      readouts={
        <>
          <Readout label="true / false positives" value={`${r.tp} / ${r.fp}`} />
          <Readout label="AP (all-point)" value={formatNumber(r.all)} />
          <Readout label="AP (11-point)" value={formatNumber(r.eleven)} />
          <Readout label="AP (101-point)" value={formatNumber(r.hundred)} />
          <Readout label="COCO AP@[.50:.95]" value={formatNumber(r.coco)} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-2xl">
        <Plot x={xAxis} y={yAxis} height={340}>
          <Curve {...series[0]} />
          <Curve {...series[1]} />
          <Points {...series[2]} />
        </Plot>
      </div>
    </Figure>
  )
}
