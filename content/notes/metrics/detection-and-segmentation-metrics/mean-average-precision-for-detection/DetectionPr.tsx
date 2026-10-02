import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from 'aifn-render'
import { linspace, rng } from '@/lib/math'
import { apAllPoint, apSampled, interpolated, prCurve } from '../_shared/ap'

const OBJECTS = 12
const BACKGROUND = 6
const COCO_THRESHOLDS = linspace(0.5, 0.95, 10)

type Detection = { score: number; object: number; iou: number }

/** Simulated detections: most objects are found with some IoU, some twice, and a few false alarms fire on background. */
function simulate(seed: number): Detection[] {
  const g = rng(seed)
  const out: Detection[] = []
  const sigmoid = (z: number) => 1 / (1 + Math.exp(-z))
  for (let o = 0; o < OBJECTS; o++) {
    if (g.uniform() < 0.9) {
      const iou = 0.35 + 0.6 * g.uniform() ** 0.6
      out.push({ object: o, iou, score: sigmoid(6 * (iou - 0.55) + 0.8 * g.normal()) })
    }
    if (g.uniform() < 0.25) {
      const iou = 0.3 + 0.4 * g.uniform()
      out.push({ object: o, iou, score: sigmoid(4 * (iou - 0.7) + 0.8 * g.normal()) })
    }
  }
  for (let b = 0; b < BACKGROUND; b++) out.push({ object: -1, iou: 0, score: sigmoid(-1 + 1.2 * g.normal()) })
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
  const threshold = useParam(0.5, { min: 0.5, max: 0.95, step: 0.05 })
  const seed = useParam(3, { min: 1, max: 40, step: 1 })

  const r = useMemo(() => {
    const dets = simulate(seed.value)
    const tp = match(dets, threshold.value)
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
  }, [threshold.value, seed.value])

  const series: XYSeries[] = [
    { name: 'precision at each detection', type: 'line', x: r.curve.recall, y: r.curve.precision, slot: 0 },
    { name: 'interpolated envelope', type: 'line', x: r.curve.recall, y: r.envelope, slot: 1, dashed: true },
    { name: 'detections', type: 'scatter', x: r.curve.recall, y: r.curve.precision, slot: 0 },
  ]

  return (
    <Interactive
      title="Average precision for one class"
      caption={`A detector's outputs for ${OBJECTS} objects, sorted by confidence and matched to ground truth at the IoU threshold: each object can be claimed once, so duplicates and background detections are false positives. Precision zigzags as recall grows; AP is the area under the dashed envelope, the best precision at each recall or higher. Raise the threshold to see strict localisation turn true positives into false ones. COCO's mAP averages the 101-point AP over thresholds 0.50 to 0.95.`}
      controls={
        <>
          <ParamSlider label="IoU threshold" param={threshold} />
          <ParamSlider label="seed" param={seed} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
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
        <XYChart series={series} xLabel="recall" yLabel="precision" xRange={[0, 1]} yRange={[0, 1.05]} height={340} />
      </div>
    </Interactive>
  )
}
