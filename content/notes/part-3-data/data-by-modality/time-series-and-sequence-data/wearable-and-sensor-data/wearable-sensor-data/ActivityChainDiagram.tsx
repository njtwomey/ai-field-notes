import { Diagram, Figure } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const W = 3
const H = 1.2
const STEP = 3.6

const top = [
  { id: 'sensors', label: 'raw sensor\nstreams' },
  { id: 'calibrate', label: 'calibrate\nunits and offsets' },
  { id: 'orient', label: 'orientation,\ngravity removal' },
  { id: 'filter', label: 'filter noise\nand artefacts' },
  { id: 'sync', label: 'resample and\nsynchronise' },
]
const bottom = [
  { id: 'windows', label: 'segment into\nwindows' },
  { id: 'features', label: 'features or\nlearned encoder' },
  { id: 'classify', label: 'classify\neach window' },
  { id: 'smooth', label: 'smooth over\ntime' },
  { id: 'activities', label: 'activity\nsequence' },
]

const spec: DiagramSpec = {
  unit: 40,
  nodes: [
    ...top.map((n, i) => ({ ...n, x: i * STEP, y: 0, w: W, h: H, tone: i === 0 ? ('neutral' as const) : 0 })),
    ...bottom.map((n, i) => ({
      ...n,
      x: (top.length - 1 - i) * STEP,
      y: 2.8,
      w: W,
      h: H,
      tone: i === bottom.length - 1 ? ('neutral' as const) : i < 2 ? 1 : 2,
    })),
  ],
  edges: [
    ...top.slice(1).map((n, i) => ({ from: top[i].id, to: n.id })),
    { from: 'sync', to: 'windows' },
    ...bottom.slice(1).map((n, i) => ({ from: bottom[i].id, to: n.id })),
  ],
}

/** The usual chain from raw wearable or ambient sensor streams to a sequence of recognised activities. */
export function ActivityChainDiagram() {
  return (
    <Figure
      title="The activity-recognition chain"
      hoverReadout={false}
      caption="The steps from raw sensor streams to recognised activities. The first row prepares the signals: calibration, orientation and gravity handling, filtering, and alignment of every sensor onto one timeline. The second row represents them: windows, features or a learned encoder, a classifier per window, and smoothing of the window predictions over time. Errors made early, such as a wrong clock offset or a gravity component left in, propagate through every later step."
    >
      <Diagram
        spec={spec}
        ariaLabel="Raw sensor streams are calibrated, corrected for orientation and gravity, filtered, then resampled and synchronised; the result is segmented into windows, turned into features, classified per window, and smoothed over time into an activity sequence"
      />
    </Figure>
  )
}
