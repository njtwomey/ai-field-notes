import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamButton,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  type Handle,
  type Segment,
  type XYSeries,
} from '@/components/viz'
import { rng } from '@/lib/math'

const PAIRS = 4
const R = 1.35
const CIRCLE = Array.from({ length: 121 }, (_, i) => (2 * Math.PI * i) / 120)
const CIRCLE_X = CIRCLE.map(Math.cos)
const CIRCLE_Y = CIRCLE.map(Math.sin)
const GROUP = Array.from({ length: 2 * PAIRS }, (_, i) => Math.floor(i / 2))
const GROUP_NAMES = Array.from({ length: PAIRS }, (_, p) => `pair ${p + 1}`)

type Pt = [number, number]
const at = (theta: number): Pt => [Math.cos(theta), Math.sin(theta)]
const d2 = (a: Pt, b: Pt) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

/** Angles for the 8 embeddings: indices 2p and 2p + 1 are the two views of input p. */
const PRESETS = {
  spread: Array.from({ length: 2 * PAIRS }, (_, i) => (Math.PI / 2) * Math.floor(i / 2) + (i % 2 ? 0.12 : -0.12)),
  collapsed: Array.from({ length: 2 * PAIRS }, (_, i) => 0.8 + 0.03 * i),
  misaligned: Array.from({ length: 2 * PAIRS }, (_, i) => (Math.PI / 4) * Math.floor(i / 2) + (i % 2 ? Math.PI : 0)),
}

function random(seed: number): number[] {
  const r = rng(seed)
  return Array.from({ length: 2 * PAIRS }, () => 2 * Math.PI * r.uniform())
}

/**
 * Alignment E‖f(x) − f(y)‖² over positive pairs, uniformity log E exp(−2‖f(x) − f(y)‖²) over all pairs of distinct
 * embeddings, and the InfoNCE loss with each embedding as an anchor and the six non-partners as negatives.
 */
function metrics(points: Pt[], tau: number) {
  let align = 0
  for (let p = 0; p < PAIRS; p++) align += d2(points[2 * p], points[2 * p + 1])
  align /= PAIRS
  let u = 0
  let count = 0
  for (let i = 0; i < points.length; i++)
    for (let j = i + 1; j < points.length; j++) {
      u += Math.exp(-2 * d2(points[i], points[j]))
      count++
    }
  const uniform = Math.log(u / count)
  let nce = 0
  points.forEach((a, i) => {
    const partner = i ^ 1
    // Cosine similarity of unit vectors is 1 − ‖a − b‖² / 2.
    const logits = points.map((b, j) => (j === i ? -Infinity : (1 - d2(a, b) / 2) / tau))
    const top = Math.max(...logits)
    const z = logits.reduce((s, l) => s + Math.exp(l - top), 0)
    nce += -(logits[partner] - top - Math.log(z))
  })
  return { align, uniform, nce: nce / points.length }
}

export function AlignUniform() {
  const [angles, setAngles] = useState<number[]>(PRESETS.spread)
  const [seed, setSeed] = useState(1)
  const [logTau, setLogTau] = useState(-0.7)
  const tau = 10 ** logTau

  const points = useMemo(() => angles.map(at), [angles])
  const m = useMemo(() => metrics(points, tau), [points, tau])

  const series: XYSeries[] = [
    { name: 'unit circle', type: 'line', x: CIRCLE_X, y: CIRCLE_Y, muted: true },
    {
      name: 'embedding',
      type: 'scatter',
      x: points.map((p) => p[0]),
      y: points.map((p) => p[1]),
      group: GROUP,
      groupNames: GROUP_NAMES,
    },
  ]
  const segments: Segment[] = Array.from({ length: PAIRS }, (_, p) => ({ from: points[2 * p], to: points[2 * p + 1] }))
  const handles: Handle[] = points.map((p, i) => ({
    kind: 'point',
    at: p,
    onDrag: ([x, y]) => setAngles((a) => a.map((v, j) => (j === i ? Math.atan2(y, x) : v))),
  }))

  return (
    <Interactive
      title="Alignment and uniformity on the unit circle"
      caption="Eight normalised embeddings: four inputs, each seen through two augmentations that share a colour and marker shape. Drag any point along the circle. Alignment is small when the two views of each input sit together. Uniformity is small (more negative) when all eight points spread around the circle. A collapsed encoder is perfectly aligned but has the worst uniformity."
      controls={
        <>
          <ParamSlider
            label="InfoNCE temperature τ"
            value={logTau}
            onChange={setLogTau}
            min={-2}
            max={0}
            step={0.05}
            format={(v) => formatNumber(10 ** v)}
          />
          <div className="flex flex-wrap gap-2">
            <ParamButton onClick={() => setAngles(PRESETS.spread)}>Aligned and spread</ParamButton>
            <ParamButton onClick={() => setAngles(PRESETS.collapsed)}>Collapsed</ParamButton>
            <ParamButton onClick={() => setAngles(PRESETS.misaligned)}>Spread, not aligned</ParamButton>
            <ParamButton
              onClick={() => {
                setSeed((s) => s + 1)
                setAngles(random(seed + 1))
              }}
            >
              Random
            </ParamButton>
          </div>
        </>
      }
      readout={
        <>
          <Readout label="alignment (lower is better)" value={formatNumber(m.align)} />
          <Readout label="uniformity (lower is better)" value={formatNumber(m.uniform)} />
          <Readout label="InfoNCE loss (nats)" value={formatNumber(m.nce)} />
        </>
      }
    >
      <XYChart
        series={series}
        segments={segments}
        handles={handles}
        xRange={[-R, R]}
        yRange={[-R, R]}
        equalAspect
        bare
      />
    </Interactive>
  )
}
