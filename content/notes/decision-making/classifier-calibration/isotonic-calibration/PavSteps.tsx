import { useEffect, useMemo, useState } from 'react'
import { Interactive, Readout, StepControls, XYChart, formatNumber } from '@/components/viz'

const SCORES = [0.05, 0.12, 0.2, 0.27, 0.33, 0.41, 0.48, 0.55, 0.62, 0.7, 0.81, 0.9]
const LABELS = [0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1]

type Block = { start: number; end: number; mean: number }

/** Blocks after pool adjacent violators has absorbed the first `k` points, left to right. */
function pavPrefix(k: number): { blocks: Block[]; merges: number } {
  const blocks: Block[] = []
  let merges = 0
  for (let i = 0; i < k; i++) {
    blocks.push({ start: i, end: i, mean: LABELS[i] })
    while (blocks.length >= 2 && blocks[blocks.length - 2].mean > blocks[blocks.length - 1].mean) {
      const b = blocks.pop()!
      const a = blocks.pop()!
      const na = a.end - a.start + 1
      const nb = b.end - b.start + 1
      blocks.push({ start: a.start, end: b.end, mean: (a.mean * na + b.mean * nb) / (na + nb) })
      if (i === k - 1) merges++
    }
  }
  return { blocks, merges }
}

/** Step function through the blocks: each block spans from midway after the previous point to midway before the next. */
function stepLine(blocks: Block[]) {
  const x: number[] = []
  const y: number[] = []
  const edge = (i: number, side: -1 | 1) => {
    const j = i + side
    if (j < 0) return 0
    if (j >= SCORES.length) return 1
    return (SCORES[i] + SCORES[j]) / 2
  }
  for (const b of blocks) {
    x.push(edge(b.start, -1), edge(b.end, 1))
    y.push(b.mean, b.mean)
  }
  return { x, y }
}

/**
 * Pool adjacent violators on twelve labelled scores, one point per step. Each new point starts a block; while a block's
 * mean is below the mean of the block to its left, the two are pooled into one block with their weighted mean.
 */
export function PavSteps() {
  const [k, setK] = useState(0)
  const [running, setRunning] = useState(false)
  const { blocks, merges } = useMemo(() => pavPrefix(k), [k])
  const line = stepLine(blocks)
  const done = k >= SCORES.length

  useEffect(() => {
    // Running stops by itself at the end; Reset clears the flag.
    if (!running || done) return
    const id = setTimeout(() => setK((v) => v + 1), 500)
    return () => clearTimeout(id)
  }, [running, done, k])

  const fitted = blocks.flatMap((b) => new Array<number>(b.end - b.start + 1).fill(b.mean))
  const sse = fitted.reduce((acc, v, i) => acc + (v - LABELS[i]) ** 2, 0)
  const seen = SCORES.slice(0, k)
  const unseen = SCORES.slice(k)

  return (
    <Interactive
      title="Pool adjacent violators, one point at a time"
      caption="Twelve calibration cases sorted by score, with labels 0 or 1. Step through them from the left. Each new case starts its own block at its label. Whenever a block's value is lower than the block to its left, the monotonicity constraint is violated and the two blocks are pooled at their average. The final step function is the isotonic calibration map."
      controls={
        <StepControls
          onStep={() => setK((v) => Math.min(SCORES.length, v + 1))}
          onRun={() => setRunning(true)}
          onReset={() => {
            setRunning(false)
            setK(0)
          }}
          done={done}
        />
      }
      readout={
        <>
          <Readout label="cases absorbed" value={`${k} of ${SCORES.length}`} />
          <Readout label="blocks" value={blocks.length} />
          <Readout label="pools in the last step" value={merges} />
          <Readout label="squared error" value={formatNumber(sse)} />
        </>
      }
    >
      <XYChart
        height={300}
        xLabel="classifier score"
        yLabel="label / calibrated probability"
        xRange={[0, 1]}
        yRange={[-0.05, 1.05]}
        series={[
          { name: 'absorbed cases', type: 'scatter', x: seen, y: LABELS.slice(0, k), slot: 0 },
          { name: 'remaining cases', type: 'scatter', x: unseen, y: LABELS.slice(k), muted: true },
          { name: 'isotonic fit so far', type: 'line', x: line.x, y: line.y, emphasis: true },
        ]}
      />
    </Interactive>
  )
}
