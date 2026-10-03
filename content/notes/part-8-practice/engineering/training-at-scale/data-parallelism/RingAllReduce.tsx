import { useMemo, useState } from 'react'
import { Heatmap, Interactive, ParamSlider, Readout, formatNumber } from 'aifn-render'

/**
 * Ring all-reduce on N workers, each holding N chunks. Cell (chunk c, worker i) counts how many workers' gradients have
 * been summed into worker i's copy of chunk c. Reduce-scatter adds a neighbour's partial sum; all-gather copies a
 * finished chunk.
 */
function simulate(n: number, steps: number): number[][] {
  let c = Array.from({ length: n }, () => new Array<number>(n).fill(1))
  for (let s = 0; s < steps; s++) {
    const next = c.map((row) => [...row])
    const gather = s >= n - 1
    const k = gather ? s - (n - 1) : s
    for (let i = 0; i < n; i++) {
      const chunk = (((gather ? i + 1 - k : i - k) % n) + n) % n
      const to = (i + 1) % n
      next[to][chunk] = gather ? c[i][chunk] : next[to][chunk] + c[i][chunk]
    }
    c = next
  }
  return c
}

export function RingAllReduce() {
  const [n, setN] = useState(4)
  const [step, setStep] = useState(0)
  const total = 2 * (n - 1)
  const s = Math.min(step, total)
  const z = useMemo(() => simulate(n, s), [n, s])
  const axis = useMemo(() => Array.from({ length: n }, (_, i) => i + 1), [n])
  const phase = s === 0 ? 'start' : s <= n - 1 ? 'reduce-scatter' : 'all-gather'

  return (
    <Interactive
      title="Ring all-reduce, one step at a time"
      caption="Each worker's gradient is split into N chunks. In every step, each worker sends one chunk to its right-hand neighbour. During reduce-scatter the neighbour adds the chunk to its own copy; after N − 1 steps each worker holds one chunk summed over all N workers. During all-gather the finished chunks travel round the ring and are copied. Colour counts the workers summed into each copy."
      controls={
        <>
          <ParamSlider
            label="workers N"
            value={n}
            onChange={(v) => {
              setN(v)
              setStep(0)
            }}
            min={2}
            max={8}
            step={1}
          />
          <ParamSlider label="step" value={s} onChange={setStep} min={0} max={total} step={1} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="phase" value={phase} />
          <Readout label="data sent per worker" value={`${s} × S/N = ${formatNumber(s / n)} S`} />
          <Readout label="total after 2(N − 1) steps" value={`${formatNumber(total / n)} S`} />
        </>
      }
    >
      <Heatmap
        x={axis}
        y={axis}
        z={z}
        xLabel="chunk"
        yLabel="worker"
        scale="sequential"
        range={[1, n]}
        valueLabel="workers summed"
        height={280}
      />
    </Interactive>
  )
}
