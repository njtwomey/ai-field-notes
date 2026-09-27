import { useMemo, useState } from 'react'
import { Heatmap, Interactive, ParamSwitch, Readout, formatNumber } from '@/components/viz'
import { rng } from '@/lib/math'

const N = 8

// Fixed random attention scores, one row per query position.
const SCORES = (() => {
  const r = rng(3)
  return Array.from({ length: N }, () => Array.from({ length: N }, () => 1.5 * r.normal()))
})()

function weights(masked: boolean): number[][] {
  return SCORES.map((row, i) => {
    const s = row.map((v, j) => (masked && j > i ? -Infinity : v))
    const m = Math.max(...s)
    const e = s.map((v) => Math.exp(v - m))
    const z = e.reduce((a, b) => a + b, 0)
    return e.map((v) => v / z)
  })
}

const POS = Array.from({ length: N }, (_, i) => i)

/** Attention weights of an 8-token sequence with and without the causal mask. */
export function MaskedWeights() {
  const [masked, setMasked] = useState(true)
  const w = useMemo(() => weights(masked), [masked])
  const future = w.reduce((a, row, i) => a + row.slice(i + 1).reduce((s, v) => s + v, 0), 0) / N

  return (
    <Interactive
      title="Causal mask on attention weights"
      caption="Each row is one query position and holds its softmax weights over the key positions; every row sums to 1. With the mask on, scores for keys after the query are set to −∞ before the softmax, so those weights are exactly 0 and the rest of the row is renormalised. Query 0 can attend only to itself."
      controls={<ParamSwitch label="causal mask" checked={masked} onChange={setMasked} />}
      readout={
        <>
          <Readout label="mean weight on future keys" value={formatNumber(future)} />
          <Readout label="non-zero weights" value={`${w.flat().filter((v) => v > 0).length} of ${N * N}`} />
        </>
      }
    >
      <Heatmap
        x={POS}
        y={POS}
        z={w}
        range={[0, 1]}
        xLabel="key position j"
        yLabel="query position i"
        valueLabel="weight"
        height={340}
      />
    </Interactive>
  )
}
