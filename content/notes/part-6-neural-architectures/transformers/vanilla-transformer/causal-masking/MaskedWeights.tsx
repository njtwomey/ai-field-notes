import { useMemo } from 'react'
import { Figure, formatNumber, Plot, Raster, Readout, setting, useAxis, useFigureState } from 'aifn-render'
import { normal, stream } from 'aifn/foundation/random'

const N = 8

// Fixed random attention scores, one row per query position.
const SCORES = (() => {
  const r = stream(3)
  return Array.from({ length: N }, () => Array.from({ length: N }, () => 1.5 * normal(r)))
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
  const state = useFigureState({
    masked: setting(true, 'causal mask'),
  })
  const w = useMemo(() => weights(state.masked), [state.masked])
  const future = w.reduce((a, row, i) => a + row.slice(i + 1).reduce((s, v) => s + v, 0), 0) / N

  const xAxis = useAxis({ label: 'key position j' })
  const yAxis = useAxis({ label: 'query position i' })
  return (
    <Figure
      title="Causal mask on attention weights"
      state={state}
      caption="Each row is one query position and holds its softmax weights over the key positions; every row sums to 1. With the mask on, scores for keys after the query are set to −∞ before the softmax, so those weights are exactly 0 and the rest of the row is renormalised. Query 0 can attend only to itself."

      readouts={
        <>
          <Readout label="mean weight on future keys" value={formatNumber(future)} />
          <Readout label="non-zero weights" value={`${w.flat().filter((v) => v > 0).length} of ${N * N}`} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={340}>
        <Raster x={POS} y={POS} z={w} range={[0, 1]} valueLabel={'weight'} />
      </Plot>
    </Figure>
  )
}
