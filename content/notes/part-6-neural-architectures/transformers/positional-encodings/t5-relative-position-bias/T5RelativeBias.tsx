import { useMemo } from 'react'
import { positionMask, positionRange, t5RelativeBucket } from 'aifn/nn/attention'
import { toFlat } from 'aifn/foundation/tensor'
import { Curve, Figure, Plot, Plots, Raster, Readout, row, setting, slider, useAxis, useFigureState } from 'aifn-render'

const BIAS_T = 24
const range = (n: number) => Array.from({ length: n }, (_, i) => i)

export function T5RelativeBias() {
  const state = useFigureState({
    config: row('Buckets', {
      buckets: slider(8, 64, 32, { label: 'buckets B', step: 8 }),
      maxDistance: slider(16, 256, 128, { label: 'max distance', step: 16 }),
      bidirectional: setting(true, 'bidirectional (encoder)'),
    }),
  })

  const { buckets, maxDistance, bidirectional } = state.config

  const pos = useMemo(() => positionRange(BIAS_T), [])

  const grid = useMemo(() => {
    const b = pos.flatMap((p) => pos.map((q) => t5RelativeBucket(q - p, { buckets, maxDistance, bidirectional })))
    const mask = toFlat(positionMask(pos, pos, { causal: !bidirectional }))
    return range(BIAS_T)
      .map((i) => range(BIAS_T).map((j) => (mask[i * BIAS_T + j] ? b[i * BIAS_T + j] : NaN)))
      .reverse()
  }, [buckets, maxDistance, bidirectional, pos])

  const offsets = useMemo(() => {
    const span = Math.min(128, maxDistance + 16)
    return range(2 * span + 1).map((r) => r - span)
  }, [maxDistance])

  const curve = useMemo(() => {
    return offsets.map((r) => {
      if (!bidirectional && r > 0) return NaN
      return t5RelativeBucket(r, { buckets, maxDistance, bidirectional })
    })
  }, [offsets, buckets, maxDistance, bidirectional])

  const exactHalf = Math.floor((bidirectional ? Math.floor(buckets / 2) : buckets) / 2)

  const kx = useAxis({ label: 'key position j' })
  const qy = useAxis({ label: 'query position i (top: 0)', equal: kx })
  const rx = useAxis({ label: 'relative offset (key − query)' })
  const ry = useAxis({ label: 'bucket index β(j − i)', range: [0, buckets] })

  return (
    <Figure
      title="T5 relative position bucketing"
      purpose="T5 assigns relative offsets to discrete buckets: small offsets are exact (1 bucket per integer shift), medium offsets are pooled logarithmically (wider buckets further out), and all offsets beyond maxDistance share the final bucket."
      defaultSize="L"
      state={state}
      readouts={{
        buckets: (
          <>
            <Readout label="exact bucket threshold" value={`|r| < ${exactHalf}`} />
            <Readout label="logarithmic range" value={`${exactHalf} ≤ |r| < ${maxDistance}`} />
            <Readout
              label="bucket at offset −1"
              value={String(t5RelativeBucket(-1, { buckets, maxDistance, bidirectional }))}
            />
            <Readout
              label="bucket at offset −10"
              value={String(t5RelativeBucket(-10, { buckets, maxDistance, bidirectional }))}
            />
            <Readout
              label="bucket at offset −100"
              value={String(t5RelativeBucket(-100, { buckets, maxDistance, bidirectional }))}
            />
          </>
        ),
      }}
      caption="Left: Toeplitz bucket assignment matrix over 24 tokens. Each diagonal has constant relative distance, mapping to identical bucket indices. In causal mode, upper triangle values are omitted. Right: Bucket index against relative position (j − i). Offsets near zero show a 1:1 staircase, transitions to a logarithmic curve as distance increases, and plateaus for distances exceeding maxDistance."
    >
      <Plots cols={2}>
        <Plot x={kx} y={qy}>
          <Raster x={pos} y={pos} z={grid} valueLabel="bucket" />
        </Plot>
        <Plot x={rx} y={ry}>
          <Curve name="bucket index" x={offsets} y={curve} slot={0} />
        </Plot>
      </Plots>
    </Figure>
  )
}
