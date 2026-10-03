import { useMemo } from 'react'
import { toFlat } from 'aifn/foundation/tensor'
import { alibiBias, alibiSlopes, positionMask, positionRange } from 'aifn/nn/attention'
import {
  Curve,
  Figure,
  Plot,
  Plots,
  Raster,
  Readout,
  formatNumber,
  row,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'

const BIAS_T = 24
const range = (n: number) => Array.from({ length: n }, (_, i) => i)

export function AlibiBias() {
  const state = useFigureState({
    config: row('Heads', {
      heads: slider(1, 16, 8, { label: 'total heads H', step: 1 }),
      head: slider(1, 16, 1, { label: 'active head h', step: 1 }),
    }),
  })

  const { heads, head } = state.config
  const h = Math.min(head, heads) - 1
  const slopes = useMemo(() => alibiSlopes(heads), [heads])
  const currentSlope = slopes[h]

  const pos = useMemo(() => positionRange(BIAS_T), [])

  const grid = useMemo(() => {
    const raw = toFlat(alibiBias(heads, pos, pos)).slice(h * BIAS_T * BIAS_T, (h + 1) * BIAS_T * BIAS_T)
    const mask = toFlat(positionMask(pos, pos, { causal: true }))
    return range(BIAS_T)
      .map((i) => range(BIAS_T).map((j) => (mask[i * BIAS_T + j] ? raw[i * BIAS_T + j] : NaN)))
      .reverse()
  }, [heads, h, pos])

  const offsets = useMemo(() => range(61).map((r) => r - 60), [])
  const curves = useMemo(() => slopes.map((m) => offsets.map((r) => m * r)), [slopes, offsets])

  const halfLife = Math.LN2 / currentSlope
  const maxPenalty = -currentSlope * (BIAS_T - 1)

  const kx = useAxis({ label: 'key position j' })
  const qy = useAxis({ label: 'query position i (top: 0)', equal: kx })
  const rx = useAxis({ label: 'relative offset (key − query)', range: [-60, 0] })
  const ry = useAxis({ label: 'bias −m_h · (i − j)', range: [-40, 0] })

  return (
    <Figure
      title="ALiBi geometric head biases"
      purpose="ALiBi penalises distance linearly with geometric slope m_h = 2^(-8h/H). Some heads have steep slopes (acting locally), while others have gentle slopes (attending across long context horizons)."
      defaultSize="L"
      state={state}
      readouts={{
        head: (
          <>
            <Readout label={`head ${h + 1} slope m_${h + 1}`} value={formatNumber(currentSlope)} />
            <Readout label="attention half-life" value={`${formatNumber(halfLife)} tokens`} />
            <Readout label={`penalty at offset −${BIAS_T - 1}`} value={formatNumber(maxPenalty)} />
            <Readout label="steepest head (h=1)" value={formatNumber(slopes[0])} />
            <Readout label={`gentlest head (h=${heads})`} value={formatNumber(slopes[heads - 1])} />
          </>
        ),
      }}
      caption="Left: The causal bias matrix for head h over sequence length 24. Future tokens (upper triangle) are masked out; past tokens are penalised linearly in distance. Right: The linear bias curves for all heads. Head 1 (steepest, shortest half-life) focuses on immediate neighbors, while head H attends across long distances."
    >
      <Plots cols={2}>
        <Plot x={kx} y={qy}>
          <Raster x={pos} y={pos} z={grid} valueLabel="bias" />
        </Plot>
        <Plot x={rx} y={ry}>
          {curves.map((curve, k) => (
            <Curve
              key={k}
              name={`head ${k + 1}`}
              x={offsets}
              y={curve}
              slot={k % 8}
              thin={k !== h}
              emphasis={k === h}
            />
          ))}
        </Plot>
      </Plots>
    </Figure>
  )
}
