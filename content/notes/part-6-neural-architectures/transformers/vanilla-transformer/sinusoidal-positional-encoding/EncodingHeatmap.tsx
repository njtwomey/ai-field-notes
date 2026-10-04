import { useMemo } from 'react'
import { Figure, float, int, Plot, Raster, useAxis, useFigureState } from 'aifn-render'

/** Sinusoidal encoding PE[pos, 2i] = sin(pos / base^(2i/d)), PE[pos, 2i+1] = cos(pos / base^(2i/d)). */
function encoding(positions: number, d: number, base: number): number[][] {
  return Array.from({ length: positions }, (_, pos) =>
    Array.from({ length: d }, (_, j) => {
      const angle = pos / base ** ((2 * Math.floor(j / 2)) / d)
      return j % 2 === 0 ? Math.sin(angle) : Math.cos(angle)
    }),
  )
}

export function EncodingHeatmap() {
  const state = useFigureState({
    positions: int(48, { min: 8, max: 64, step: 1, label: 'positions' }),
    d: int(32, { min: 8, max: 64, step: 2, label: 'd_model' }),
    logBase: float(4, {
      min: 1,
      max: 5,
      step: 0.1,
      label: 'base',
      points_per_decade: 2,
      logTransform: 'value-is-log',
      format: (v) => Math.round(10 ** v).toLocaleString(),
    }),
  })
  const base = 10 ** state.logBase
  const z = useMemo(() => encoding(state.positions, state.d, base), [state.positions, state.d, base])
  const dims = useMemo(() => Array.from({ length: state.d }, (_, j) => j), [state.d])
  const rows = useMemo(() => Array.from({ length: state.positions }, (_, p) => p), [state.positions])

  const xAxis = useAxis({ label: 'dimension' })
  const yAxis = useAxis({ label: 'position' })
  return (
    <Figure
      title="Sinusoidal positional encoding"
      state={state}
      caption="Each row is one position t and each column one dimension of the encoding. Columns come in (sin, cos) pairs at a shared frequency. Low dimensions oscillate fast and high dimensions slowly, so each row is a distinct pattern. A smaller base raises every frequency."
    >
      <Plot x={xAxis} y={yAxis} height={380}>
        <Raster x={dims} y={rows} z={z} scale={'diverging'} range={[-1, 1]} valueLabel={'PE'} />
      </Plot>
    </Figure>
  )
}
