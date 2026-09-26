import { useMemo, useState } from 'react'
import { Heatmap, Interactive, ParamSlider } from '@/components/viz'

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
  const [positions, setPositions] = useState(48)
  const [d, setD] = useState(32)
  const [logBase, setLogBase] = useState(4)
  const base = 10 ** logBase
  const z = useMemo(() => encoding(positions, d, base), [positions, d, base])

  return (
    <Interactive
      title="Sinusoidal positional encoding"
      caption="Each row is one position; each column is one dimension. Low dimensions oscillate fast and high dimensions slowly. A smaller base makes every dimension oscillate faster."
      controls={
        <>
          <ParamSlider label="positions" value={positions} onChange={setPositions} min={8} max={64} step={1} />
          <ParamSlider label="d_model" value={d} onChange={setD} min={8} max={64} step={2} />
          <ParamSlider
            label="base"
            value={logBase}
            onChange={setLogBase}
            min={1}
            max={5}
            step={0.1}
            format={(v) => Math.round(10 ** v).toLocaleString()}
          />
        </>
      }
    >
      <Heatmap
        x={Array.from({ length: d }, (_, j) => j)}
        y={Array.from({ length: positions }, (_, p) => p)}
        z={z}
        scale="diverging"
        range={[-1, 1]}
        xLabel="dimension"
        yLabel="position"
        valueLabel="PE"
        height={380}
      />
    </Interactive>
  )
}
