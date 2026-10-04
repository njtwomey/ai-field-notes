import { useMemo, useState } from 'react'
import { wasserstein1d } from 'aifn/transport'
import { normals, stream } from 'aifn/foundation/random'
import { toFlat } from 'aifn/foundation/tensor'
import {
  Area,
  ControlRow,
  Curve,
  Figure,
  formatNumber,
  Plot,
  Plots,
  Points,
  Readout,
  Select,
  Slider,
  useAxis,
} from 'aifn-render'

export function WassersteinExplorer() {
  const [shift, setShift] = useState(1.0)
  const [scale, setScale] = useState(1.5)
  const [sampleSize, setSampleSize] = useState(200)

  const n = sampleSize
  const levels = useMemo(() => Array.from({ length: n }, (_, i) => (i + 0.5) / n), [n])

  const u = useMemo(
    () => toFlat(normals(stream('w1-u-explorer'), n)).sort((a, b) => a - b),
    [n],
  )
  const v = useMemo(
    () =>
      toFlat(normals(stream('w1-v-explorer'), n))
        .map((z) => shift + scale * z)
        .sort((a, b) => a - b),
    [n, shift, scale],
  )

  const w1 = useMemo(() => wasserstein1d(u, v), [u, v])
  const w2 = useMemo(() => wasserstein1d(u, v, { p: 2 }), [u, v])
  const w2Gaussian = Math.sqrt(shift ** 2 + (scale - 1) ** 2)

  const qAxis = useAxis({ label: 'quantile level q ∈ [0, 1]', range: [0, 1] })
  const valAxis = useAxis({ label: 'value x' })
  const uAxis = useAxis({ label: 'source sample u_(i)' })
  const vAxis = useAxis({ label: 'matched target v_(i)' })

  return (
    <Figure
      title="1D Wasserstein distance & quantile matching"
      purpose="Visualize the 1D optimal transport problem where the optimal plan is the monotone quantile map and W₁ equals the integrated area between inverse CDFs."
      caption="On the real line, the optimal transport plan is uniquely given by the monotone coupling matching quantiles. Left: quantile functions F⁻¹(q) for source and target distributions; W₁ is exactly the shaded area between the two curves: ∫₀¹ |F_u⁻¹(q) - F_v⁻¹(q)| dq. Right: optimal transport map pairing each sorted source point u_(i) directly with v_(i)."
    >
      <ControlRow>
        <Slider
          label="Target mean shift"
          value={shift}
          min={-2.5}
          max={2.5}
          step={0.1}
          onChange={setShift}
        />
        <Slider
          label="Target scale σ"
          value={scale}
          min={0.3}
          max={2.5}
          step={0.1}
          onChange={setScale}
        />
        <Select
          label="Sample size N"
          value={String(sampleSize)}
          onChange={(val) => setSampleSize(Number(val))}
          options={[
            { value: '100', label: '100 samples' },
            { value: '200', label: '200 samples' },
            { value: '400', label: '400 samples' },
          ]}
        />
      </ControlRow>

      <div className="flex flex-wrap gap-4 text-xs font-mono text-muted-foreground my-2">
        <Readout label="W₁ (area)" value={formatNumber(w1)} />
        <Readout label="W₂ (empirical)" value={formatNumber(w2)} />
        <Readout label="W₂ (Gaussian truth)" value={formatNumber(w2Gaussian)} />
        <Readout label="Δμ" value={formatNumber(shift)} />
        <Readout label="ratio σ/σ₀" value={formatNumber(scale)} />
      </div>

      <Plots cols={2}>
        <Plot x={qAxis} y={valAxis} title="Quantile functions F⁻¹(q) & W₁ area">
          <Area
            name="|F_u⁻¹ − F_v⁻¹| (W₁ area)"
            x={levels}
            y={v}
            base={u}
            line={false}
            opacity={0.2}
          />
          <Curve name="F_u⁻¹(q) (Source N(0, 1))" x={levels} y={u} slot={0} />
          <Curve name="F_v⁻¹(q) (Target N(μ, σ²))" x={levels} y={v} slot={1} />
        </Plot>

        <Plot x={uAxis} y={vAxis} title="Optimal Monotone Transport Map T(u)">
          <Curve name="Monotone assignment" x={u} y={v} slot={1} />
          <Points name="Coupled pairs (u_i, v_i)" x={u} y={v} slot={0} size={5} />
        </Plot>
      </Plots>
    </Figure>
  )
}
