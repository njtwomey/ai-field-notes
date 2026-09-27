import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import { linspace } from '@/lib/math'
import { jsDiv, klDiv, tvDist } from '../_shared/ot'

type Shape = 'uniform' | 'gaussian'

const GRID = linspace(-5, 9, 1401)
const DX = GRID[1] - GRID[0]
const THETAS = linspace(0, 3, 61)
const DENSITY_X: [number, number] = [-3, 6]
const CURVE_X: [number, number] = [0, 3]

function density(shape: Shape, shift: number): number[] {
  return GRID.map((x) => {
    const z = x - shift
    if (shape === 'uniform') return z >= 0 && z < 1 ? 1 : 0
    return Math.exp(-0.5 * (z / 0.3) ** 2) / (0.3 * Math.sqrt(2 * Math.PI))
  })
}

/**
 * Divergences between the base density and its copy shifted by theta. Closed forms for the uniform (overlap 1 − θ);
 * probabilities of grid cells for the Gaussian.
 */
function divergences(shape: Shape, theta: number) {
  if (shape === 'uniform') {
    const gap = Math.min(theta, 1)
    return { tv: gap, js: gap * Math.LN2, kl: theta > 0 ? Infinity : 0 }
  }
  const p = density(shape, 0).map((v) => v * DX)
  const q = density(shape, theta).map((v) => v * DX)
  return { tv: tvDist(p, q), js: jsDiv(p, q), kl: klDiv(p, q) }
}

/**
 * A distribution and a shifted copy. Every Wasserstein distance between them equals the shift; total variation and
 * Jensen–Shannon saturate once the supports stop overlapping, and KL is infinite as soon as they differ in support.
 */
export function DisjointSupports() {
  const theta = useParam(0.6, { min: 0, max: 3, step: 0.05 })
  const [shape, setShape] = useState<Shape>('uniform')

  const curves = useMemo(() => {
    const d = THETAS.map((t) => divergences(shape, t))
    return { tv: d.map((v) => v.tv), js: d.map((v) => v.js) }
  }, [shape])

  const now = useMemo(() => divergences(shape, theta.value), [shape, theta.value])

  const densities = useMemo((): XYSeries[] => {
    const keep = GRID.map((x, i) => [x, i] as const).filter(([x]) => x >= DENSITY_X[0] && x <= DENSITY_X[1])
    const p = density(shape, 0)
    const q = density(shape, theta.value)
    const xs = keep.map(([x]) => x)
    return [
      { name: 'p', type: 'line', x: xs, y: keep.map(([, i]) => p[i]), slot: 0, area: true },
      { name: 'q, shifted by θ', type: 'line', x: xs, y: keep.map(([, i]) => q[i]), slot: 1, area: true },
    ]
  }, [shape, theta.value])

  const curveSeries = useMemo(
    (): XYSeries[] => [
      { name: 'W₁ = W₂ = θ', type: 'line', x: THETAS, y: THETAS, slot: 2 },
      { name: 'total variation', type: 'line', x: THETAS, y: curves.tv, slot: 3 },
      { name: 'Jensen–Shannon (nats)', type: 'line', x: THETAS, y: curves.js, slot: 4 },
    ],
    [curves],
  )

  const handles: Handle[] = [{ kind: 'x', at: theta.value, onDrag: theta.set, label: 'θ' }]

  return (
    <Interactive
      title="Shifting a distribution: Wasserstein against bin-wise divergences"
      caption="The right-hand distribution is the left one shifted by θ. Move the slider, or drag the vertical line on the right chart. The Wasserstein distance grows linearly with the shift. For the uniform shape, total variation and Jensen–Shannon stop changing at θ = 1, where the supports separate, and KL is infinite for every θ > 0. For the narrow Gaussian KL is finite, θ²/(2σ²), but TV and JS still saturate."
      controls={
        <>
          <ParamSlider label="shift θ" param={theta} />
          <ParamChoice
            label="shape"
            value={shape}
            onChange={setShape}
            options={[
              { value: 'uniform', label: 'uniform on [0, 1]' },
              { value: 'gaussian', label: 'Gaussian, σ = 0.3' },
            ]}
          />
        </>
      }
      readout={
        <>
          <Readout label="W₁" value={formatNumber(theta.value)} />
          <Readout label="TV" value={formatNumber(now.tv)} />
          <Readout label="JS (max log 2 = 0.693)" value={formatNumber(now.js)} />
          <Readout label="KL(p ‖ q)" value={Number.isFinite(now.kl) ? formatNumber(now.kl) : '∞'} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart
          height={280}
          series={densities}
          xRange={DENSITY_X}
          yRange={[0, undefined]}
          xLabel="x"
          yLabel="density"
        />
        <XYChart
          height={280}
          series={curveSeries}
          xRange={CURVE_X}
          yRange={[0, 3]}
          xLabel="shift θ"
          yLabel="distance"
          handles={handles}
        />
      </div>
    </Interactive>
  )
}
