import { useMemo } from 'react'
import {
  Area,
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  Plot,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { jsDiv, klDiv, tvDist } from '../_shared/ot'
import { linspace, toFlat } from 'aifn/foundation/tensor'

type Shape = 'uniform' | 'gaussian'

const GRID = toFlat(linspace(-5, 9, 1401))
const DX = GRID[1] - GRID[0]
const THETAS = toFlat(linspace(0, 3, 61))
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
  const state = useFigureState({
    theta: float(0.6, { min: 0, max: 3, step: 0.05, label: 'shift θ' }),
    shape: choice<Shape>(
      [
        { value: 'uniform', label: 'uniform on [0, 1]' },
        { value: 'gaussian', label: 'Gaussian, σ = 0.3' },
      ],
      'uniform',
      { label: 'shape' },
    ),
  })

  const curves = useMemo(() => {
    const d = THETAS.map((t) => divergences(state.shape, t))
    return { tv: d.map((v) => v.tv), js: d.map((v) => v.js) }
  }, [state.shape])

  const now = useMemo(() => divergences(state.shape, state.theta), [state.shape, state.theta])

  const densities = useMemo(() => {
    const keep = GRID.map((x, i) => [x, i] as const).filter(([x]) => x >= DENSITY_X[0] && x <= DENSITY_X[1])
    const p = density(state.shape, 0)
    const q = density(state.shape, state.theta)
    const xs = keep.map(([x]) => x)
    return [
      { name: 'p', x: xs, y: keep.map(([, i]) => p[i]), slot: 0 },
      { name: 'q, shifted by θ', x: xs, y: keep.map(([, i]) => q[i]), slot: 1 },
    ] as const
  }, [state.shape, state.theta])

  const curveSeries = useMemo(
    () =>
      [
        { name: 'W₁ = W₂ = θ', x: THETAS, y: THETAS, slot: 2 },
        { name: 'total variation', x: THETAS, y: curves.tv, slot: 3 },
        { name: 'Jensen–Shannon (nats)', x: THETAS, y: curves.js, slot: 4 },
      ] as const,
    [curves],
  )

  const xAxis = useAxis({ label: 'x', range: DENSITY_X })
  const yAxis = useAxis({ label: 'density', range: [0, undefined], hold: 'union' })
  const xAxis2 = useAxis({ label: 'shift θ', range: CURVE_X })
  const yAxis2 = useAxis({ label: 'distance', range: [0, 3] })
  return (
    <Figure
      title="Shifting a distribution: Wasserstein against bin-wise divergences"
      state={state}
      caption="The right-hand distribution is the left one shifted by θ. Move the slider, or drag the vertical line on the right chart. The Wasserstein distance grows linearly with the shift. For the uniform shape, total variation and Jensen–Shannon stop changing at θ = 1, where the supports separate, and KL is infinite for every θ > 0. For the narrow Gaussian KL is finite, θ²/(2σ²), but TV and JS still saturate."

      readouts={
        <>
          <Readout label="W₁" value={formatNumber(state.theta)} />
          <Readout label="TV" value={formatNumber(now.tv)} />
          <Readout label="JS (max log 2 = 0.693)" value={formatNumber(now.js)} />
          <Readout label="KL(p ‖ q)" value={Number.isFinite(now.kl) ? formatNumber(now.kl) : '∞'} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={280}>
          <Area {...densities[0]} />
          <Area {...densities[1]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={280}>
          <Curve {...curveSeries[0]} />
          <Curve {...curveSeries[1]} />
          <Curve {...curveSeries[2]} />
          <Handle {...state.handle('theta', { label: 'θ' })} />
        </Plot>
      </div>
    </Figure>
  )
}
