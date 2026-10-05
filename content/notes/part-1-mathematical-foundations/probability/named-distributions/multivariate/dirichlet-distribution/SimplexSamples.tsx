import { useMemo } from 'react'
import {
  Figure,
  float,
  formatNumber,
  Plot,
  Points,
  Readout,
  type Segment,
  Segments,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { normal, stream, type Stream, uniform } from 'aifn-compute/foundation/random'

const N = 600
const H = Math.sqrt(3) / 2
// Vertices of the probability simplex drawn as an equilateral triangle: θ = (1,0,0), (0,1,0), (0,0,1).
const VERTICES: [number, number][] = [
  [0, 0],
  [1, 0],
  [0.5, H],
]
const OUTLINE: Segment[] = VERTICES.map((v, i) => ({ from: v, to: VERTICES[(i + 1) % 3] }))
const X_RANGE: [number, number] = [-0.05, 1.05]
const Y_RANGE: [number, number] = [-0.05, H + 0.05]

/** Gamma(α, 1) draw by Marsaglia and Tsang's method, boosted for α < 1. */
function gammaSample(alpha: number, r: Stream): number {
  if (alpha < 1) return gammaSample(alpha + 1, r) * Math.pow(Math.max(uniform(r), 1e-300), 1 / alpha)
  const d = alpha - 1 / 3
  const c = 1 / Math.sqrt(9 * d)
  for (;;) {
    const z = normal(r)
    const v = (1 + c * z) ** 3
    if (v <= 0) continue
    const u = uniform(r)
    if (Math.log(u) < 0.5 * z * z + d - d * v + d * Math.log(v)) return d * v
  }
}

const toPlane = (t: number[]): [number, number] => [
  t[0] * VERTICES[0][0] + t[1] * VERTICES[1][0] + t[2] * VERTICES[2][0],
  t[0] * VERTICES[0][1] + t[1] * VERTICES[1][1] + t[2] * VERTICES[2][1],
]

/** Samples from a three-category Dirichlet, drawn on the triangle of probability vectors. */
export function SimplexSamples() {
  const state = useFigureState({
    a1: float(2, { min: 0.1, max: 20, step: 0.1, label: 'α₁' }),
    a2: float(2, { min: 0.1, max: 20, step: 0.1, label: 'α₂' }),
    a3: float(2, { min: 0.1, max: 20, step: 0.1, label: 'α₃' }),
  })

  const series = useMemo(() => {
    const alpha = [state.a1, state.a2, state.a3]
    const r = stream(11)
    const points = Array.from({ length: N }, () => {
      const g = alpha.map((a) => gammaSample(a, r))
      const total = g[0] + g[1] + g[2]
      return toPlane(g.map((x) => x / total))
    })
    const total = state.a1 + state.a2 + state.a3
    const mean = toPlane(alpha.map((a) => a / total))
    return [
      { name: 'samples', x: points.map((p) => p[0]), y: points.map((p) => p[1]), slot: 0 },
      { name: 'mean', x: [mean[0]], y: [mean[1]], emphasis: true },
    ] as const
  }, [state.a1, state.a2, state.a3])

  const a0 = state.a1 + state.a2 + state.a3
  const xAxis = useAxis({ range: X_RANGE })
  const yAxis = useAxis({ range: Y_RANGE, equal: xAxis })
  return (
    <Figure
      title="Dirichlet samples on the simplex"
      state={state}
      caption="Each point is a probability vector θ = (θ₁, θ₂, θ₃). The bottom-left corner is θ = (1, 0, 0), the bottom-right (0, 1, 0) and the top (0, 0, 1). Equal α below 1 pushes samples to the corners and edges (sparse vectors); α = 1 is uniform on the triangle; large α concentrates them around the mean α/α₀."

      readouts={
        <>
          <Readout label="concentration α₀" value={formatNumber(a0)} />
          <Readout
            label="mean θ"
            value={`(${[state.a1, state.a2, state.a3].map((a) => formatNumber(a / a0)).join(', ')})`}
          />
        </>
      }
    >
      {/* Equal aspect sets the height from the width, so cap the width: the whole simplex and the sliders fit on screen. */}
      <div className="mx-auto w-full max-w-md">
        <Plot x={xAxis} y={yAxis} bare>
          <Points {...series[0]} />
          <Points {...series[1]} />
          <Segments segments={OUTLINE} />
        </Plot>
      </div>
    </Figure>
  )
}
