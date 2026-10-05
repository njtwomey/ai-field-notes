import { useMemo } from 'react'
import { choice, Figure, int, Plot, Readout, seriesLayers, type SeriesSpec, useAxis, useFigureState } from 'aifn-render'
import { stream, uniform } from 'aifn-compute/foundation/random'

type Method = 'uniforms' | 'exponentials' | 'spacings'

const METHODS: { value: Method; label: string }[] = [
  { value: 'uniforms', label: 'normalised uniforms (wrong)' },
  { value: 'exponentials', label: 'normalised exponentials' },
  { value: 'spacings', label: 'sorted-uniform spacings' },
]

type Draw = [number, number, number]

function draw(method: Method, uniform: () => number): Draw {
  const u = () => Math.max(uniform(), 1e-12)
  if (method === 'spacings') {
    const [a, b] = [u(), u()].sort((p, q) => p - q)
    return [a, b - a, 1 - b]
  }
  const w = method === 'uniforms' ? [u(), u(), u()] : [-Math.log(u()), -Math.log(u()), -Math.log(u())]
  const s = w[0] + w[1] + w[2]
  return [w[0] / s, w[1] / s, w[2] / s]
}

/** The triangle with corners e₁ = (0, 0), e₂ = (1, 0), e₃ = (1/2, √3/2): barycentric weights to the plane. */
const toPlane = ([, p2, p3]: Draw): [number, number] => [p2 + p3 / 2, (p3 * Math.sqrt(3)) / 2]

const outline = (corners: Draw[], name: string): SeriesSpec => {
  const pts = [...corners, corners[0]].map(toPlane)
  return { name, type: 'line', x: pts.map((p) => p[0]), y: pts.map((p) => p[1]), muted: true }
}

/**
 * Points on the 2-simplex {p ≥ 0, p₁ + p₂ + p₃ = 1}, drawn in barycentric coordinates. The inner triangle, where every
 * weight is at most 1/2, has a quarter of the area.
 */
export function SimplexSampler() {
  const state = useFigureState({
    method: choice<Method>(METHODS, 'uniforms', { label: 'method' }),
    n: int(2000, { min: 200, max: 4000, step: 100, suggestions: [500, 1000, 2000, 4000], label: 'points' }),
  })

  const { points, middle } = useMemo(() => {
    const r = stream(9)
    const draws = Array.from({ length: state.n }, () => draw(state.method, () => uniform(r)))
    const plane = draws.map(toPlane)
    return {
      points: { name: 'samples', type: 'scatter', x: plane.map((p) => p[0]), y: plane.map((p) => p[1]), slot: 0 },
      middle: draws.filter((p) => Math.max(...p) <= 0.5).length / state.n,
    } satisfies { points: SeriesSpec; middle: number }
  }, [state.method, state.n])

  const guides = useMemo(
    () => [
      outline(
        [
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
        ],
        'simplex',
      ),
      outline(
        [
          [0.5, 0.5, 0],
          [0, 0.5, 0.5],
          [0.5, 0, 0.5],
        ],
        'every pᵢ ≤ 1/2',
      ),
    ],
    [],
  )

  const xAxis = useAxis({ range: [-0.05, 1.05] })
  const yAxis = useAxis({ range: [-0.05, 0.92], equal: xAxis })
  return (
    <Figure
      title="Three ways to split one into three"
      state={state}
      caption="Each point is a probability vector (p₁, p₂, p₃), drawn in the triangle whose corners are the three one-hot vectors. The inner triangle, where no weight exceeds 1/2, covers a quarter of the area, so a uniform sampler puts 25% of its points there. Normalising uniform numbers crowds the centre. Normalising exponential numbers, or cutting the unit interval at sorted uniform points, is exactly uniform."

      readouts={
        <>
          <Readout label="inside the inner triangle" value={`${(100 * middle).toFixed(1)}%`} />
          <Readout label="uniform target" value="25%" />
        </>
      }
    >
      <div className="mx-auto w-full max-w-xl">
        <Plot x={xAxis} y={yAxis} bare>
          {seriesLayers([points, ...guides])}
        </Plot>
      </div>
    </Figure>
  )
}
