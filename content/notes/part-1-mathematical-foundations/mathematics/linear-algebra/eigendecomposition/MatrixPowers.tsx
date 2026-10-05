import { useMemo } from 'react'
import {
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { apply2, eig2, type Mat2, type Vec2 } from 'aifn-compute/numerics/linalg'

const R = 4
const entry = (initial: number, label: string) => slider(-1.5, 1.5, initial, { step: 0.05, label })
const start = (initial: number) => slider(-R, R, initial, { step: 0.1, onChart: true })

/** Repeated multiplication x, Ax, A²x, … drifts toward the eigenvector with the largest |λ|. */
export function MatrixPowers() {
  const state = useFigureState({
    a: entry(1.1, 'a'),
    b: entry(0.3, 'b'),
    c: entry(0.2, 'c'),
    d: entry(0.6, 'd'),
    k: int(8, { min: 1, max: 20, step: 1, label: 'power k' }),
    x1: start(-1.5),
    x2: start(2.5),
  })
  const x: Vec2 = useMemo(() => [state.x1, state.x2], [state.x1, state.x2])

  const A: Mat2 = useMemo(
    () => [
      [state.a, state.b],
      [state.c, state.d],
    ],
    [state.a, state.b, state.c, state.d],
  )

  const r = useMemo(() => {
    const path: Vec2[] = [x]
    for (let i = 0; i < state.k; i++) path.push(apply2(A, path[path.length - 1]))
    const eig = eig2(A)
    const series: SeriesSpec[] = []
    if (eig.kind === 'real') {
      eig.vectors.forEach((v, i) =>
        series.push({
          name: `eigenvector ${i + 1} direction`,
          type: 'line',
          x: [-2 * R * v[0], 2 * R * v[0]],
          y: [-2 * R * v[1], 2 * R * v[1]],
          slot: i,
          dashed: true,
        }),
      )
    }
    series.push(
      { name: 'x, Ax, A²x, …', type: 'line', x: path.map((p) => p[0]), y: path.map((p) => p[1]), slot: 2 },
      { name: 'iterates', type: 'scatter', x: path.map((p) => p[0]), y: path.map((p) => p[1]), slot: 2 },
    )
    series.push({ name: 'Aᵏx', type: 'scatter', x: [path[state.k][0]], y: [path[state.k][1]], emphasis: true })
    return { path, eig, series }
  }, [A, x, state.k])

  const last = r.path[state.k]
  const eigText =
    r.eig.kind === 'real'
      ? `${formatNumber(r.eig.values[0])}, ${formatNumber(r.eig.values[1])}`
      : `${formatNumber(r.eig.re)} ± ${formatNumber(r.eig.im)}i`

  const xAxis = useAxis({ label: 'x₁', range: [-R, R] })
  const yAxis = useAxis({ label: 'x₂', range: [-R, R], equal: xAxis })
  return (
    <Figure
      title="Powers of a matrix follow its eigenvectors"
      state={state}
      caption="Set the entries of A = [[a, b], [c, d]] and drag the starting point x. The path joins x, Ax, A²x, … up to Aᵏx. Write x in the eigenbasis: each application multiplies the coordinate along eigenvector i by λᵢ, so the component with the largest |λ| takes over and the path turns towards that direction. Eigenvalues above 1 in size push the path out; below 1 pull it in. Complex eigenvalues have no real eigen-directions, and the path spirals."
      readouts={
        <>
          <Readout label="eigenvalues" value={eigText} />
          <Readout label="Aᵏx" value={`(${formatNumber(last[0])}, ${formatNumber(last[1])})`} />
          <Readout label="‖Aᵏx‖" value={formatNumber(Math.hypot(...last))} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-lg">
        <Plot x={xAxis} y={yAxis}>
          {seriesLayers(r.series)}
          <Handle {...state.handle(['x1', 'x2'], { label: 'x' })} />
        </Plot>
      </div>
    </Figure>
  )
}
