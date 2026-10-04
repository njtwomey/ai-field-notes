import { useMemo } from 'react'
import { Figure, float, formatNumber, Plot, Points, Readout, useAxis, useFigureState } from 'aifn-render'
import { rings } from '../../_shared/datasets'
import { distances, doubleCentre, linearSplitAccuracy, topScaled } from '../../_shared/manifold'

const N = 150

export function KernelPcaRings() {
  const data = useMemo(() => {
    const { points, labels } = rings(N, 0.3, 0.05, 4)
    return { points, labels, d: distances(points) }
  }, [])
  const state = useFigureState({
    sigma: float(0.35, { min: 0.05, max: 2, step: 0.05, label: 'σ (kernel width)' }),
  })
  const fit = useMemo(() => {
    const kern = data.d.map((row) => row.map((v) => Math.exp(-(v * v) / (2 * state.sigma * state.sigma))))
    const { y, values } = topScaled(doubleCentre(kern))
    const total = values.reduce((s, v) => s + Math.max(v, 0), 0)
    return { y, share: (values[0] + values[1]) / total, accuracy: linearSplitAccuracy(y, data.labels) }
  }, [data, state.sigma])
  const inputAccuracy = useMemo(() => linearSplitAccuracy(data.points, data.labels), [data])
  const names = ['outer ring', 'inner ring']

  const xAxis = useAxis({ label: 'x₁', range: [-1.3, 1.3] })
  const yAxis = useAxis({ label: 'x₂', range: [-1.3, 1.3], equal: xAxis })
  const xAxis2 = useAxis({ label: 'kernel PC 1', hold: 'union' })
  const yAxis2 = useAxis({ label: 'kernel PC 2', hold: 'union' })
  return (
    <Figure
      title="Kernel PCA separates two rings"
      state={state}
      caption="Left: two concentric rings in the plane; no straight line separates them. Right: the first two kernel principal components with a Gaussian kernel of width σ. For σ from 0.2 to 0.5 the rings fall on opposite sides of a line, and the best linear split is perfect. A very narrow kernel treats each small arc as its own feature; a very wide kernel is almost linear in the squared distances and returns a picture much like the input."

      readouts={
        <>
          <Readout label="best linear split, input" value={formatNumber(inputAccuracy)} />
          <Readout label="best linear split, kernel PCs 1–2" value={formatNumber(fit.accuracy)} />
          <Readout label="share of feature-space variance in PCs 1–2" value={formatNumber(fit.share)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis}>
          <Points
            name="data"
            x={data.points.map((p) => p[0])}
            y={data.points.map((p) => p[1])}
            group={data.labels}
            groupNames={names}
          />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={340}>
          <Points
            name="kernel PCA"
            x={fit.y.map((p) => p[0])}
            y={fit.y.map((p) => p[1])}
            group={data.labels}
            groupNames={names}
          />
        </Plot>
      </div>
    </Figure>
  )
}
