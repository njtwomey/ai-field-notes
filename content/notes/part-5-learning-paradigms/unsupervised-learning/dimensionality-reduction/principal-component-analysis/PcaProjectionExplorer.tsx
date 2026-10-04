import { useMemo, useState } from 'react'
import {
  Figure,
  ControlGroup,
  NumberSelector,
  Plot,
  Points,
  Curve,
  Vectors,
  Handle,
  Readout,
  useAxis,
  type Vec2,
  type Vector,
} from 'aifn-render'
import { dataset } from 'aifn/learning/estimators'
import { gaussians } from 'aifn-methods/data/synthetic'
import { pca } from 'aifn-methods/unsupervised/embedding/linear'
import { stream } from 'aifn/foundation/random'
import { fromData, toFlat, toRows } from 'aifn/foundation/tensor'

export function PcaProjectionExplorer() {
  const [angle, setAngle] = useState(0.5)
  const [ratio, setRatio] = useState(0.3)
  const [query, setQuery] = useState<Vec2>([2, -1])

  const data = useMemo(() => {
    // Gaussian cloud with standard deviations 2 and 2*ratio along axes rotated by `angle`
    const c = Math.cos(angle)
    const s = Math.sin(angle)
    const [a, b] = [4, 4 * ratio * ratio]
    const cov = [
      [a * c * c + b * s * s, (a - b) * c * s],
      [(a - b) * c * s, a * s * s + b * c * c],
    ]
    return gaussians(stream('pca/2d/cloud'), {
      means: [[1, 0.5]],
      covariances: [cov],
      n: 200,
    })
  }, [angle, ratio])

  const full = useMemo(() => pca().fit(dataset(data.x)), [data])
  const one = useMemo(() => pca({ components: 1 }).fit(dataset(data.x)), [data])

  const mean = toFlat(full.mean) as Vec2
  const comps = toRows(full.components)
  const sd = toFlat(full.explainedVariance).map(Math.sqrt)

  const vectors: Vector[] = comps.map((c, j) => ({
    from: mean,
    to: [mean[0] + 2 * sd[j] * c[0], mean[1] + 2 * sd[j] * c[1]],
    slot: j + 2,
    label: `PC${j + 1}`,
  }))

  const q = fromData(Float64Array.from(query), [1, 2])
  const score = toFlat(one.transform(q))[0]
  const back = toFlat(one.inverseTransform(one.transform(q))) as Vec2

  const cloud = useMemo(() => {
    const rows = toRows(data.x)
    return { x: rows.map((r) => r[0]), y: rows.map((r) => r[1]) }
  }, [data])

  const x0 = useAxis({ label: 'x₀', range: [-6, 6] })
  const x1 = useAxis({ label: 'x₁', range: [-6, 6], equal: x0 })

  const reconstructionError = Math.hypot(query[0] - back[0], query[1] - back[1])

  return (
    <Figure
      title="PCA axes and orthogonal projection of a 2D cloud"
      purpose="The principal axes are the eigenvectors of the sample covariance matrix, scaled by standard deviation; orthogonal projection onto PC1 retains maximal variance while minimising squared residual error."
      defaultSize="L"
      controls={
        <ControlGroup title="Distribution shape & orientation">
          <NumberSelector
            label="orientation (rad)"
            value={angle}
            onChange={(v) => setAngle(Math.max(-1.5, Math.min(1.5, v)))}
            min={-1.5}
            max={1.5}
            step={0.1}
            suggestions={[-1.0, -0.5, 0, 0.5, 1.0]}
          />
          <NumberSelector
            label="minor / major spread"
            value={ratio}
            onChange={(v) => setRatio(Math.max(0.05, Math.min(1.0, v)))}
            min={0.05}
            max={1.0}
            step={0.05}
            suggestions={[0.1, 0.2, 0.3, 0.5, 0.8, 1.0]}
          />
        </ControlGroup>
      }
      readouts={
        <>
          <Readout
            label="explained variance ratio (PC1, PC2)"
            value={toFlat(full.explainedVarianceRatio)
              .map((v) => v.toFixed(3))
              .join(', ')}
          />
          <Readout label="PC1 score of query q" value={score.toFixed(3)} />
          <Readout label="reconstruction error ‖q - q̂‖₂" value={reconstructionError.toFixed(3)} />
        </>
      }
      caption="Principal component arrows run two standard deviations (2σ) along each eigenvector starting from the empirical center. Drag the query point q: its orthogonal projection onto PC1 (the 1-component reconstruction q̂) connects back to q via the dashed residual line. Notice that as the spread ratio approaches 1 (isotropic circular data), the variance ratio approaches 0.5 and the axes lose any preferred direction."
    >
      <Plot x={x0} y={x1}>
        <Points name="data points" x={cloud.x} y={cloud.y} muted />
        <Vectors vectors={vectors} />
        <Curve name="projection residual" x={[query[0], back[0]]} y={[query[1], back[1]]} dashed slot={1} />
        <Points name="reconstruction" x={[back[0]]} y={[back[1]]} slot={1} emphasis />
        <Handle kind="point" at={query} label="query q" onDrag={setQuery} />
      </Plot>
    </Figure>
  )
}
