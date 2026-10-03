import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type ParamSpec,
  type XYSeries,
} from 'aifn-render'
import { apply, eig2, type Mat2, type Vec2 } from '@/lib/math/mat2'

const R = 4
const ENTRY: ParamSpec = { min: -1.5, max: 1.5, step: 0.05 }
const snap = (v: number) => Math.round(Math.min(Math.max(v, -R), R) * 10) / 10

/** Repeated multiplication x, Ax, A²x, … drifts toward the eigenvector with the largest |λ|. */
export function MatrixPowers() {
  const pa = useParam(1.1, ENTRY)
  const pb = useParam(0.3, ENTRY)
  const pc = useParam(0.2, ENTRY)
  const pd = useParam(0.6, ENTRY)
  const k = useParam(8, { min: 1, max: 20, step: 1 })
  const [x, setX] = useState<Vec2>([-1.5, 2.5])

  const A: Mat2 = useMemo(
    () => [
      [pa.value, pb.value],
      [pc.value, pd.value],
    ],
    [pa.value, pb.value, pc.value, pd.value],
  )

  const r = useMemo(() => {
    const path: Vec2[] = [x]
    for (let i = 0; i < k.value; i++) path.push(apply(A, path[path.length - 1]))
    const eig = eig2(A)
    const series: XYSeries[] = []
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
    series.push({ name: 'Aᵏx', type: 'scatter', x: [path[k.value][0]], y: [path[k.value][1]], emphasis: true })
    return { path, eig, series }
  }, [A, x, k.value])

  const handles: Handle[] = [{ kind: 'point', at: x, label: 'x', onDrag: ([p, q]) => setX([snap(p), snap(q)]) }]
  const last = r.path[k.value]
  const eigText =
    r.eig.kind === 'real'
      ? `${formatNumber(r.eig.values[0])}, ${formatNumber(r.eig.values[1])}`
      : `${formatNumber(r.eig.re)} ± ${formatNumber(r.eig.im)}i`

  return (
    <Interactive
      title="Powers of a matrix follow its eigenvectors"
      caption="Set the entries of A = [[a, b], [c, d]] and drag the starting point x. The path joins x, Ax, A²x, … up to Aᵏx. Write x in the eigenbasis: each application multiplies the coordinate along eigenvector i by λᵢ, so the component with the largest |λ| takes over and the path turns towards that direction. Eigenvalues above 1 in size push the path out; below 1 pull it in. Complex eigenvalues have no real eigen-directions, and the path spirals."
      controls={
        <>
          <ParamSlider label="a" param={pa} />
          <ParamSlider label="b" param={pb} />
          <ParamSlider label="c" param={pc} />
          <ParamSlider label="d" param={pd} />
          <ParamSlider label="power k" param={k} />
        </>
      }
      readout={
        <>
          <Readout label="eigenvalues" value={eigText} />
          <Readout label="Aᵏx" value={`(${formatNumber(last[0])}, ${formatNumber(last[1])})`} />
          <Readout label="‖Aᵏx‖" value={formatNumber(Math.hypot(...last))} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-lg">
        <XYChart
          equalAspect
          xRange={[-R, R]}
          yRange={[-R, R]}
          xLabel="x₁"
          yLabel="x₂"
          series={r.series}
          handles={handles}
        />
      </div>
    </Interactive>
  )
}
