import { useMemo, useState } from 'react'
import { blobs } from 'aifn-applied/data/synthetic'
import { contourLines, convexHull, covarianceEllipse, evaluateGrid } from 'aifn/numerics/geometry'
import { lttb, minMaxDecimate } from '@lab/viz/drawing/decimate'
import { stream, uniform } from 'aifn/foundation/random'
import { linspace, toFlat, type Tensor } from 'aifn/foundation/tensor'
import { Select, Slider } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { Heatmap, Panel, Readout, Subplots, XYChart, type XYSeries } from '@lab/viz'

const columns = (t: Tensor) => {
  const v = toFlat(t)
  return { x: v.filter((_, i) => i % 2 === 0), y: v.filter((_, i) => i % 2 === 1) }
}

// ---------------------------------------------------------------------------------------------------------------------
// 1. Covariance ellipses and a convex hull.

const MASSES = [0.5, 0.9, 0.99]

export function EllipseSpecimen() {
  const [mean, setMean] = useState<[number, number]>([0, 0])
  const [sx, setSx] = useState(2)
  const [sy, setSy] = useState(1)
  const [rho, setRho] = useState(0.6)
  const cov = useMemo(
    () => [
      [sx * sx, rho * sx * sy],
      [rho * sx * sy, sy * sy],
    ],
    [sx, sy, rho],
  )
  const ellipses = MASSES.map((mass) => covarianceEllipse(mean, cov, { mass }))
  const series: XYSeries[] = ellipses.map((e, i) => ({
    name: `${MASSES[i] * 100}% mass`,
    type: 'line',
    ...columns(e.points),
    slot: i,
  }))
  return (
    <Figure
      title="Covariance ellipses at a chosen probability mass"
      description="The level sets (x − μ)ᵀ Σ⁻¹ (x − μ) = k² of a 2-D Gaussian with k = √(−2 ln(1 − mass)): their semi-axes are k√λ along the eigenvectors of Σ."
      controls={
        <ControlRow label="covariance">
          <Slider label="σ₁" value={sx} onChange={setSx} min={0.2} max={3} />
          <Slider label="σ₂" value={sy} onChange={setSy} min={0.2} max={3} />
          <Slider label="correlation ρ" value={rho} onChange={setRho} min={-0.95} max={0.95} />
        </ControlRow>
      }
      readouts={
        <>
          <Readout label="k at 90%" value={ellipses[1].k.toFixed(3)} />
          <Readout label="semi-axes (90%)" value={ellipses[1].radii.map((r) => r.toFixed(2)).join(', ')} />
          <Readout label="angle" value={`${((ellipses[1].angle * 180) / Math.PI).toFixed(1)}°`} />
        </>
      }
      caption="aifn/geometry covarianceEllipse; drag the centre to move the mean."
    >
      <XYChart
        series={series}
        live={[{ name: 'mean', type: 'scatter', x: [mean[0]], y: [mean[1]], emphasis: true }]}
        aspect="equal"
        xRange={[-8, 8]}
        yRange={[-6, 6]}
        rescaleOnChange={false}
        handles={[{ kind: 'point', at: mean, label: 'μ', onDrag: setMean }]}
        xLabel="x₁"
        yLabel="x₂"
      />
    </Figure>
  )
}

export function HullContourSpecimen() {
  const [level, setLevel] = useState(1)
  const data = useMemo(() => blobs(stream('hull'), { n: 60, centers: [[0, 0]], sd: 1 }), [])
  const hull = convexHull(data.x)
  const x = useMemo(() => linspace(-3, 3, 90), [])
  const z = useMemo(() => evaluateGrid((a, b) => Math.sin(a) * Math.cos(b) + 0.1 * a * a, x, x), [x])
  const lines = contourLines(x, x, z, level)
  const rows = useMemo(() => {
    const v = toFlat(z)
    return Array.from({ length: 90 }, (_, i) => v.slice(i * 90, i * 90 + 90))
  }, [z])
  const hullPts = columns(hull.points)
  const hullSeries: XYSeries[] = [
    { name: 'points', type: 'scatter', ...columns(data.x), slot: 0 },
    { name: 'hull', type: 'line', x: [...hullPts.x, hullPts.x[0]], y: [...hullPts.y, hullPts.y[0]], slot: 1 },
  ]
  return (
    <Figure
      title="Convex hull and contour lines"
      description="Left: the convex hull by Andrew's monotone chain; right: marching-squares segments joined into polylines at one level of a field."
      defaultSize="L"
      controls={<Slider label="contour level" value={level} onChange={setLevel} min={-0.8} max={1.8} />}
      readouts={
        <>
          <Readout label="hull vertices" value={hull.indices.shape[0]} />
          <Readout label="hull area" value={hull.area.toFixed(3)} />
          <Readout label="contour pieces" value={lines.length} />
        </>
      }
      caption="aifn/geometry convexHull, evaluateGrid and contourLines; f(x, y) = sin x cos y + 0.1x²."
    >
      <Subplots cols={2}>
        <Panel>
          <XYChart series={hullSeries} aspect="equal" xLabel="x₁" yLabel="x₂" />
        </Panel>
        <Panel>
          <Heatmap
            x={toFlat(x)}
            y={toFlat(x)}
            z={rows}
            xLabel="x"
            yLabel="y"
            fillOpacity={0.6}
            overlay={lines.map((l, i) => ({ name: `level ${i}`, type: 'line' as const, ...columns(l), slot: 1 }))}
          />
        </Panel>
      </Subplots>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 2. Decimation for drawing.

export function DecimationSpecimen() {
  const [points, setPoints] = useState(60)
  const [method, setMethod] = useState<'lttb' | 'minmax' | 'stride'>('lttb')
  const n = 5000
  const { xs, ys } = useMemo(() => {
    const s = stream('decimate')
    const xs = Array.from({ length: n }, (_, i) => i)
    let v = 0
    const ys = xs.map((i) => {
      v += uniform(s) - 0.5
      return v + (i === 3100 ? 12 : 0)
    })
    return { xs, ys }
  }, [])
  const kept = useMemo(() => {
    if (method === 'lttb') return lttb(xs, ys, points)
    if (method === 'minmax') return minMaxDecimate(xs, ys, Math.floor(points / 2) - 1)
    const idx = Array.from({ length: points }, (_, k) => Math.round((k * (n - 1)) / (points - 1)))
    return { x: idx, y: idx.map((i) => ys[i]) }
  }, [method, points, xs, ys])
  const kx = 'shape' in kept.x ? toFlat(kept.x) : kept.x
  const ky = 'shape' in kept.y ? toFlat(kept.y) : kept.y
  const series: XYSeries[] = [
    { name: `all ${n} points`, type: 'line', x: xs, y: ys, thin: true, slot: 0 },
    { name: `${kx.length} kept`, type: 'line', x: kx, y: ky, slot: 1, showPoints: true },
  ]
  return (
    <Figure
      title="Decimating a long line for drawing"
      description="Largest-triangle-three-buckets keeps the points that shape the line, so a spike survives decimation to 60 points; taking every k-th point loses it."
      controls={
        <ControlRow label="decimation">
          <Select
            label="method"
            value={method}
            onChange={setMethod}
            options={[
              { value: 'lttb', label: 'LTTB' },
              { value: 'minmax', label: 'min–max per bucket' },
              { value: 'stride', label: 'every k-th point' },
            ]}
          />
          <Slider
            label="points kept"
            value={points}
            onChange={(v) => setPoints(Math.round(v))}
            min={10}
            max={400}
            step={10}
          />
        </ControlRow>
      }
      caption="the lab's lttb and minMaxDecimate on a 5000-step random walk with one spike at t = 3100."
    >
      <XYChart series={series} xLabel="t" yLabel="value" rescaleOnChange={false} />
    </Figure>
  )
}
