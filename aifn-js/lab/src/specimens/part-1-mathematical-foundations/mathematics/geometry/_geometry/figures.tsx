import { useMemo } from 'react'
import { blobs } from 'aifn-applied/data/synthetic'
import { contourLines, convexHull, covarianceEllipse, evaluateGrid } from 'aifn/numerics/geometry'
import { lttb, minMaxDecimate } from '@lab/viz/drawing/decimate'
import { stream, uniform } from 'aifn/foundation/random'
import { linspace, toFlat, type Tensor } from 'aifn/foundation/tensor'
import { Figure } from '@lab/layout'
import { choice, row, slider, useFigureState } from '@lab/state'
import { Curve, Handle, Plot, Plots, Points, Raster, Readout, useAxis } from '@lab/viz'

const columns = (t: Tensor) => {
  const v = toFlat(t)
  return { x: v.filter((_, i) => i % 2 === 0), y: v.filter((_, i) => i % 2 === 1) }
}

// ---------------------------------------------------------------------------------------------------------------------
// 1. Covariance ellipses and a convex hull.

const MASSES = [0.5, 0.9, 0.99]

export function EllipseSpecimen() {
  const state = useFigureState({
    cov: row('covariance', {
      sx: slider(0.2, 3, 2, { label: 'σ₁' }),
      sy: slider(0.2, 3, 1, { label: 'σ₂' }),
      rho: slider(-0.95, 0.95, 0.6, { label: 'correlation ρ' }),
    }),
    mx: slider(-6, 6, 0, { onChart: true }),
    my: slider(-5, 5, 0, { onChart: true }),
  })
  const { sx, sy, rho } = state.cov
  const mean = useMemo((): [number, number] => [state.mx, state.my], [state.mx, state.my])
  const ellipses = useMemo(() => {
    const cov = [
      [sx * sx, rho * sx * sy],
      [rho * sx * sy, sy * sy],
    ]
    return MASSES.map((mass) => {
      const e = covarianceEllipse(mean, cov, { mass })
      return { ...e, ...columns(e.points) }
    })
  }, [mean, sx, sy, rho])
  const x = useAxis({ label: 'x₁', range: [-8, 8] })
  const y = useAxis({ label: 'x₂', range: [-6, 6], equal: x })
  return (
    <Figure
      title="Covariance ellipses at a chosen probability mass"
      purpose="The level sets (x − μ)ᵀ Σ⁻¹ (x − μ) = k² of a 2-D Gaussian with k = √(−2 ln(1 − mass)): their semi-axes are k√λ along the eigenvectors of Σ."
      state={state}
      readouts={
        <>
          <Readout label="k at 90%" value={ellipses[1].k.toFixed(3)} />
          <Readout label="semi-axes (90%)" value={ellipses[1].radii.map((r) => r.toFixed(2)).join(', ')} />
          <Readout label="angle" value={`${((ellipses[1].angle * 180) / Math.PI).toFixed(1)}°`} />
        </>
      }
      caption="Drag the centre to move the mean; the sliders set Σ. The three ellipses hold 50%, 90% and 99% of the mass: k = 1.18, 2.15 and 3.03, so each is the same ellipse scaled. With ρ = 0 the axes line up with the coordinates."
    >
      <Plot x={x} y={y}>
        {ellipses.map((e, i) => (
          <Curve key={i} name={`${MASSES[i] * 100}% mass`} x={e.x} y={e.y} slot={i} live />
        ))}
        <Handle {...state.handle(['mx', 'my'], { label: 'μ' })} />
      </Plot>
    </Figure>
  )
}

export function HullContourSpecimen() {
  const state = useFigureState({ level: slider(-0.8, 1.8, 1, { label: 'contour level' }) })
  const level = state.level
  const data = useMemo(() => blobs(stream('hull'), { n: 60, centers: [[0, 0]], sd: 1 }), [])
  const hull = useMemo(() => convexHull(data.x), [data])
  const xg = useMemo(() => linspace(-3, 3, 90), [])
  const z = useMemo(() => evaluateGrid((a, b) => Math.sin(a) * Math.cos(b) + 0.1 * a * a, xg, xg), [xg])
  const lines = useMemo(() => contourLines(xg, xg, z, level).map(columns), [xg, z, level])
  const rows = useMemo(() => {
    const v = toFlat(z)
    return Array.from({ length: 90 }, (_, i) => v.slice(i * 90, i * 90 + 90))
  }, [z])
  const gridX = useMemo(() => toFlat(xg), [xg])
  const pts = useMemo(() => columns(data.x), [data])
  const ring = useMemo(() => {
    const h = columns(hull.points)
    return { x: [...h.x, h.x[0]], y: [...h.y, h.y[0]] }
  }, [hull])
  const ax = useAxis({ label: 'x₁' })
  const ay = useAxis({ label: 'x₂', equal: ax })
  const bx = useAxis({ label: 'x' })
  const by = useAxis({ label: 'y', equal: bx })
  return (
    <Figure
      title="Convex hull and contour lines"
      purpose="The convex hull is the smallest convex polygon holding every point (Andrew's monotone chain); a contour is the polyline where a field crosses one level, joined from marching-squares segments."
      defaultSize="L"
      state={state}
      readouts={
        <>
          <Readout label="hull vertices" value={hull.indices.shape[0]} />
          <Readout label="hull area" value={hull.area.toFixed(3)} />
          <Readout label="contour pieces" value={lines.length} />
        </>
      }
      caption="Right: f(x, y) = sin x cos y + 0.1x² with its level set at the chosen level. Move the level: pieces appear, merge and split where the level passes a saddle of f."
    >
      <Plots cols={2}>
        <Plot x={ax} y={ay} title="convex hull">
          <Points name="points" x={pts.x} y={pts.y} slot={0} />
          <Curve name="hull" x={ring.x} y={ring.y} slot={1} />
        </Plot>
        <Plot x={bx} y={by} title="contour of f">
          <Raster x={gridX} y={gridX} z={rows} valueLabel="f(x, y)" fillOpacity={0.7} />
          {lines.map((l, i) => (
            <Curve key={i} name="contour" x={l.x} y={l.y} emphasis />
          ))}
        </Plot>
      </Plots>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 2. Decimation for drawing.

const DECIMATORS = [
  { value: 'lttb', label: 'LTTB' },
  { value: 'minmax', label: 'min–max per bucket' },
  { value: 'stride', label: 'every k-th point' },
] as const

export function DecimationSpecimen() {
  const state = useFigureState({
    decimation: row('decimation', {
      method: choice(DECIMATORS, 'stride', { label: 'method' }),
      points: slider(10, 400, 60, { step: 10, label: 'points kept' }),
    }),
  })
  const { method, points } = state.decimation
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
    const k =
      method === 'lttb'
        ? lttb(xs, ys, points)
        : method === 'minmax'
          ? minMaxDecimate(xs, ys, Math.floor(points / 2) - 1)
          : (() => {
              const idx = Array.from({ length: points }, (_, j) => Math.round((j * (n - 1)) / (points - 1)))
              return { x: idx, y: idx.map((i) => ys[i]) }
            })()
    return {
      x: 'shape' in k.x ? toFlat(k.x) : (k.x as number[]),
      y: 'shape' in k.y ? toFlat(k.y) : (k.y as number[]),
    }
  }, [method, points, xs, ys])
  const x = useAxis({ label: 't' })
  const y = useAxis({ label: 'value', hold: 'initial' })
  return (
    <Figure
      title="Decimating a long line for drawing"
      purpose="Largest-triangle-three-buckets keeps the points that shape the line, so a spike survives decimation to 60 points; taking every k-th point loses it."
      state={state}
      caption="A 5000-step random walk with one spike at t = 3100, decimated by the lab's drawing helpers (lttb, minMaxDecimate). It opens on 'every k-th point', which misses the spike; switch to LTTB or min–max and it is kept."
    >
      <Plot x={x} y={y}>
        <Curve name={`all ${n} points`} x={xs} y={ys} thin slot={0} />
        <Curve name={`${kept.x.length} kept`} x={kept.x} y={kept.y} slot={1} showPoints />
      </Plot>
    </Figure>
  )
}
