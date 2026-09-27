import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from '@/components/viz'
import { linspace, rng } from '@/lib/math'
import { costMatrix, hungarian, type Pt } from '../_shared/ot'

type View = 'densities' | 'clouds'

// ---------- One dimension: quantile averaging ----------

const XS = linspace(-7, 7, 1401)
const DX = XS[1] - XS[0]
const gauss = (x: number, m: number, s: number) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI))
const P0 = XS.map((x) => 0.5 * gauss(x, -3.5, 0.45) + 0.5 * gauss(x, -1, 0.45))
const P1 = XS.map((x) => gauss(x, 3, 0.8))
const US = linspace(0.001, 0.999, 999)

/** Quantile function of a density on the grid XS, evaluated at the levels US, by inverting the cumulative sum. */
function quantiles(p: number[]): number[] {
  const cdf: number[] = []
  let c = 0
  for (const v of p) {
    c += v * DX
    cdf.push(c)
  }
  const total = cdf[cdf.length - 1]
  let k = 0
  return US.map((u) => {
    const target = u * total
    while (k < cdf.length - 1 && cdf[k + 1] < target) k++
    const lo = cdf[k]
    const hi = cdf[k + 1] ?? lo
    const w = hi > lo ? (target - lo) / (hi - lo) : 0
    return XS[k] + w * DX
  })
}
const Q0 = quantiles(P0)
const Q1 = quantiles(P1)

const moments = (p: number[]) => {
  const m = p.reduce((s, v, i) => s + v * XS[i] * DX, 0)
  const v = p.reduce((s, w, i) => s + w * (XS[i] - m) ** 2 * DX, 0)
  return { m, v }
}
const M0 = moments(P0)
const M1 = moments(P1)

// ---------- Two dimensions: moving matched points ----------

const NPTS = 40
const CLOUDS = (() => {
  const r = rng(8)
  const src: Pt[] = Array.from({ length: NPTS }, () => [-2.2 + 0.45 * r.normal(), -1 + 0.45 * r.normal()])
  const tgt: Pt[] = Array.from({ length: NPTS }, () => {
    const a = 2 * Math.PI * r.uniform()
    const rad = 1.1 + 0.08 * r.normal()
    return [2 + rad * Math.cos(a), 1 + rad * Math.sin(a)]
  })
  const col = hungarian(costMatrix(src, tgt, 2))
  return { src, tgt, col }
})()
const RANGE: [number, number] = [-3.5, 3.5]

/**
 * McCann's displacement interpolation moves mass along the optimal plan; the mixture fades one distribution out and the
 * other in. In 1-D the displacement interpolant has quantile function (1 − t)Q₀ + tQ₁.
 */
export function DisplacementInterpolation() {
  const t = useParam(0.5, { min: 0, max: 1, step: 0.01 })
  const [view, setView] = useState<View>('densities')

  const oneD = useMemo(() => {
    const q = US.map((_, k) => (1 - t.value) * Q0[k] + t.value * Q1[k])
    // Density of the pushed-forward quantile: du / dq at the midpoints.
    const dx: number[] = []
    const dy: number[] = []
    for (let k = 0; k < q.length - 1; k++) {
      const dq = q[k + 1] - q[k]
      if (dq <= 1e-9) continue
      dx.push((q[k] + q[k + 1]) / 2)
      dy.push(Math.min((US[k + 1] - US[k]) / dq, 0.7))
    }
    const mixture = XS.map((_, i) => (1 - t.value) * P0[i] + t.value * P1[i])
    const mean = q.reduce((s, v) => s + v, 0) / q.length
    const sdDisp = Math.sqrt(q.reduce((s, v) => s + (v - mean) ** 2, 0) / q.length)
    const mm = (1 - t.value) * M0.m + t.value * M1.m
    const sdMix = Math.sqrt((1 - t.value) * M0.v + t.value * M1.v + t.value * (1 - t.value) * (M0.m - M1.m) ** 2)
    const series: XYSeries[] = [
      { name: 'μ₀', type: 'line', x: XS, y: P0, muted: true, dashed: true },
      { name: 'μ₁', type: 'line', x: XS, y: P1, muted: true, dashed: true },
      {
        name: 'displacement interpolation',
        type: 'line',
        x: [XS[0], ...dx, XS[XS.length - 1]],
        y: [0, ...dy, 0],
        slot: 0,
        area: true,
      },
      { name: 'mixture (1 − t)μ₀ + tμ₁', type: 'line', x: XS, y: mixture, slot: 1 },
    ]
    return { series, mean: mm, sdDisp, sdMix }
  }, [t.value])

  const twoD = useMemo(() => {
    const { src, tgt, col } = CLOUDS
    const moved = src.map((p, i): Pt => {
      const q = tgt[col[i]]
      return [(1 - t.value) * p[0] + t.value * q[0], (1 - t.value) * p[1] + t.value * q[1]]
    })
    // The mixture keeps each cloud intact and changes how much of each is present.
    const k0 = Math.round((1 - t.value) * NPTS)
    const mix = [...src.slice(0, k0), ...tgt.slice(0, NPTS - k0)]
    const series: XYSeries[] = [
      {
        name: 'displacement interpolation',
        type: 'scatter',
        x: moved.map((p) => p[0]),
        y: moved.map((p) => p[1]),
        slot: 0,
      },
      { name: 'mixture sample', type: 'scatter', x: mix.map((p) => p[0]), y: mix.map((p) => p[1]), slot: 1 },
    ]
    return series
  }, [t.value])

  return (
    <Interactive
      title="Displacement interpolation against mixing"
      caption="Move t from 0 to 1. The displacement interpolant moves every piece of mass along the optimal plan at constant speed: the two bumps of μ₀ slide together and merge into μ₁, and in 2-D the blob travels and spreads into the ring. The mixture never has mass in between: it fades μ₀ out and μ₁ in, and its spread at t = 0.5 is far larger than either end."
      controls={
        <>
          <ParamSlider label="t" param={t} />
          <ParamChoice
            label="view"
            value={view}
            onChange={setView}
            options={[
              { value: 'densities', label: '1-D densities' },
              { value: 'clouds', label: '2-D point clouds' },
            ]}
          />
        </>
      }
      readout={
        view === 'densities' ? (
          <>
            <Readout label="mean (both)" value={formatNumber(oneD.mean)} />
            <Readout label="sd, displacement" value={formatNumber(oneD.sdDisp)} />
            <Readout label="sd, mixture" value={formatNumber(oneD.sdMix)} />
          </>
        ) : (
          <Readout label="points moved along the exact matching" value={NPTS} />
        )
      }
    >
      {view === 'densities' ? (
        <XYChart height={320} series={oneD.series} xRange={[-6, 6]} yRange={[0, 0.7]} xLabel="x" yLabel="density" />
      ) : (
        <XYChart series={twoD} xRange={RANGE} yRange={RANGE} equalAspect xLabel="x₁" yLabel="x₂" />
      )}
    </Interactive>
  )
}
