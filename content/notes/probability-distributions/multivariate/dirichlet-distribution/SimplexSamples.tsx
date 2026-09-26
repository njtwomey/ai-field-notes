import { useMemo, useState } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, type Segment, type XYSeries } from '@/components/viz'
import { rng } from '@/lib/math'

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
function gammaSample(alpha: number, r: ReturnType<typeof rng>): number {
  if (alpha < 1) return gammaSample(alpha + 1, r) * Math.pow(Math.max(r.uniform(), 1e-300), 1 / alpha)
  const d = alpha - 1 / 3
  const c = 1 / Math.sqrt(9 * d)
  for (;;) {
    const z = r.normal()
    const v = (1 + c * z) ** 3
    if (v <= 0) continue
    const u = r.uniform()
    if (Math.log(u) < 0.5 * z * z + d - d * v + d * Math.log(v)) return d * v
  }
}

const toPlane = (t: number[]): [number, number] => [
  t[0] * VERTICES[0][0] + t[1] * VERTICES[1][0] + t[2] * VERTICES[2][0],
  t[0] * VERTICES[0][1] + t[1] * VERTICES[1][1] + t[2] * VERTICES[2][1],
]

/** Samples from a three-category Dirichlet, drawn on the triangle of probability vectors. */
export function SimplexSamples() {
  const [a1, setA1] = useState(2)
  const [a2, setA2] = useState(2)
  const [a3, setA3] = useState(2)

  const series = useMemo((): XYSeries[] => {
    const alpha = [a1, a2, a3]
    const r = rng(11)
    const points = Array.from({ length: N }, () => {
      const g = alpha.map((a) => gammaSample(a, r))
      const total = g[0] + g[1] + g[2]
      return toPlane(g.map((x) => x / total))
    })
    const total = a1 + a2 + a3
    const mean = toPlane(alpha.map((a) => a / total))
    return [
      { name: 'samples', type: 'scatter', x: points.map((p) => p[0]), y: points.map((p) => p[1]), slot: 0 },
      { name: 'mean', type: 'scatter', x: [mean[0]], y: [mean[1]], emphasis: true },
    ]
  }, [a1, a2, a3])

  const a0 = a1 + a2 + a3
  return (
    <Interactive
      title="Dirichlet samples on the simplex"
      caption="Each point is a probability vector θ = (θ₁, θ₂, θ₃). The bottom-left corner is θ = (1, 0, 0), the bottom-right (0, 1, 0) and the top (0, 0, 1). Equal α below 1 pushes samples to the corners and edges (sparse vectors); α = 1 is uniform on the triangle; large α concentrates them around the mean α/α₀."
      controls={
        <>
          <ParamSlider label="α₁" value={a1} onChange={setA1} min={0.1} max={20} step={0.1} />
          <ParamSlider label="α₂" value={a2} onChange={setA2} min={0.1} max={20} step={0.1} />
          <ParamSlider label="α₃" value={a3} onChange={setA3} min={0.1} max={20} step={0.1} />
        </>
      }
      readout={
        <>
          <Readout label="concentration α₀" value={formatNumber(a0)} />
          <Readout label="mean θ" value={`(${[a1, a2, a3].map((a) => formatNumber(a / a0)).join(', ')})`} />
        </>
      }
    >
      <XYChart series={series} segments={OUTLINE} xRange={X_RANGE} yRange={Y_RANGE} equalAspect bare />
    </Interactive>
  )
}
