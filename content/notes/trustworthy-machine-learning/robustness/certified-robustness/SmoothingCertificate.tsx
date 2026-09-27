import { useMemo } from 'react'
import {
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import { linspace, rng } from '@/lib/math'
import { incompleteBeta, normalQuantile } from '@/lib/math/special'

const R = 3
const N = 2000
const SHOWN = 400
const ALPHA = 0.001
// The base classifier: class A inside a three-lobed blob, except for a small pocket of class B inside it.
const HOLE = { x: 0.7, y: 0.25, r: 0.2 }
const blobRadius = (t: number) => 1.7 + 0.35 * Math.cos(3 * t)
const inA = (x: number, y: number) =>
  Math.hypot(x, y) < blobRadius(Math.atan2(y, x)) && Math.hypot(x - HOLE.x, y - HOLE.y) > HOLE.r

const ANGLES = linspace(0, 2 * Math.PI, 361)
const BLOB = { x: ANGLES.map((t) => blobRadius(t) * Math.cos(t)), y: ANGLES.map((t) => blobRadius(t) * Math.sin(t)) }
const HOLE_LINE = {
  x: ANGLES.map((t) => HOLE.x + HOLE.r * Math.cos(t)),
  y: ANGLES.map((t) => HOLE.y + HOLE.r * Math.sin(t)),
}

// Fixed standard normal draws: the same noise for every input, so the estimate moves smoothly under a drag.
const NOISE = (() => {
  const g = rng(7)
  return Array.from({ length: N }, () => [g.normal(), g.normal()] as const)
})()

/** One-sided Clopper–Pearson lower bound: the α quantile of Beta(k, n − k + 1). */
function lowerBound(k: number, n: number, alpha: number) {
  if (k === 0) return 0
  let lo = 0
  let hi = 1
  for (let i = 0; i < 50; i++) {
    const mid = (lo + hi) / 2
    if (incompleteBeta(mid, k, n - k + 1) < alpha) lo = mid
    else hi = mid
  }
  return lo
}

/** Distance from (x, y) to the nearest point of the base classifier's decision boundary. */
function boundaryDistance(x: number, y: number) {
  let d = Math.abs(Math.hypot(x - HOLE.x, y - HOLE.y) - HOLE.r)
  for (let i = 0; i < BLOB.x.length; i++) d = Math.min(d, Math.hypot(x - BLOB.x[i], y - BLOB.y[i]))
  return d
}

export function SmoothingCertificate() {
  const sigma = useParam(0.5, { min: 0.1, max: 1.2, step: 0.01 })
  const px = useParam(0.35, { min: -R, max: R, step: 0.01 })
  const py = useParam(0.1, { min: -R, max: R, step: 0.01 })

  const est = useMemo(() => {
    const s = sigma.value
    const zx: number[] = []
    const zy: number[] = []
    const cls: number[] = []
    let countA = 0
    NOISE.forEach(([a, b], i) => {
      const x = px.value + s * a
      const y = py.value + s * b
      const isA = inA(x, y)
      if (isA) countA++
      if (i < SHOWN) {
        zx.push(x)
        zy.push(y)
        cls.push(isA ? 0 : 1)
      }
    })
    const topIsA = countA >= N - countA
    const k = topIsA ? countA : N - countA
    const pLower = lowerBound(k, N, ALPHA)
    const radius = pLower > 0.5 ? s * normalQuantile(pLower) : 0
    return { zx, zy, cls, topIsA, pHat: k / N, pLower, radius }
  }, [sigma.value, px.value, py.value])

  const circle = {
    x: ANGLES.map((t) => px.value + est.radius * Math.cos(t)),
    y: ANGLES.map((t) => py.value + est.radius * Math.sin(t)),
  }
  const series: XYSeries[] = [
    {
      name: 'noisy copies',
      type: 'scatter',
      x: est.zx,
      y: est.zy,
      group: est.cls,
      groupNames: ['f = A', 'f = B'],
    },
    { name: 'base boundary', type: 'line', x: BLOB.x, y: BLOB.y, emphasis: true },
    { name: 'pocket of class B', type: 'line', x: HOLE_LINE.x, y: HOLE_LINE.y, emphasis: true },
    { name: 'certified radius', type: 'line', x: circle.x, y: circle.y, slot: 2 },
  ]
  const handles: Handle[] = [
    {
      kind: 'point',
      at: [px.value, py.value],
      label: 'input',
      onDrag: ([x, y]) => {
        px.set(x)
        py.set(y)
      },
    },
  ]

  return (
    <Interactive
      title="Randomised smoothing certificate"
      caption={`The base classifier f says A inside the three-lobed curve, except in the small circular pocket, and B elsewhere. Drag the input. The smoothed classifier g votes over ${N} copies of the input with Gaussian noise of standard deviation σ (${SHOWN} shown, shaped by f's label). The circle is the certified ℓ₂ radius σΦ⁻¹(p_A), with p_A the one-sided Clopper–Pearson lower bound at α = ${ALPHA}. Near the pocket f can be flipped by a tiny perturbation, but g cannot: its certificate covers the pocket when σ is large enough. Larger σ certifies larger radii far from the boundary and loses accuracy near it.`}
      controls={<ParamSlider label="noise σ" param={sigma} />}
      readout={
        <>
          <Readout label="smoothed prediction" value={est.pLower > 0.5 ? (est.topIsA ? 'A' : 'B') : 'abstain'} />
          <Readout label="vote share of top class" value={formatNumber(est.pHat)} />
          <Readout label="lower bound p_A" value={formatNumber(est.pLower)} />
          <Readout label="certified radius" value={formatNumber(est.radius)} />
          <Readout label="distance to f's boundary" value={formatNumber(boundaryDistance(px.value, py.value))} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-md">
        <XYChart
          series={series}
          handles={handles}
          xRange={[-R, R]}
          yRange={[-R, R]}
          equalAspect
          xLabel="x₁"
          yLabel="x₂"
          ariaLabel="A base classifier's regions, Gaussian noise samples around an input and the certified radius of the smoothed classifier"
        />
      </div>
    </Interactive>
  )
}
