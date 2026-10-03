import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type Segment,
  type XYSeries,
} from 'aifn-render'
import { linspace, rng } from '@/lib/math'
import { eigSym, type Vec2 } from '@/lib/math/mat2'

const N = 400
// Fixed standard-normal pairs z; every sample is μ + R D z, so any change deforms the same cloud smoothly.
const Z = (() => {
  const r = rng(3)
  return Array.from({ length: N }, () => [r.normal(), r.normal()] as const)
})()
const ANGLES = linspace(0, 2 * Math.PI, 97)
const RANGE: [number, number] = [-6, 6]
const SD = { min: 0.2, max: 3, step: 0.05 }
/** The axis handles sit on the 2σ contour, where the tips are clear of the densest samples. */
const TIP = 2

type View = 'principal' | 'marginal'

const radians = (deg: number) => (deg * Math.PI) / 180
const degrees = (rad: number) => ((((rad * 180) / Math.PI) % 360) + 360) % 360

/**
 * Samples and 1σ, 2σ contours of a bivariate Gaussian. The shape is stored in principal form, Σ = R(θ) diag(a², b²) R(θ)ᵀ:
 * two perpendicular axes with standard deviations a and b. Their tips are draggable, so the reader stretches and turns
 * the principal axes directly; the marginal view shows the same Σ as σ₁, σ₂ and ρ.
 */
export function CovarianceExplorer() {
  const [view, setView] = useState<View>('principal')
  const a = useParam(1.6, SD)
  const b = useParam(0.6, SD)
  const theta = useParam(30, { min: 0, max: 359, step: 1 })
  const mx = useParam(0, { min: -3, max: 3, step: 0.1 })
  const my = useParam(0, { min: -3, max: 3, step: 0.1 })

  const t = radians(theta.value)
  const u1: Vec2 = [Math.cos(t), Math.sin(t)]
  const u2: Vec2 = [-Math.sin(t), Math.cos(t)]
  const [a2, b2] = [a.value ** 2, b.value ** 2]
  // Σ = a² u₁u₁ᵀ + b² u₂u₂ᵀ.
  const s11 = a2 * u1[0] ** 2 + b2 * u2[0] ** 2
  const s22 = a2 * u1[1] ** 2 + b2 * u2[1] ** 2
  const s12 = a2 * u1[0] * u1[1] + b2 * u2[0] * u2[1]
  const sigma1 = Math.sqrt(s11)
  const sigma2 = Math.sqrt(s22)
  const rho = s12 / (sigma1 * sigma2)

  const { series, inside } = useMemo(() => {
    // x = μ + a z₁ u₁ + b z₂ u₂, a square root of Σ that keeps the principal axes explicit.
    const map = (p: number, q: number): Vec2 => [
      mx.value + a.value * p * u1[0] + b.value * q * u2[0],
      my.value + a.value * p * u1[1] + b.value * q * u2[1],
    ]
    const points = Z.map(([p, q]) => map(p, q))
    // A sample lies inside the radius-r contour when its z has length below r (Mahalanobis distance = ‖z‖).
    const inside = [1, 2].map((r) => Z.filter(([p, q]) => p * p + q * q < r * r).length / N)
    const contour = (r: number): XYSeries => {
      const pts = ANGLES.map((s) => map(r * Math.cos(s), r * Math.sin(s)))
      return {
        name: `${r}σ contour`,
        type: 'line',
        x: pts.map((p) => p[0]),
        y: pts.map((p) => p[1]),
        slot: 1,
        dashed: r === 2,
      }
    }
    const series: XYSeries[] = [
      { name: 'samples', type: 'scatter', x: points.map((p) => p[0]), y: points.map((p) => p[1]), slot: 0 },
      contour(1),
      contour(2),
    ]
    return { series, inside }
    // u₁ and u₂ are derived from θ, so θ covers them.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [a.value, b.value, theta.value, mx.value, my.value])

  const tip1: Vec2 = [mx.value + TIP * a.value * u1[0], my.value + TIP * a.value * u1[1]]
  const tip2: Vec2 = [mx.value + TIP * b.value * u2[0], my.value + TIP * b.value * u2[1]]
  const axes: Segment[] = [
    { from: [mx.value, my.value], to: tip1 },
    { from: [mx.value, my.value], to: tip2 },
  ]

  const handles: Handle[] = [
    {
      kind: 'point',
      at: [mx.value, my.value],
      label: 'mean',
      onDrag: ([x, y]) => {
        mx.set(x)
        my.set(y)
      },
    },
    {
      kind: 'point',
      at: tip1,
      label: 'axis 1',
      // The tip of axis 1 sets its direction and its standard deviation; axis 2 stays perpendicular.
      onDrag: ([x, y]) => {
        const d: Vec2 = [x - mx.value, y - my.value]
        a.set(Math.hypot(...d) / TIP)
        theta.set(Math.round(degrees(Math.atan2(d[1], d[0]))))
      },
    },
    {
      kind: 'point',
      at: tip2,
      label: 'axis 2',
      onDrag: ([x, y]) => {
        const d: Vec2 = [x - mx.value, y - my.value]
        b.set(Math.hypot(...d) / TIP)
        theta.set(Math.round(degrees(Math.atan2(d[1], d[0]) - Math.PI / 2)))
      },
    },
  ]

  // Marginal sliders rebuild Σ from σ₁, σ₂, ρ and convert back to principal form. The eigenvector nearer the current
  // axis 1 stays axis 1, so a small slider change never swaps the axes' roles.
  const setMarginal = (next: { s1?: number; s2?: number; r?: number }) => {
    const n1 = next.s1 ?? sigma1
    const n2 = next.s2 ?? sigma2
    const nr = next.r ?? rho
    const {
      values: [l1, l2],
      vectors: [v1, v2],
    } = eigSym(n1 * n1, nr * n1 * n2, n2 * n2)
    const keepFirst = Math.abs(v1[0] * u1[0] + v1[1] * u1[1]) >= Math.abs(v2[0] * u1[0] + v2[1] * u1[1])
    const [axis, sdA, sdB] = keepFirst ? [v1, l1, l2] : [v2, l2, l1]
    // Point axis 1 the same way as before, so θ moves continuously.
    const sign = axis[0] * u1[0] + axis[1] * u1[1] < 0 ? -1 : 1
    a.set(Math.sqrt(Math.max(sdA, 0)))
    b.set(Math.sqrt(Math.max(sdB, 0)))
    theta.set(Math.round(degrees(Math.atan2(sign * axis[1], sign * axis[0]))))
  }

  return (
    <Interactive
      title="Covariance as shape"
      caption="Each sample is μ + a z₁ u₁ + b z₂ u₂ for a fixed set of standard-normal pairs z: the cloud is stretched by a along the principal axis u₁ and by b along the perpendicular axis u₂. Drag the tips of the two axes to stretch and turn them, or drag the mean. The contours are the points at Mahalanobis distance 1 (solid) and 2 (dashed), holding about 39% and 86% of the mass in two dimensions. Switch to the marginal sliders to set the same covariance through σ₁, σ₂ and ρ instead."
      controls={
        <>
          <ParamChoice
            label="sliders"
            value={view}
            onChange={setView}
            options={[
              { value: 'principal', label: 'principal axes' },
              { value: 'marginal', label: 'σ₁, σ₂, ρ' },
            ]}
          />
          {view === 'principal' ? (
            <>
              <ParamSlider label="sd along axis 1, a" param={a} />
              <ParamSlider label="sd along axis 2, b" param={b} />
              <ParamSlider label="rotation θ (°)" param={theta} format={(v) => `${Math.round(v)}°`} />
            </>
          ) : (
            <>
              <ParamSlider
                label="σ₁"
                value={sigma1}
                onChange={(v) => setMarginal({ s1: v })}
                min={0.2}
                max={3}
                step={0.05}
              />
              <ParamSlider
                label="σ₂"
                value={sigma2}
                onChange={(v) => setMarginal({ s2: v })}
                min={0.2}
                max={3}
                step={0.05}
              />
              <ParamSlider
                label="correlation ρ"
                value={rho}
                onChange={(v) => setMarginal({ r: v })}
                min={-0.95}
                max={0.95}
                step={0.01}
              />
            </>
          )}
        </>
      }
      readout={
        <>
          <Readout label="eigenvalues a², b²" value={`${formatNumber(a2)}, ${formatNumber(b2)}`} />
          <Readout label="σ₁, σ₂" value={`${formatNumber(sigma1)}, ${formatNumber(sigma2)}`} />
          <Readout label="ρ" value={formatNumber(rho)} />
          <Readout label="det Σ = a²b²" value={formatNumber(a2 * b2)} />
          <Readout label="inside 1σ" value={`${formatNumber(100 * inside[0])}%`} />
          <Readout label="inside 2σ" value={`${formatNumber(100 * inside[1])}%`} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-xl">
        <XYChart
          series={series}
          vectors={axes}
          xLabel="x₁"
          yLabel="x₂"
          xRange={RANGE}
          yRange={RANGE}
          equalAspect
          handles={handles}
        />
      </div>
    </Interactive>
  )
}
