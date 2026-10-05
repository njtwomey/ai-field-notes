import { useMemo } from 'react'
import {
  choice,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Readout,
  type Segment,
  seriesLayers,
  type SeriesSpec,
  slider,
  Slider,
  useAxis,
  useFigureState,
  Vectors,
  when,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { normal, stream } from 'aifn-compute/foundation/random'
import { eigh2, type Vec2 } from 'aifn-compute/numerics/linalg'

const N = 400
// Fixed standard-normal pairs z; every sample is μ + R D z, so any change deforms the same cloud smoothly.
const Z = (() => {
  const r = stream(3)
  return Array.from({ length: N }, () => [normal(r), normal(r)] as const)
})()
const ANGLES = toFlat(linspace(0, 2 * Math.PI, 97))
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
  const state = useFigureState({
    view: choice<View>(
      [
        { value: 'principal', label: 'principal axes' },
        { value: 'marginal', label: 'σ₁, σ₂, ρ' },
      ],
      'principal',
      { label: 'sliders' },
    ),
    mx: slider(-3, 3, 0, { step: 0.1, onChart: true }),
    my: slider(-3, 3, 0, { step: 0.1, onChart: true }),
    a: slider(SD.min, SD.max, 1.6, { step: SD.step, label: 'sd along axis 1, a', when: when('view', 'principal') }),
    b: slider(SD.min, SD.max, 0.6, { step: SD.step, label: 'sd along axis 2, b', when: when('view', 'principal') }),
    theta: slider(0, 359, 30, {
      step: 1,
      label: 'rotation θ (°)',
      format: (v) => `${Math.round(v)}°`,
      when: when('view', 'principal'),
    }),
  })
  const a = { value: state.a, set: (v: number) => state.set('a', v) }
  const b = { value: state.b, set: (v: number) => state.set('b', v) }
  const theta = { value: state.theta, set: (v: number) => state.set('theta', v) }

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
      state.mx + a.value * p * u1[0] + b.value * q * u2[0],
      state.my + a.value * p * u1[1] + b.value * q * u2[1],
    ]
    const points = Z.map(([p, q]) => map(p, q))
    // A sample lies inside the radius-r contour when its z has length below r (Mahalanobis distance = ‖z‖).
    const inside = [1, 2].map((r) => Z.filter(([p, q]) => p * p + q * q < r * r).length / N)
    const contour = (r: number): SeriesSpec => {
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
    const series: SeriesSpec[] = [
      { name: 'samples', type: 'scatter', x: points.map((p) => p[0]), y: points.map((p) => p[1]), slot: 0 },
      contour(1),
      contour(2),
    ]
    return { series, inside }
    // u₁ and u₂ are derived from θ, so θ covers them.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [a.value, b.value, theta.value, state.mx, state.my])

  const tip1: Vec2 = [state.mx + TIP * a.value * u1[0], state.my + TIP * a.value * u1[1]]
  const tip2: Vec2 = [state.mx + TIP * b.value * u2[0], state.my + TIP * b.value * u2[1]]
  const axes: Segment[] = [
    { from: [state.mx, state.my], to: tip1 },
    { from: [state.mx, state.my], to: tip2 },
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
    } = eigh2([
      [n1 * n1, nr * n1 * n2],
      [nr * n1 * n2, n2 * n2],
    ])
    const keepFirst = Math.abs(v1[0] * u1[0] + v1[1] * u1[1]) >= Math.abs(v2[0] * u1[0] + v2[1] * u1[1])
    const [axis, sdA, sdB] = keepFirst ? [v1, l1, l2] : [v2, l2, l1]
    // Point axis 1 the same way as before, so θ moves continuously.
    const sign = axis[0] * u1[0] + axis[1] * u1[1] < 0 ? -1 : 1
    a.set(Math.sqrt(Math.max(sdA, 0)))
    b.set(Math.sqrt(Math.max(sdB, 0)))
    theta.set(Math.round(degrees(Math.atan2(sign * axis[1], sign * axis[0]))))
  }

  const xAxis = useAxis({ label: 'x₁', range: RANGE })
  const yAxis = useAxis({ label: 'x₂', range: RANGE, equal: xAxis })
  return (
    <Figure
      title="Covariance as shape"
      state={state}
      caption="Each sample is μ + a z₁ u₁ + b z₂ u₂ for a fixed set of standard-normal pairs z: the cloud is stretched by a along the principal axis u₁ and by b along the perpendicular axis u₂. Drag the tips of the two axes to stretch and turn them, or drag the mean. The contours are the points at Mahalanobis distance 1 (solid) and 2 (dashed), holding about 39% and 86% of the mass in two dimensions. Switch to the marginal sliders to set the same covariance through σ₁, σ₂ and ρ instead."
      controls={
        <>
          {state.view === 'marginal' && (
            <>
              <Slider
                label="σ₁"
                value={sigma1}
                onChange={(v) => setMarginal({ s1: v })}
                min={0.2}
                max={3}
                step={0.05}
              />
              <Slider
                label="σ₂"
                value={sigma2}
                onChange={(v) => setMarginal({ s2: v })}
                min={0.2}
                max={3}
                step={0.05}
              />
              <Slider
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
      readouts={
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
        <Plot x={xAxis} y={yAxis}>
          {seriesLayers(series)}
          <Vectors vectors={axes} />
          <Handle
            kind="point"
            at={[state.mx, state.my]}
            label="mean"
            onDrag={([x, y]) => {
              state.set('mx', x)
              state.set('my', y)
            }}
          />
          <Handle
            kind="point"
            at={tip1}
            label="axis 1"
            onDrag={([x, y]) => {
              const d: Vec2 = [x - state.mx, y - state.my]
              a.set(Math.hypot(...d) / TIP)
              theta.set(Math.round(degrees(Math.atan2(d[1], d[0]))))
            }}
          />
          <Handle
            kind="point"
            at={tip2}
            label="axis 2"
            onDrag={([x, y]) => {
              const d: Vec2 = [x - state.mx, y - state.my]
              b.set(Math.hypot(...d) / TIP)
              theta.set(Math.round(degrees(Math.atan2(d[1], d[0]) - Math.PI / 2)))
            }}
          />
        </Plot>
      </div>
    </Figure>
  )
}
