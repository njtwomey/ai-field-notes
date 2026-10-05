import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Raster,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { normal, stream, uniform } from 'aifn-compute/foundation/random'
import { decision, makeData, trainOcSvm, zeroContour, type Point, type Shape } from './ocsvm'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'

const SHAPES = [
  { value: 'blob' as const, label: 'one blob' },
  { value: 'blobs' as const, label: 'two blobs' },
  { value: 'ring' as const, label: 'ring' },
  { value: 'banana' as const, label: 'banana' },
]
const LIM = 4
const AXIS = toFlat(linspace(-LIM, LIM, 49))
// A training point counts as outside the region when f is below this; points on the hyperplane sit within the solver
// tolerance (about 1e-7) of zero.
const OUTSIDE = -1e-6

const fmt = (v: number) => v.toFixed(2)

/** A one-class SVM with the Gaussian kernel, solved in the browser, with its ν-property read off the fit. */
export function OneClassSvmExplorer() {
  const state = useFigureState({
    shape: choice<Shape>(SHAPES, 'blob', { label: 'data' }),
    nu: float(0.1, { min: 0.02, max: 0.5, step: 0.01, label: 'ν', format: fmt }),
    logGamma: float(Math.log10(0.5), {
      min: -1.3,
      max: 1,
      step: 0.05,
      label: 'kernel width γ',
      points_per_decade: 2,
      logTransform: 'value-is-log',
      format: (v) => formatNumber(10 ** v),
    }),
    seed: int(1, { min: 1, max: 20, step: 1, label: 'data seed', format: (v) => String(v) }),
    px: slider(-LIM, LIM, 2.5, { step: 0.05, onChart: true }),
    py: slider(-LIM, LIM, -2.5, { step: 0.05, onChart: true }),
  })
  const gamma = 10 ** state.logGamma

  const x = useMemo(() => {
    const s = stream(state.seed)
    return makeData(state.shape, { uniform: () => uniform(s), normal: () => normal(s) })
  }, [state.shape, state.seed])
  const r = useMemo(() => {
    const fit = trainOcSvm(x, state.nu, gamma)
    const z = AXIS.map((v) => AXIS.map((u) => decision(fit, x, gamma, [u, v])))
    const top = Math.max(...z.flat().map(Math.abs))
    // 0: inside, α = 0. 1: support vector on the hyperplane, 0 < α < C. 2: support vector at the bound, α = C.
    const kind = fit.alpha.map((a) => (a >= fit.C * (1 - 1e-9) ? 2 : a > 1e-9 ? 1 : 0))
    return {
      fit,
      z,
      top,
      kind,
      contour: zeroContour(AXIS, AXIS, z),
      outside: fit.f.filter((f) => f < OUTSIDE).length,
      sv: kind.filter((k) => k > 0).length,
      bound: kind.filter((k) => k === 2).length,
    }
  }, [x, state.nu, gamma])

  const overlay = useMemo(
    () =>
      [
        {
          name: 'boundary f = 0',
          x: r.contour.x,
          y: r.contour.y,
          emphasis: true,
        },
        {
          name: 'training points',
          x: x.map((p) => p[0]),
          y: x.map((p) => p[1]),
          group: r.kind,
          groupNames: ['inside (α = 0)', 'on the boundary (0 < α < C)', 'at the bound (α = C)'],
        },
      ] as const,
    [x, r],
  )

  const probe: Point = [state.px, state.py]
  const score = decision(r.fit, x, gamma, probe)
  const n = x.length

  const xAxis = useAxis({ label: 'x₁' })
  const yAxis = useAxis({ label: 'x₂' })
  return (
    <Figure
      title="The one-class SVM and its ν-property"
      state={state}
      caption="The colour is the decision function f(x) = Σ αᵢ k(xᵢ, x) − ρ of a one-class SVM with the Gaussian kernel, fitted to 110 points of the chosen shape and 6 points scattered uniformly. Red is inside the estimated region, blue outside, and the black line is the boundary f = 0. Circles have α = 0. Squares are support vectors on the boundary. Triangles are support vectors at the bound α = C = 1/(νn), which include every training point outside. Drag the probe to read its score. The readout checks the ν-property: the fraction outside never exceeds ν, and the fraction of support vectors never falls below it. A large γ wraps the region tightly around small groups of points; a small γ gives one smooth, convex-looking region."

      readouts={
        <>
          <Readout
            label="outside ≤ ν ≤ support vectors"
            value={`${fmt(r.outside / n)} ≤ ${fmt(state.nu)} ≤ ${fmt(r.sv / n)}`}
          />
          <Readout label="support vectors" value={`${r.sv} of ${n} (${r.bound} at C)`} />
          <Readout label="ρ" value={formatNumber(r.fit.rho)} />
          <Readout label="probe f(x)" value={`${formatNumber(score)} (${score >= 0 ? 'normal' : 'anomaly'})`} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-lg">
        <Plot x={xAxis} y={yAxis} height={440}>
          <Raster x={AXIS} y={AXIS} z={r.z} scale={'diverging'} range={[-r.top, r.top]} valueLabel={'f(x)'} />
          <Curve {...overlay[0]} live />
          <Points {...overlay[1]} live />
          <Handle
            kind="point"
            at={probe}
            label="probe"
            onDrag={([a, b]) => {
              state.set('px', a)
              state.set('py', b)
            }}
          />
        </Plot>
      </div>
    </Figure>
  )
}
