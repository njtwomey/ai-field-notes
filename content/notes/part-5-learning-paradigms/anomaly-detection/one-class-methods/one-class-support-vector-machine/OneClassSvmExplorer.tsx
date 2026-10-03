import { useMemo, useState } from 'react'
import {
  Heatmap,
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  formatNumber,
  useParam,
  type Handle,
  type HeatmapOverlay,
} from 'aifn-render'
import { linspace, rng } from '@/lib/math'
import { decision, makeData, trainOcSvm, zeroContour, type Point, type Shape } from './ocsvm'

const SHAPES = [
  { value: 'blob' as const, label: 'one blob' },
  { value: 'blobs' as const, label: 'two blobs' },
  { value: 'ring' as const, label: 'ring' },
  { value: 'banana' as const, label: 'banana' },
]
const LIM = 4
const AXIS = linspace(-LIM, LIM, 49)
// A training point counts as outside the region when f is below this; points on the hyperplane sit within the solver
// tolerance (about 1e-7) of zero.
const OUTSIDE = -1e-6

const fmt = (v: number) => v.toFixed(2)

/** A one-class SVM with the Gaussian kernel, solved in the browser, with its ν-property read off the fit. */
export function OneClassSvmExplorer() {
  const [shape, setShape] = useState<Shape>('blob')
  const nu = useParam(0.1, { min: 0.02, max: 0.5, step: 0.01 })
  const logGamma = useParam(Math.log10(0.5), { min: -1.3, max: 1, step: 0.05 })
  const seed = useParam(1, { min: 1, max: 20, step: 1 })
  const px = useParam(2.5, { min: -LIM, max: LIM, step: 0.05 })
  const py = useParam(-2.5, { min: -LIM, max: LIM, step: 0.05 })
  const gamma = 10 ** logGamma.value

  const x = useMemo(() => makeData(shape, rng(seed.value)), [shape, seed.value])
  const r = useMemo(() => {
    const fit = trainOcSvm(x, nu.value, gamma)
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
  }, [x, nu.value, gamma])

  const overlay = useMemo(
    (): HeatmapOverlay[] => [
      {
        name: 'boundary f = 0',
        type: 'line',
        x: r.contour.x,
        y: r.contour.y,
        emphasis: true,
      },
      {
        name: 'training points',
        type: 'scatter',
        x: x.map((p) => p[0]),
        y: x.map((p) => p[1]),
        group: r.kind,
        groupNames: ['inside (α = 0)', 'on the boundary (0 < α < C)', 'at the bound (α = C)'],
      },
    ],
    [x, r],
  )

  const probe: Point = [px.value, py.value]
  const score = decision(r.fit, x, gamma, probe)
  const handles: Handle[] = [
    {
      kind: 'point',
      at: probe,
      label: 'probe',
      onDrag: ([a, b]) => {
        px.set(a)
        py.set(b)
      },
    },
  ]
  const n = x.length

  return (
    <Interactive
      title="The one-class SVM and its ν-property"
      caption="The colour is the decision function f(x) = Σ αᵢ k(xᵢ, x) − ρ of a one-class SVM with the Gaussian kernel, fitted to 110 points of the chosen shape and 6 points scattered uniformly. Red is inside the estimated region, blue outside, and the black line is the boundary f = 0. Circles have α = 0. Squares are support vectors on the boundary. Triangles are support vectors at the bound α = C = 1/(νn), which include every training point outside. Drag the probe to read its score. The readout checks the ν-property: the fraction outside never exceeds ν, and the fraction of support vectors never falls below it. A large γ wraps the region tightly around small groups of points; a small γ gives one smooth, convex-looking region."
      controls={
        <>
          <ParamChoice label="data" value={shape} onChange={setShape} options={SHAPES} />
          <ParamSlider label="ν" param={nu} format={fmt} />
          <ParamSlider label="kernel width γ" param={logGamma} format={(v) => formatNumber(10 ** v)} />
          <ParamSlider label="data seed" param={seed} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
        <>
          <Readout
            label="outside ≤ ν ≤ support vectors"
            value={`${fmt(r.outside / n)} ≤ ${fmt(nu.value)} ≤ ${fmt(r.sv / n)}`}
          />
          <Readout label="support vectors" value={`${r.sv} of ${n} (${r.bound} at C)`} />
          <Readout label="ρ" value={formatNumber(r.fit.rho)} />
          <Readout label="probe f(x)" value={`${formatNumber(score)} (${score >= 0 ? 'normal' : 'anomaly'})`} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-lg">
        <Heatmap
          x={AXIS}
          y={AXIS}
          z={r.z}
          scale="diverging"
          range={[-r.top, r.top]}
          overlay={overlay}
          handles={handles}
          xLabel="x₁"
          yLabel="x₂"
          valueLabel="f(x)"
          height={440}
        />
      </div>
    </Interactive>
  )
}
