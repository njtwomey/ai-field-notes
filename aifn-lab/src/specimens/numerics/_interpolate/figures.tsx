import { useMemo } from 'react'
import { child, normals, stream, uniform } from 'aifn/foundation/random'
import {
  akima,
  bsplineBasis,
  cubicSpline,
  evaluatePiecewise,
  interpolatingPolynomial,
  linearInterpolant,
  pchip,
  pspline,
  psplineGcvPath,
  uniformKnots,
  type PiecewisePolynomial,
} from 'aifn/numerics/interpolate'
import { argmin, linspace, tensor, toFlat, toRows, type Tensor } from 'aifn/foundation/tensor'
import { categorical, useTheme } from '@lab/design'
import { Figure } from '@lab/layout'
import { choice, row, slider, toggle, useComputed, useFigureState, type ParamDefs, type SliderDef } from '@lab/state'
import { Area, Curve, Handle, Plot, Plots, Points, Readout, useAxis } from '@lab/viz'
import { formatValue } from '@lab/views'
const GRID = linspace(0, 10, 401)
const GRID_X = toFlat(GRID)

const START: [number, number][] = [
  [0.5, 1],
  [2, 1.2],
  [3, 3.5],
  [4.5, 3.6],
  [6, 1],
  [7.5, 1.3],
  [9.5, 4],
]

const METHODS = ['linear', 'natural cubic', 'not-a-knot cubic', 'PCHIP', 'Akima', 'polynomial'] as const
type MethodName = (typeof METHODS)[number]

function interpolant(method: MethodName, x: Tensor, y: Tensor): PiecewisePolynomial | null {
  switch (method) {
    case 'linear':
      return linearInterpolant(x, y)
    case 'natural cubic':
      return cubicSpline(x, y, { bc: 'natural' })
    case 'not-a-knot cubic':
      return cubicSpline(x, y)
    case 'PCHIP':
      return pchip(x, y)
    case 'Akima':
      return akima(x, y)
    case 'polynomial':
      return null
  }
}

/** Each point's coordinates as two on-chart fields, p0x, p0y, …: dragged on the chart, kept in the URL. */
const POINT_FIELDS: Record<string, SliderDef> = Object.fromEntries(
  START.flatMap(([px, py], i) => [
    [`p${i}x`, slider(0, 10, px, { onChart: true, step: 0.05 })],
    [`p${i}y`, slider(-1, 7, py, { onChart: true, step: 0.05 })],
  ]),
)
const INTERPOLANT_SCHEMA = {
  focus: choice(METHODS, 'PCHIP', { label: 'show the second derivative of' }),
  ...POINT_FIELDS,
} satisfies ParamDefs

/** Interpolants through the same points: smoothness against overshoot. */
export function InterpolantsCompared() {
  const state = useFigureState(INTERPOLANT_SCHEMA)
  const focus = state.focus as MethodName
  const values = state.values as unknown as Record<string, number>
  const key = START.map((_, i) => `${values[`p${i}x`]},${values[`p${i}y`]}`).join(';')
  const sorted = useMemo(
    () =>
      key
        .split(';')
        .map((p) => p.split(',').map(Number) as [number, number])
        .sort((a, b) => a[0] - b[0]),
    [key],
  )
  const x = useMemo(() => tensor(sorted.map((p) => p[0])), [sorted])
  const y = useMemo(() => tensor(sorted.map((p) => p[1])), [sorted])
  const distinct = sorted.every((p, i) => i === 0 || p[0] > sorted[i - 1][0])
  const curves = useMemo(() => {
    if (!distinct) return null
    return METHODS.map((m) => {
      const pp = interpolant(m, x, y)
      if (pp) {
        return {
          m,
          f: toFlat(evaluatePiecewise(pp, GRID)),
          d2: toFlat(evaluatePiecewise(pp, GRID, { derivative: 2 })),
        }
      }
      const p = interpolatingPolynomial(x, y)
      return { m, f: toFlat(p.evaluate(GRID)), d2: toFlat(p.evaluate(GRID, 2)) }
    })
  }, [x, y, distinct])
  const chosen = curves?.find((c) => c.m === focus)
  // The lower panel follows on release: redrawing it on every pointer move is what made the drag slow.
  const d2 = useComputed(() => chosen?.d2 ?? null, [chosen], { mode: 'release' })
  const overshoot = chosen ? Math.max(...chosen.f) - Math.max(...sorted.map((p) => p[1])) : NaN
  const px = useMemo(() => sorted.map((p) => p[0]), [sorted])
  const py = useMemo(() => sorted.map((p) => p[1]), [sorted])
  // The curves are live (patched while a point is dragged) and live layers draw no legend: the key is a readout.
  const colours = categorical(useTheme().resolved)
  const xa = useAxis({ label: 'x', range: [0, 10] })
  const ya = useAxis({ label: 'f(x)', range: [-1, 7] })
  const yd = useAxis({ label: 'f″(x)', hold: 'initial', key: focus })
  return (
    <Figure
      title="Six interpolants through the same points"
      defaultSize="L"
      purpose="Cubic splines are C² but overshoot between points; PCHIP and Akima give up the continuous second derivative to stay within the data's shape."
      state={state}
      readouts={{
        interpolants: METHODS.map((m, i) => <Readout key={m} label={m} value="" color={colours[i]} />),
        [focus]: (
          <>
            <Readout label="overshoot above the highest point" value={formatValue(overshoot)} />
            {!distinct && <Readout label="note" value="two points share an x; interpolants need distinct x" />}
          </>
        ),
      }}
      caption="Drag the points. The chosen interpolant is drawn thick, the others thin. Put two neighbours at the same height: the cubic splines bulge between them, PCHIP stays flat. The lower panel shows the chosen interpolant's f″: continuous for the cubic splines, zero for linear, jumps at the knots for PCHIP and Akima. The degree-6 polynomial swings far outside the data."
    >
      <Plots rows={2} heights={[2, 1]}>
        <Plot x={xa} y={ya}>
          {(curves ?? []).map((c, i) => (
            <Curve
              key={c.m}
              name={c.m}
              x={GRID_X}
              y={c.f}
              slot={i}
              thin={c.m !== focus}
              width={c.m === focus ? 3 : undefined}
              live
            />
          ))}
          <Points name="points" x={px} y={py} emphasis live />
          {START.map((_, i) => (
            <Handle key={i} {...state.handle([`p${i}x`, `p${i}y`] as unknown as readonly [string, string])} />
          ))}
        </Plot>
        <Plot x={xa} y={yd}>
          {d2.value && (
            <Curve name={`${focus}: f″(x)`} x={GRID_X} y={d2.value} slot={METHODS.indexOf(focus)} stale={d2.stale} />
          )}
        </Plot>
      </Plots>
    </Figure>
  )
}

const LOG_LAMBDA = linspace(-4, 5, 91)
const LOG_LAMBDA_X = toFlat(LOG_LAMBDA)
const TRUTH = GRID_X.map((v) => Math.sin(v) + 0.3 * Math.cos(3 * v))

/** P-spline smoothing against λ, with the GCV curve that chooses it. */
export function PsplineSmoothing() {
  const state = useFigureState({
    smoother: row('1 · smoother', {
      logLambda: slider(-4, 5, 0, { label: 'log₁₀ λ' }),
      segments: slider(5, 40, 20, { step: 1, label: 'segments' }),
    }),
    reveal: row('2 · reveal', { showBasis: toggle(false, 'scaled B-splines cⱼBⱼ(x)') }),
  })
  const { logLambda, segments } = state.smoother
  const showBasis = state.reveal.showBasis
  const data = useMemo(() => {
    const s = stream('pspline')
    const x = toFlat(uniform(child(s, 'x'), 0, 10, { shape: [100] }) as Tensor)
    const e = toFlat(normals(child(s, 'e'), 100, 0, 0.35))
    return { x, y: x.map((v, i) => Math.sin(v) + 0.3 * Math.cos(3 * v) + e[i]) }
  }, [])
  const x = useMemo(() => tensor(data.x), [data])
  const y = useMemo(() => tensor(data.y), [data])
  const options = useMemo(() => ({ segments, range: [0, 10] as const }), [segments])
  const fit = useMemo(() => pspline(x, y, { ...options, lambda: 10 ** logLambda }), [x, y, options, logLambda])
  const path = useMemo(() => psplineGcvPath(x, y, LOG_LAMBDA, options), [x, y, options])
  const gcv = useMemo(() => toFlat(path.gcv), [path])
  const best = argmin(path.gcv) as number
  const band = useMemo(() => {
    const f = toFlat(fit.evaluate(GRID))
    const se = toFlat(fit.standardError(GRID))
    return { f, hi: f.map((v, i) => v + 2 * se[i]), lo: f.map((v, i) => v - 2 * se[i]) }
  }, [fit])
  const basis = useMemo(() => {
    if (!showBasis) return []
    const B = toRows(bsplineBasis(GRID, uniformKnots(0, 10, segments, 3), 3))
    return toFlat(fit.coefficients).map((cj, j) => B.map((r) => r[j] * cj))
  }, [showBasis, segments, fit])
  const xa = useAxis({ label: 'x', range: [0, 10] })
  const ya = useAxis({ label: 'y', range: [-2.5, 2.5] })
  const la = useAxis({ label: 'log₁₀ λ', range: [-4, 5] })
  const ga = useAxis({ label: 'GCV', hold: 'union', key: segments })
  return (
    <Figure
      title="P-spline smoothing against λ"
      defaultSize="L"
      purpose="A P-spline penalises second differences of neighbouring B-spline coefficients; λ moves the fit from interpolating noise to a straight line, and GCV picks a λ in between."
      state={state}
      readouts={
        <>
          <Readout label="effective degrees of freedom" value={formatValue(fit.edf)} />
          <Readout label="GCV" value={formatValue(fit.gcv)} />
          <Readout label="σ̂" value={formatValue(Math.sqrt(fit.sigma2))} />
          <Readout label="GCV-optimal log₁₀ λ" value={formatValue(LOG_LAMBDA_X[best])} />
        </>
      }
      caption="Drag the vertical line on the GCV curve (or the slider). Small λ gives a wiggly fit with many effective degrees of freedom; large λ flattens it towards a line, which the second-difference penalty leaves unpenalised. Once there are enough segments, adding more barely changes the fit: the penalty, not the knots, sets the smoothness."
    >
      <Plots rows={2} heights={[2, 1]}>
        <Plot x={xa} y={ya}>
          <Points name="data" x={data.x} y={data.y} muted />
          {basis.map((b, j) => (
            <Curve key={j} name="cⱼBⱼ(x)" x={GRID_X} y={b} thin slot={3} silent />
          ))}
          <Area name="± 2 se" x={GRID_X} y={band.hi} base={band.lo} slot={0} line={false} live />
          <Curve name="fit" x={GRID_X} y={band.f} slot={0} live />
          <Curve name="truth" x={GRID_X} y={TRUTH} slot={2} dashed />
        </Plot>
        <Plot x={la} y={ga}>
          <Curve name="GCV(λ)" x={LOG_LAMBDA_X} y={gcv} slot={1} />
          <Points name="minimum" x={[LOG_LAMBDA_X[best]]} y={[gcv[best]]} emphasis />
          <Handle {...state.handle('smoother.logLambda', { label: 'λ' })} />
        </Plot>
      </Plots>
    </Figure>
  )
}
