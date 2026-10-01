import { useMemo } from 'react'
import { boxCox, powerTransform, splineFeatures, yeoJohnson } from 'aifn-applied/learning/preprocessing'
import { normals, stream } from 'aifn/foundation/random'
import { fromData, linspace, toFlat, toRows } from 'aifn/foundation/tensor'
import { Dashboard, DashboardCell, DashboardRow, Figure } from '@lab/layout'
import { choice, row, slider, useFigureState } from '@lab/state'
import { Annotation, Curve, Histogram, Plot, Readout, useAxis } from '@lab/viz'
import { formatValue } from '@lab/views'

const grid = (lo: number, hi: number, n: number) => toFlat(linspace(lo, hi, n))

/** Population variance. */
function variance(v: readonly number[]): number {
  const m = v.reduce((a, b) => a + b, 0) / v.length
  return v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length
}

type Method = 'box-cox' | 'yeo-johnson'

export function PowerTransformSpecimen() {
  const state = useFigureState({
    data: row('1 · data', {
      sigma: slider(0.05, 1.5, 0.8, { label: 'log-normal σ (skew)' }),
      shift: slider(-3, 3, 0, { label: 'shift' }),
    }),
    transform: row('2 · transform', {
      method: choice(['box-cox', 'yeo-johnson'] as Method[], 'box-cox', { label: 'method' }),
    }),
  })
  const { sigma, shift } = state.data
  const method = state.transform.method as Method
  const x = useMemo(
    () => toFlat(normals(stream('power'), [400])).map((z) => Math.exp(0.5 + sigma * z) + shift),
    [sigma, shift],
  )
  const usable = method === 'yeo-johnson' || Math.min(...x) > 0
  const fitted = useMemo(
    () => (usable ? powerTransform({ method }).fit({ x: fromData(Float64Array.from(x), [x.length, 1]) }) : null),
    [x, method, usable],
  )
  const lambdas = useMemo(() => grid(-1.5, 2, 141), [])
  const profile = useMemo(() => {
    if (!usable) return []
    // The profile log-likelihood the fit maximises (up to a constant). TODO(aifn): aifn's λ searches keep this
    // function private; the lab should evaluate it from aifn (component request in phase5c-3.md).
    const jac =
      method === 'box-cox'
        ? x.reduce((a, v) => a + Math.log(v), 0)
        : x.reduce((a, v) => a + Math.sign(v) * Math.log1p(Math.abs(v)), 0)
    return lambdas.map((l) => {
      const t = x.map((v) => (method === 'box-cox' ? boxCox(v, l) : yeoJohnson(v, l)))
      return (l - 1) * jac - (x.length / 2) * Math.log(variance(t))
    })
  }, [x, method, usable, lambdas])
  const z = useMemo(
    () => (fitted ? toFlat(fitted.transform(fromData(Float64Array.from(x), [x.length, 1]))) : []),
    [fitted, x],
  )
  const lambda = fitted?.lambdas[0] ?? NaN
  const best = profile.length ? Math.max(...profile) : 0
  const xa = useAxis({ label: 'x' })
  const da = useAxis({ label: 'density' })
  const za = useAxis({ label: 'transformed, standardised', range: [-4, 4] })
  const dz = useAxis({ label: 'density', range: [0, 0.6] })
  const la = useAxis({ label: 'λ', range: [-1.5, 2] })
  const ll = useAxis({ label: 'log-likelihood', range: [best - 50, best + 5] })
  return (
    <Figure
      purpose="A power transform with λ chosen by maximum likelihood makes skewed data look Gaussian: for log-normal data Box–Cox picks λ ≈ 0, the log."
      title="Power transforms with λ by maximum likelihood"
      state={state}
      defaultSize="L"
      readouts={
        <>
          <Readout label="λ" value={formatValue(lambda)} />
          <Readout label="Brent iterations" value={fitted?.searches[0].iterations ?? '–'} />
          {!usable && <Readout label="Box–Cox" value="needs positive data: use Yeo–Johnson" />}
        </>
      }
      caption="Top: 400 draws of exp(0.5 + σZ) + shift before and after the transform (standardised). Bottom: the profile log-likelihood over λ, with the maximum found by Brent's method (the vertical line). A negative shift makes Box–Cox inapplicable; Yeo–Johnson still applies."
    >
      <Dashboard>
        <DashboardRow minHeight={220}>
          <DashboardCell>
            <Plot x={xa} y={da}>
              <Histogram name="x" values={x} bins={30} slot={0} />
            </Plot>
          </DashboardCell>
          <DashboardCell>
            <Plot x={za} y={dz}>
              {z.length > 0 && <Histogram name="transformed" values={z} bins={30} slot={1} />}
            </Plot>
          </DashboardCell>
        </DashboardRow>
        <DashboardRow ratio={0.8} minHeight={200}>
          <DashboardCell>
            <Plot x={la} y={ll}>
              <Curve name="profile log-likelihood" x={lambdas} y={profile} slot={2} />
              {usable && <Annotation x={lambda} text="maximum" />}
            </Plot>
          </DashboardCell>
        </DashboardRow>
      </Dashboard>
    </Figure>
  )
}

type Extrapolation = 'constant' | 'continue'

export function SplineBasisSpecimen() {
  const state = useFigureState({
    basis: row('1 · basis', {
      knots: slider(2, 12, 5, { label: 'knots', step: 1 }),
      degree: slider(0, 5, 3, { label: 'degree', step: 1 }),
    }),
    outside: row('2 · outside [0, 1]', {
      extrapolation: choice(['constant', 'continue'] as Extrapolation[], 'constant', { label: 'extrapolation' }),
    }),
  })
  const { knots, degree } = state.basis
  const extrapolation = state.outside.extrapolation as Extrapolation
  const xs = useMemo(() => grid(-0.4, 1.4, 361), [])
  const { columns, sum, model } = useMemo(() => {
    const model = splineFeatures({ knots, degree, extrapolation }).fit({ x: fromData(Float64Array.of(0, 1), [2, 1]) })
    const B = toRows(model.transform(fromData(Float64Array.from(xs), [xs.length, 1]))) as number[][]
    const k = B[0].length
    return {
      columns: Array.from({ length: k }, (_, j) => B.map((r) => r[j])),
      sum: B.map((r) => r.reduce((a, b) => a + b, 0)),
      model,
    }
  }, [knots, degree, extrapolation, xs])
  const xa = useAxis({ label: 'x', range: [-0.4, 1.4] })
  const ya = useAxis({ label: 'B(x)', hold: 'union', key: extrapolation })
  return (
    <Figure
      purpose="B-spline features turn one column into knots + degree − 1 local bumps that sum to one on the training range, so a linear model on them fits a smooth curve."
      title="B-spline features"
      state={state}
      readouts={
        <>
          <Readout label="features per column" value={model.perColumn} />
          <Readout
            label="knot vector"
            value={toFlat(model.knots)
              .map((v) => formatValue(v))
              .join(', ')}
          />
        </>
      }
      caption="splineFeatures fitted on [0, 1]: each coloured curve is one feature Bⱼ(x); the dashed line is their sum. Outside the training range, 'constant' holds the boundary values and 'continue' extends the end polynomials."
    >
      <Plot x={xa} y={ya} legend={false}>
        {columns.map((c, j) => (
          <Curve key={j} name={`B${j}`} x={xs} y={c} slot={j % 8} />
        ))}
        <Curve name="Σ B" x={xs} y={sum} emphasis dashed />
      </Plot>
    </Figure>
  )
}
