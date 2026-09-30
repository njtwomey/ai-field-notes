import { useMemo, useState } from 'react'
import { boxCox, powerTransform, splineFeatures, yeoJohnson } from 'aifn-applied/learning/preprocessing'
import { normals, stream } from 'aifn/foundation/random'
import { histogram } from 'aifn/probability/stats'
import { fromData, linspace, toFlat, toRows } from 'aifn/foundation/tensor'
import { Select, Slider } from '@lab/controls'
import { Figure } from '@lab/layout'
import { ChartSize, Readout, XYChart, type XYSeries } from '@lab/viz'
import { formatValue } from '@lab/views'

const grid = (lo: number, hi: number, n: number) => toFlat(linspace(lo, hi, n))

/** Population variance. */
function variance(v: readonly number[]): number {
  const m = v.reduce((a, b) => a + b, 0) / v.length
  return v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length
}

function bars(values: readonly number[], name: string, slot: number): XYSeries {
  const h = histogram(values, { bins: 30 })
  const edges = toFlat(h.edges)
  return {
    name,
    type: 'bar',
    x: edges.slice(1).map((e, i) => (edges[i] + e) / 2),
    y: toFlat(h.density),
    slot,
  }
}

type Method = 'box-cox' | 'yeo-johnson'

export function PowerTransformSpecimen() {
  const [sigma, setSigma] = useState(0.8)
  const [shift, setShift] = useState(0)
  const [method, setMethod] = useState<Method>('box-cox')
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
    // The profile log-likelihood the fit maximises (up to a constant).
    const jac =
      method === 'box-cox'
        ? x.reduce((a, v) => a + Math.log(v), 0)
        : x.reduce((a, v) => a + Math.sign(v) * Math.log1p(Math.abs(v)), 0)
    return lambdas.map((l) => {
      const t = x.map((v) => (method === 'box-cox' ? boxCox(v, l) : yeoJohnson(v, l)))
      return (l - 1) * jac - (x.length / 2) * Math.log(variance(t))
    })
  }, [x, method, usable, lambdas])
  const z = fitted ? toFlat(fitted.transform(fromData(Float64Array.from(x), [x.length, 1]))) : []
  const lambda = fitted?.lambdas[0] ?? NaN
  const best = profile.length ? Math.max(...profile) : 0
  return (
    <Figure
      title="Power transforms with λ by maximum likelihood"
      defaultSize="L"
      controls={
        <>
          <Select label="method" value={method} onChange={setMethod} options={['box-cox', 'yeo-johnson']} />
          <Slider label="log-normal σ (skew)" value={sigma} min={0.05} max={1.5} onChange={setSigma} />
          <Slider label="shift" value={shift} min={-3} max={3} onChange={setShift} />
        </>
      }
      readouts={
        <>
          <Readout label="λ" value={formatValue(lambda)} />
          <Readout label="Brent iterations" value={fitted?.searches[0].iterations ?? '–'} />
          {!usable && <Readout label="Box–Cox" value="needs positive data: use Yeo–Johnson" />}
        </>
      }
      caption="Draws of exp(0.5 + σZ) + shift before and after the transform (standardised), and the profile log-likelihood over λ with the maximum found by Brent's method. For log-normal data Box–Cox finds λ ≈ 0, the log. A negative shift makes Box–Cox inapplicable; Yeo–Johnson still applies."
    >
      <div className="grid gap-3 md:grid-cols-2">
        <ChartSize scale={0.6}>
          <XYChart series={[bars(x, 'x', 0)]} xLabel="x" yLabel="density" />
        </ChartSize>
        <ChartSize scale={0.6}>
          <XYChart
            series={z.length ? [bars(z, 'transformed', 1)] : []}
            xLabel="transformed, standardised"
            yLabel="density"
          />
        </ChartSize>
      </div>
      <ChartSize scale={0.5}>
        <XYChart
          series={[
            { name: 'profile log-likelihood', type: 'line', x: lambdas, y: profile, slot: 2 },
            { name: 'λ̂', type: 'line', x: [lambda, lambda], y: [best - 50, best], emphasis: true },
          ]}
          xLabel="λ"
          yLabel="log-likelihood"
          yRange={[best - 50, best + 5]}
        />
      </ChartSize>
    </Figure>
  )
}

type Extrapolation = 'constant' | 'continue'

export function SplineBasisSpecimen() {
  const [knots, setKnots] = useState(5)
  const [degree, setDegree] = useState(3)
  const [extrapolation, setExtrapolation] = useState<Extrapolation>('constant')
  const xs = useMemo(() => grid(-0.4, 1.4, 361), [])
  const { series, model } = useMemo(() => {
    const model = splineFeatures({ knots, degree, extrapolation }).fit({ x: fromData(Float64Array.of(0, 1), [2, 1]) })
    const B = toRows(model.transform(fromData(Float64Array.from(xs), [xs.length, 1])))
    const k = B[0].length
    const series: XYSeries[] = Array.from({ length: k }, (_, j) => ({
      name: `B${j}`,
      type: 'line',
      x: xs,
      y: B.map((r) => r[j]),
      slot: j % 8,
    }))
    series.push({
      name: 'Σ B',
      type: 'line',
      x: xs,
      y: B.map((r) => r.reduce((a, b) => a + b, 0)),
      emphasis: true,
      dashed: true,
    })
    return { series, model }
  }, [knots, degree, extrapolation, xs])
  return (
    <Figure
      title="B-spline features"
      controls={
        <>
          <Slider label="knots" value={knots} min={2} max={12} step={1} onChange={setKnots} />
          <Slider label="degree" value={degree} min={0} max={5} step={1} onChange={setDegree} />
          <Select
            label="extrapolation"
            value={extrapolation}
            onChange={setExtrapolation}
            options={['constant', 'continue']}
          />
        </>
      }
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
      caption="splineFeatures fitted on [0, 1]: knots + degree − 1 B-splines that sum to one inside the training range. Outside it, 'constant' holds the boundary values and 'continue' extends the end polynomials."
    >
      <XYChart series={series} xLabel="x" yLabel="B(x)" xRange={[-0.4, 1.4]} />
    </Figure>
  )
}
