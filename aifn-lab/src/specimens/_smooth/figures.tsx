import { useMemo, useState } from 'react'
import { normals, stream, uniform } from 'aifn/random'
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
} from 'aifn/smooth'
import { argmin, linspace, tensor, toFlat, toRows, type Tensor } from 'aifn/tensor'
import { Select, Slider, Switch } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { Panel, Readout, Subplots, XYChart, type Handle, type XYSeries } from '@lab/viz'
import { formatValue } from '@lab/views'

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))
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

/** Interpolants through the same points: smoothness against overshoot. */
export function InterpolantsCompared() {
  const [points, setPoints] = useState(START)
  const [focus, setFocus] = useState<MethodName>('PCHIP')
  const sorted = useMemo(() => [...points].sort((a, b) => a[0] - b[0]), [points])
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
  const top: XYSeries[] = [
    ...(curves ?? []).map((c, i) => ({
      name: c.m,
      type: 'line' as const,
      x: GRID_X,
      y: c.f,
      slot: i,
      thin: c.m !== focus,
    })),
    { name: 'points', type: 'scatter', x: sorted.map((p) => p[0]), y: sorted.map((p) => p[1]), emphasis: true },
  ]
  const chosen = curves?.find((c) => c.m === focus)
  const bottom: XYSeries[] = chosen
    ? [{ name: `${focus}: f″(x)`, type: 'line', x: GRID_X, y: chosen.d2, slot: METHODS.indexOf(focus) }]
    : []
  const handles: Handle[] = points.map((p, i) => ({
    kind: 'point',
    at: p,
    onDrag: ([a, b]) => setPoints((ps) => ps.map((q, j) => (j === i ? [clamp(a, 0, 10), clamp(b, -1, 7)] : q))),
  }))
  const overshoot = chosen ? Math.max(...chosen.f) - Math.max(...sorted.map((p) => p[1])) : NaN
  return (
    <Figure
      title="Six interpolants through the same points"
      defaultSize="L"
      description="Cubic splines are C² but overshoot between points; PCHIP and Akima give up the continuous second derivative to stay within the data's shape."
      controls={
        <ControlRow label="Show the second derivative of">
          <Select label="interpolant" value={focus} onChange={setFocus} options={[...METHODS]} />
        </ControlRow>
      }
      readouts={
        <>
          <Readout label="overshoot above the highest point" value={formatValue(overshoot)} />
          {!distinct && <Readout label="note" value="two points share an x; interpolants need distinct x" />}
        </>
      }
      caption="Drag the points. Put two neighbours at the same height: the cubic splines bulge between them, PCHIP stays flat. The lower panel shows the chosen interpolant's f″: continuous for the cubic splines, a step function for linear (zero), jumps at the knots for PCHIP and Akima. The degree-6 polynomial swings far outside the data."
    >
      <Subplots rows={2} sharex heightRatios={[2, 1]}>
        <Panel>
          <XYChart series={top} xLabel="x" yLabel="f(x)" xRange={[0, 10]} yRange={[-1, 7]} handles={handles} />
        </Panel>
        <Panel rescaleOnChange={false} axisKey={focus}>
          <XYChart series={bottom} xLabel="x" yLabel="f″(x)" />
        </Panel>
      </Subplots>
    </Figure>
  )
}

const LOG_LAMBDA = linspace(-4, 5, 91)
const LOG_LAMBDA_X = toFlat(LOG_LAMBDA)

/** P-spline smoothing against λ, with the GCV curve that chooses it. */
export function PsplineSmoothing() {
  const [logLambda, setLogLambda] = useState(0)
  const [segments, setSegments] = useState(20)
  const [showBasis, setShowBasis] = useState(false)
  const data = useMemo(() => {
    const s = stream('pspline')
    const x = toFlat(uniform(s.child('x'), 0, 10, { shape: [100] }) as Tensor)
    const e = toFlat(normals(s.child('e'), 100, 0, 0.35))
    return { x, y: x.map((v, i) => Math.sin(v) + 0.3 * Math.cos(3 * v) + e[i]) }
  }, [])
  const x = useMemo(() => tensor(data.x), [data])
  const y = useMemo(() => tensor(data.y), [data])
  const options = useMemo(() => ({ segments, range: [0, 10] as const }), [segments])
  const fit = useMemo(() => pspline(x, y, { ...options, lambda: 10 ** logLambda }), [x, y, options, logLambda])
  const path = useMemo(() => psplineGcvPath(x, y, LOG_LAMBDA, options), [x, y, options])
  const gcv = toFlat(path.gcv)
  const best = argmin(path.gcv) as number
  const f = toFlat(fit.evaluate(GRID))
  const se = toFlat(fit.standardError(GRID))
  const basis = useMemo(() => {
    if (!showBasis) return []
    const B = toRows(bsplineBasis(GRID, uniformKnots(0, 10, segments, 3), 3))
    const c = toFlat(fit.coefficients)
    return c.map((cj, j): XYSeries => ({
      name: `c${j} B${j}(x)`,
      type: 'line',
      x: GRID_X,
      y: B.map((r) => r[j] * cj),
      thin: true,
      slot: 3,
    }))
  }, [showBasis, segments, fit])
  const top: XYSeries[] = [
    { name: 'data', type: 'scatter', x: data.x, y: data.y, muted: true },
    ...basis,
    { name: 'fit', type: 'line', x: GRID_X, y: f, slot: 0 },
    { name: '± 2 se', type: 'line', x: GRID_X, y: f.map((v, i) => v + 2 * se[i]), slot: 0, dashed: true },
    { name: '± 2 se ', type: 'line', x: GRID_X, y: f.map((v, i) => v - 2 * se[i]), slot: 0, dashed: true },
    {
      name: 'truth',
      type: 'line',
      x: GRID_X,
      y: GRID_X.map((v) => Math.sin(v) + 0.3 * Math.cos(3 * v)),
      slot: 2,
      dashed: true,
    },
  ]
  const bottom: XYSeries[] = [
    { name: 'GCV(λ)', type: 'line', x: LOG_LAMBDA_X, y: gcv, slot: 1 },
    { name: 'minimum', type: 'scatter', x: [LOG_LAMBDA_X[best]], y: [gcv[best]], emphasis: true },
  ]
  return (
    <Figure
      title="P-spline smoothing against λ"
      defaultSize="L"
      description="A P-spline penalises second differences of neighbouring B-spline coefficients; λ moves the fit from interpolating noise to a straight line, and GCV picks a λ in between."
      controls={
        <>
          <ControlRow label="Smoother">
            <Slider label="log₁₀ λ" value={logLambda} onChange={setLogLambda} min={-4} max={5} />
            <Slider label="segments" value={segments} onChange={setSegments} min={5} max={40} step={1} />
          </ControlRow>
          <ControlRow label="Reveal">
            <Switch label="scaled B-splines cⱼBⱼ(x)" checked={showBasis} onChange={setShowBasis} />
          </ControlRow>
        </>
      }
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
      <Subplots rows={2} heightRatios={[2, 1]}>
        <Panel>
          <XYChart series={top} xLabel="x" yLabel="y" xRange={[0, 10]} yRange={[-2.5, 2.5]} />
        </Panel>
        <Panel>
          <XYChart
            series={bottom}
            xLabel="log₁₀ λ"
            yLabel="GCV"
            yLog
            handles={[{ kind: 'x', at: logLambda, label: 'λ', onDrag: (v) => setLogLambda(clamp(v, -4, 5)) }]}
          />
        </Panel>
      </Subplots>
    </Figure>
  )
}
