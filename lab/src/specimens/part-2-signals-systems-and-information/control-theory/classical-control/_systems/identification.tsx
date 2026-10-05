/**
 * "System identification": a known second-order discrete system driven by a random binary input, with equation,
 * coloured or output noise; ARX by least squares, ARMAX by the prediction-error method and N4SID recover it (core
 * `arx`, `arxOrderSelection`, `predictionErrorMethod`, `n4sid`), with model-order selection by AIC and by the singular
 * values of the subspace projection.
 */
import { useMemo } from 'react'
import {
  arx,
  arxOrderSelection,
  n4sid,
  poles,
  predictionErrorMethod,
  polynomialModel,
  stepResponse,
  transferFunction,
} from 'aifn-compute/systems'
import { linearFilter } from 'aifn-compute/foundation/convolution'
import { child, normal, stream } from 'aifn-compute/foundation/random'
import { fromData, toComplexFlat, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import type { LtiSystem } from 'aifn-compute/foundation/contracts'
import { Figure } from 'aifn-render/layout'
import { choice, int, row, slider, useFigureState } from 'aifn-render/state'
import { Annotation, Bars, Curve, Plot, Plots, Points, Readout, formatNumber, useAxis } from 'aifn-render/viz'

const fmt = (v: number) => formatNumber(v)
const A = [1, -1.5, 0.7]
const B = [0, 1, 0.5]
const C = [1, 0.6, 0.3]
const NOISE = {
  equation: 'equation noise e/A (ARX is right)',
  coloured: 'coloured noise Ce/A (ARMAX is right)',
  output: 'output noise e (output error)',
} as const
const CIRCLE = Array.from({ length: 121 }, (_, k) => (2 * Math.PI * k) / 120)
const METHODS = ['ARX', 'ARMAX (PEM)', 'N4SID'] as const

const filt = (b: number[], a: number[], x: Float64Array) =>
  Float64Array.from(toFlat(linearFilter(b, a, fromData(x, [x.length])) as Tensor))
const stepOf = (sys: LtiSystem) => toFlat(stepResponse(sys, { tEnd: 40 }).y)
const polesOf = (sys: LtiSystem) => toComplexFlat(poles(sys))

export function IdentificationSpecimen() {
  const state = useFigureState({
    data: row('1 · data', {
      noise: choice(
        Object.entries(NOISE).map(([value, label]) => ({ value, label })),
        'coloured',
        { label: 'noise' },
      ),
      sigma: slider(0, 1.5, 0.5, { label: 'noise σ', step: 0.05 }),
      samples: int(600, { ge: 100, le: 5000, suggestions: [200, 600, 2000], label: 'samples N' }),
      seed: int(0, { ge: 0, le: 9999, label: 'seed' }),
    }),
    model: row('2 · models', {
      order: slider(1, 6, 2, { label: 'ARX / ARMAX order n (na = nb = nc)', step: 1 }),
      horizon: int(10, { ge: 4, le: 30, suggestions: [6, 10, 15], label: 'N4SID horizon i' }),
      pemSteps: int(30, { ge: 1, le: 200, suggestions: [10, 30, 100], label: 'PEM iterations' }),
    }),
  })
  const { noise, sigma, samples, seed } = state.data
  const { order, horizon, pemSteps } = state.model
  const data = useMemo(() => {
    const root = stream(`sysid-${seed}`)
    const u = Float64Array.from(toFlat(normal(child(root, 'u'), 0, 1, { shape: [samples] }) as Tensor), (v) =>
      v >= 0 ? 1 : -1,
    )
    const e = Float64Array.from(toFlat(normal(child(root, 'e'), 0, sigma, { shape: [samples] }) as Tensor))
    const clean = filt(B, A, u)
    const v = noise === 'equation' ? filt([1], A, e) : noise === 'coloured' ? filt(C, A, e) : e
    return { u, y: clean.map((c, i) => c + v[i]), clean }
  }, [noise, sigma, samples, seed])
  const fits = useMemo(() => {
    const a = arx(data.y, data.u, { na: order, nb: order })
    const tr = trace(predictionErrorMethod(data.y, data.u, { na: order, nb: order, nc: order }), {}, pemSteps, {
      record: { loss: (s) => s.loss },
    })
    const pem = polynomialModel(
      data.y,
      data.u,
      { na: order, nb: order, nc: order },
      { theta0: tr.final.theta, maxSteps: 0 },
    )
    let sub: ReturnType<typeof n4sid> | null = null
    try {
      sub = n4sid(data.y, data.u, { horizon })
    } catch {
      sub = null
    }
    return { arx: a, pem, pemLoss: toFlat(tr.series.loss), sub }
  }, [data, order, pemSteps, horizon])
  const selection = useMemo(() => arxOrderSelection(data.y, data.u, { maxOrder: 6 }), [data])
  const truth = useMemo(() => transferFunction(B, A, { dt: 1 }), [])
  const systems: (LtiSystem | null)[] = [fits.arx.system, fits.pem.system, fits.sub?.system ?? null]
  const steps = useMemo(() => systems.map((s) => (s ? stepOf(s) : null)), [fits]) // eslint-disable-line react-hooks/exhaustive-deps
  const trueStep = useMemo(() => stepOf(truth), [truth])
  const trueP = useMemo(() => polesOf(truth), [truth])
  const k = trueStep.map((_, i) => i)
  const sv = fits.sub ? toFlat(fits.sub.singularValues) : []
  const t = useAxis({ label: 'sample k' })
  const ys = useAxis({ label: 'step response', hold: 'union', key: noise })
  const zr = useAxis({ label: 'Re z', range: [-1.3, 1.3] })
  const zi = useAxis({ label: 'Im z', range: [-1.3, 1.3], equal: zr })
  const ord = useAxis({ label: 'model order', integer: true, range: [0.5, 8.5] })
  const crit = useAxis({ label: 'AIC (ARX), log σ (N4SID)', hold: 'union', key: `${noise}${samples}` })
  const it = useAxis({ label: 'PEM iteration', integer: true })
  const loss = useAxis({ label: 'mean squared prediction error', log: true, hold: 'union', key: `${noise}${order}` })
  const bestAic = selection.reduce((a, b) => (b.aic < a.aic ? b : a))
  const svLog = sv.map((v) => Math.log10(Math.max(v, 1e-12)))
  // AIC and log σ share one axis by rescaling log σ to AIC's range.
  const aic = selection.map((s) => s.aic)
  const [lo, hi] = [Math.min(...aic), Math.max(...aic)]
  const [slo, shi] = svLog.length ? [Math.min(...svLog.slice(0, 8)), Math.max(...svLog)] : [0, 1]
  const svScaled = svLog.slice(0, 8).map((v) => lo + ((v - slo) / (shi - slo || 1)) * (hi - lo))
  return (
    <Figure
      title="System identification"
      purpose="A known system G = (q⁻¹ + 0.5q⁻²)/(1 − 1.5q⁻¹ + 0.7q⁻²) recovered from noisy input–output data: ARX in one least-squares solve, ARMAX by iterating the prediction-error method, and N4SID from an SVD. With coloured noise ARX is biased and ARMAX is not."
      defaultSize="XL"
      state={state}
      readouts={{
        fits: (
          <>
            <Readout label="ARX loss" value={fmt(fits.arx.loss)} />
            <Readout label="ARMAX loss" value={fmt(fits.pem.loss)} />
            <Readout label="N4SID order (largest singular-value gap)" value={fits.sub ? fits.sub.order : 'failed'} />
            <Readout label="ARX order by AIC" value={bestAic.order} />
          </>
        ),
        poles: (
          <>
            {METHODS.map((name, j) => (
              <Readout
                key={name}
                label={`${name} poles`}
                value={
                  systems[j]
                    ? polesOf(systems[j]!)
                        .map((c) => `${fmt(c.re)}${c.im >= 0 ? '+' : '−'}${fmt(Math.abs(c.im))}i`)
                        .join(', ')
                    : '—'
                }
              />
            ))}
            <Readout
              label="true poles"
              value={trueP.map((c) => `${fmt(c.re)}${c.im >= 0 ? '+' : '−'}${fmt(Math.abs(c.im))}i`).join(', ')}
            />
          </>
        ),
      }}
      caption={`N = ${samples} samples of a random ±1 input drive the true system; noise of standard deviation σ enters as chosen. Step responses: the truth in ink, each identified model in its colour. z-plane: true poles (ink), identified poles by method, the unit circle in grey. Order selection: ARX's Akaike criterion against the order (the minimum, marked, picks the order) and N4SID's singular values on a log scale rescaled onto the same axis (the order is where they drop). PEM: the prediction error of ARMAX falling over its damped Gauss–Newton iterations from the ARX start. Raise the order past 2 to see over-fitted models add poles that nearly cancel zeros; with coloured noise and low order, ARX's poles miss and ARMAX's do not.`}
    >
      <Plots cols={2} rows={2}>
        <Plot x={t} y={ys} title="step responses">
          <Curve name="true" x={k} y={trueStep} emphasis />
          {METHODS.map((name, j) =>
            steps[j] ? <Curve key={name} name={name} x={k} y={steps[j]!} slot={j} dashed={j === 2} /> : null,
          )}
        </Plot>
        <Plot x={zr} y={zi} title="poles in the z-plane">
          <Curve name="unit circle" x={CIRCLE.map(Math.cos)} y={CIRCLE.map(Math.sin)} muted />
          <Points name="true" x={trueP.map((c) => c.re)} y={trueP.map((c) => c.im)} emphasis size={7} />
          {METHODS.map((name, j) => {
            const p = systems[j] ? polesOf(systems[j]!) : []
            return <Points key={name} name={name} x={p.map((c) => c.re)} y={p.map((c) => c.im)} slot={j} size={9} />
          })}
        </Plot>
        <Plot x={ord} y={crit} title="order selection">
          <Curve name="ARX AIC" x={selection.map((s) => s.order)} y={aic} slot={0} showPoints />
          <Annotation x={bestAic.order} dashed />
          {svScaled.length > 0 && (
            <Bars
              name="N4SID log σ (rescaled)"
              x={svScaled.map((_, i) => i + 1)}
              y={svScaled}
              base={lo}
              slot={2}
              opacity={0.5}
            />
          )}
        </Plot>
        <Plot x={it} y={loss} title="prediction-error method">
          <Curve name="ARMAX loss" x={fits.pemLoss.map((_, i) => i)} y={fits.pemLoss} slot={1} showPoints />
          <Annotation y={Math.max(fits.arx.loss, 1e-12)} dashed text="ARX loss" />
        </Plot>
      </Plots>
    </Figure>
  )
}
