import { useMemo } from 'react'
import {
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  Segments,
  seriesLayers,
  type SeriesSpec,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'

const X_MIN = -6
const X_MAX = 0
const N_POINTS = 241
const LOG_P = Array.from({ length: N_POINTS }, (_, i) => X_MIN + ((X_MAX - X_MIN) * i) / (N_POINTS - 1))
const X_RANGE: [number, number] = [X_MIN, X_MAX]
const Y_RANGE: [number, number] = [0, 1]

/** Population optimum of gBCE with sampling rate α and power β (β = 1 is plain BCE): βp / (βp + α(1 − p)). */
const predicted = (p: number, alpha: number, beta: number) => (beta * p) / (beta * p + alpha * (1 - p))

/** Sum of predicted probabilities over a catalogue whose true next-item distribution is Zipf, p_i ∝ 1/i. */
function zipfSum(n: number, alpha: number, beta: number) {
  let h = 0
  for (let i = 1; i <= n; i++) h += 1 / i
  let s = 0
  for (let i = 1; i <= n; i++) s += predicted(1 / (i * h), alpha, beta)
  return s
}

const pow10 = (v: number) => Math.round(10 ** v).toLocaleString('en-GB')

/**
 * The probability a model trained with binary cross-entropy and k uniform negatives assigns to an item, against the
 * item's true next-item probability, for plain BCE and for gBCE with calibration parameter t.
 */
export function OverconfidenceLab() {
  const state = useFigureState({
    logN: float(4, { min: 2, max: 6, step: 0.1, label: 'catalogue size |I|', format: pow10 }),
    logK: int(0, { min: 0, max: 10, step: 1, label: 'negatives per positive k', format: (v) => String(2 ** v) }),
    t: float(0.75, { min: 0, max: 1, step: 0.01, label: 'calibration t' }),
    logP: slider(X_MIN, -0.01, -2, { step: 0.01, label: 'true probability p', format: (v) => formatNumber(10 ** v) }),
  })

  const n = Math.round(10 ** state.logN)
  const k = Math.min(2 ** state.logK, n - 1)
  const alpha = k / (n - 1)
  const beta = alpha * (state.t * (1 - 1 / alpha) + 1 / alpha)
  const p = 10 ** state.logP

  const series = useMemo((): SeriesSpec[] => {
    const ps = LOG_P.map((x) => 10 ** x)
    return [
      { name: 'calibrated, q = p', type: 'line', x: LOG_P, y: ps, muted: true, dashed: true },
      { name: 'BCE (t = 0)', type: 'line', x: LOG_P, y: ps.map((v) => predicted(v, alpha, 1)), slot: 0 },
      { name: 'gBCE', type: 'line', x: LOG_P, y: ps.map((v) => predicted(v, alpha, beta)), slot: 1 },
    ]
  }, [alpha, beta])

  const markers = useMemo(
    (): SeriesSpec => ({
      name: 'at p',
      type: 'scatter',
      x: [state.logP, state.logP],
      y: [predicted(10 ** state.logP, alpha, 1), predicted(10 ** state.logP, alpha, beta)],
      emphasis: true,
    }),
    [state.logP, alpha, beta],
  )

  const sums = useMemo(() => [zipfSum(n, alpha, 1), zipfSum(n, alpha, beta)], [n, alpha, beta])
  const logAlpha = Math.log10(alpha)
  const segments = useMemo(
    () =>
      logAlpha > X_MIN ? [{ from: [logAlpha, 0] as [number, number], to: [logAlpha, 1] as [number, number] }] : [],
    [logAlpha],
  )
  const allSeries = useMemo(() => [...series, markers] as const, [series, markers])

  const xAxis = useAxis({ label: 'log₁₀ true probability p', range: X_RANGE })
  const yAxis = useAxis({ label: 'predicted probability q', range: Y_RANGE })
  return (
    <Figure
      title="Overconfidence from negative sampling"
      state={state}
      caption="Each curve is the probability q that a model converges to for an item, against the item's true next-item probability p (log scale), when trained with binary cross-entropy on one positive and k negatives drawn uniformly from the other items. The sampling rate is α = k / (|I| − 1), marked by the thin vertical line. Plain BCE gives q = p / (p + α(1 − p)), which is close to 1 for every item with p well above α. gBCE raises the positive's sigmoid to the power β = 1 − t(1 − α); at t = 1 (β = α) the model is calibrated. Drag the vertical handle to move p. The Zipf readouts sum q over a catalogue whose true probabilities fall as 1/rank; a calibrated model sums to 1."

      readouts={
        <>
          <Readout label="α" value={formatNumber(alpha)} />
          <Readout label="β" value={formatNumber(beta)} />
          <Readout label="q (BCE)" value={formatNumber(predicted(p, alpha, 1))} />
          <Readout label="q (gBCE)" value={formatNumber(predicted(p, alpha, beta))} />
          <Readout label="Zipf Σq (BCE)" value={formatNumber(sums[0])} />
          <Readout label="Zipf Σq (gBCE)" value={formatNumber(sums[1])} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        {seriesLayers(allSeries)}
        <Segments segments={segments} />
        <Handle {...state.handle('logP', { label: 'p' })} />
      </Plot>
    </Figure>
  )
}
