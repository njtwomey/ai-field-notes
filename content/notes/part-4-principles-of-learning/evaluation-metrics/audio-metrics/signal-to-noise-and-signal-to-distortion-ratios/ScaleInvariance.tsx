import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, Handle, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { normal, stream } from 'aifn-compute/foundation/random'

const N = 400
const T = Array.from({ length: N }, (_, n) => n)
// The reference is two tones; the interference a third; the noise is fixed per page load.
const REF = T.map((n) => Math.sin((2 * Math.PI * 5 * n) / N) + 0.5 * Math.sin((2 * Math.PI * 13 * n) / N))
const INTERFERENCE = T.map((n) => Math.sin((2 * Math.PI * 29 * n) / N + 0.7))
const NOISE = (() => {
  const g = stream(11)
  return T.map(() => normal(g))
})()
const GAINS = toFlat(linspace(0.05, 2, 80))

const dot = (a: number[], b: number[]) => a.reduce((s, v, i) => s + v * b[i], 0)
const db = (num: number, den: number) => 10 * Math.log10(num / Math.max(den, 1e-300))

function scores(g: number, a: number, b: number) {
  // The gain scales the whole estimate: a louder or quieter copy of the same separated signal.
  const est = T.map((n) => g * (REF[n] + a * INTERFERENCE[n] + b * NOISE[n]))
  const ss = dot(REF, REF)
  const snr = db(
    ss,
    est.reduce((s, v, n) => s + (REF[n] - v) ** 2, 0),
  )
  // SI-SDR: project the estimate on the reference; everything else is error.
  const alpha = dot(est, REF) / ss
  const target = REF.map((v) => alpha * v)
  const err = est.map((v, n) => v - target[n])
  const sisdr = db(dot(target, target), dot(err, err))
  // BSS-eval-style split (gain-only distortion): project on span{reference, interference}; the rest is artefact.
  const si = dot(REF, INTERFERENCE)
  const ii = dot(INTERFERENCE, INTERFERENCE)
  const es = dot(est, REF)
  const ei = dot(est, INTERFERENCE)
  const det = ss * ii - si * si
  const cs = (es * ii - ei * si) / det
  const ci = (ei * ss - es * si) / det
  const proj = T.map((n) => cs * REF[n] + ci * INTERFERENCE[n])
  const interf = proj.map((v, n) => v - target[n])
  const artif = est.map((v, n) => v - proj[n])
  const sir = db(dot(target, target), dot(interf, interf))
  const sar = db(dot(proj, proj), dot(artif, artif))
  return { est, alpha, target, snr, sisdr, sir, sar }
}

/** SNR changes with the estimate's gain; SI-SDR does not, because it first rescales the reference to fit. */
export function ScaleInvariance() {
  const state = useFigureState({
    gain: float(1, { min: 0.05, max: 2, step: 0.01, label: 'gain g' }),
    interference: float(0.2, { min: 0, max: 1, step: 0.01, label: 'interference level a' }),
    noise: float(0.1, { min: 0, max: 1, step: 0.01, label: 'noise level b' }),
  })

  const s = useMemo(
    () => scores(state.gain, state.interference, state.noise),
    [state.gain, state.interference, state.noise],
  )
  const curves = useMemo(() => {
    const rows = GAINS.map((g) => scores(g, state.interference, state.noise))
    return {
      snr: rows.map((r) => r.snr),
      sisdr: rows.map((r) => r.sisdr),
    }
  }, [state.interference, state.noise])

  const waves = [
    { name: 'reference s', x: T.slice(0, 200), y: REF.slice(0, 200), slot: 0 },
    { name: 'estimate ŝ', x: T.slice(0, 200), y: s.est.slice(0, 200), slot: 1 },
    { name: 'scaled target αs', x: T.slice(0, 200), y: s.target.slice(0, 200), slot: 2, dashed: true },
  ] as const
  const vsGain = [
    { name: 'SNR', x: GAINS, y: curves.snr, slot: 0 },
    { name: 'SI-SDR', x: GAINS, y: curves.sisdr, slot: 1 },
  ] as const

  const xAxis = useAxis({ label: 'sample', hold: 'union' })
  const yAxis = useAxis({ label: 'amplitude', hold: 'union' })
  const xAxis2 = useAxis({ label: 'gain g', hold: 'union' })
  const yAxis2 = useAxis({ label: 'dB', hold: 'union' })
  return (
    <Figure
      title="Scale invariance"
      state={state}
      caption="The estimate is g·(s + a·interference + b·noise): a separated signal played back at gain g. SNR compares it with the reference as is, so a wrong gain counts as error and SNR peaks near g = 1. SI-SDR first finds the scale α that best fits the reference to the estimate and measures the error against αs, so it does not change with g at all. Drag the gain line, or raise the interference and noise to see both fall. SIR and SAR split the SI-SDR error into the interference part and the rest."

      readouts={
        <>
          <Readout label="SNR (dB)" value={formatNumber(s.snr)} />
          <Readout label="SI-SDR (dB)" value={formatNumber(s.sisdr)} />
          <Readout label="α" value={formatNumber(s.alpha)} />
          <Readout label="SIR (dB)" value={formatNumber(s.sir)} />
          <Readout label="SAR (dB)" value={formatNumber(s.sar)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={280}>
          <Curve {...waves[0]} />
          <Curve {...waves[1]} />
          <Curve {...waves[2]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={280}>
          <Curve {...vsGain[0]} />
          <Curve {...vsGain[1]} />
          <Handle {...state.handle('gain', { label: 'gain' })} />
        </Plot>
      </div>
    </Figure>
  )
}
