import { useMemo } from 'react'
import { Curve, Figure, formatNumber, int, Plot, Readout, setting, useAxis, useFigureState } from 'aifn-render'
import { stream, uniform } from 'aifn/foundation/random'

const N = 8192
const SHOWN = 160
// A frequency that is not a simple fraction of f_s, so the error does not repeat in a short cycle.
const F0 = 0.01237
const X = Array.from({ length: N }, (_, n) => Math.sin(2 * Math.PI * F0 * n))
const INDEX = Array.from({ length: SHOWN }, (_, n) => n)

/** Mid-rise uniform quantiser with 2^b levels on [−1, 1]: step Δ = 2/2^b, outputs at odd multiples of Δ/2. */
const quantiser = (bits: number) => {
  const step = 2 / 2 ** bits
  return {
    step,
    q: (v: number) => Math.min(1 - step / 2, Math.max(-1 + step / 2, step * (Math.floor(v / step) + 0.5))),
  }
}

/** A full-scale sine quantised to b bits: the error, its size, and the measured SNR against 6.02b + 1.76 dB. */
export function QuantiseSine() {
  const state = useFigureState({
    bits: int(3, { min: 1, max: 16, step: 1, label: 'bits b', format: (v) => String(v) }),
    dither: setting(false, 'triangular dither'),
  })

  const r = useMemo(() => {
    const { step, q } = quantiser(state.bits)
    const g = stream(11)
    // Triangular (TPDF) dither: the sum of two independent uniforms on [−Δ/2, Δ/2].
    const noise = () => (uniform(g) - 0.5) * step + (uniform(g) - 0.5) * step
    const out = X.map((v) => q(state.dither ? v + noise() : v))
    const err = out.map((v, n) => v - X[n])
    const power = (a: number[]) => a.reduce((s, v) => s + v * v, 0) / a.length
    const snr = 10 * Math.log10(power(X) / power(err))
    return { out, err, step, snr }
  }, [state.bits, state.dither])

  const theory = 6.02 * state.bits + 1.76 - (state.dither ? 4.77 : 0)
  const signal = [
    { name: 'x[n]', x: INDEX, y: X.slice(0, SHOWN), slot: 0 },
    { name: 'quantised', x: INDEX, y: r.out.slice(0, SHOWN), slot: 1 },
  ] as const
  const error = [{ name: 'error', x: INDEX, y: r.err.slice(0, SHOWN), slot: 2 }] as const
  const bound = Math.max(r.step, 1e-3)

  const xAxis = useAxis({ label: 'n', hold: 'union' })
  const yAxis = useAxis({ label: 'amplitude', range: [-1.1, 1.1] })
  const xAxis2 = useAxis({ label: 'n', hold: 'union' })
  const yAxis2 = useAxis({ label: 'error', range: [-bound, bound] })
  return (
    <Figure
      title="Quantising a sine"
      state={state}
      caption="A full-scale sine is rounded to one of 2^b levels. The error stays within ±Δ/2 and, for enough bits, behaves like white noise of power Δ²/12, which gives an SNR of about 6.02b + 1.76 dB. At few bits the error is visibly correlated with the signal. Dither adds a little random noise before quantising: it decorrelates the error from the signal at the cost of about 4.8 dB."

      readouts={
        <>
          <Readout label="levels" value={2 ** state.bits} />
          <Readout label="step Δ" value={formatNumber(r.step)} />
          <Readout label="measured SNR" value={`${formatNumber(r.snr)} dB`} />
          <Readout label="theory" value={`${formatNumber(theory)} dB`} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={260}>
          <Curve {...signal[0]} />
          <Curve {...signal[1]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={260}>
          <Curve {...error[0]} />
        </Plot>
      </div>
    </Figure>
  )
}
