import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamSlider,
  ParamSwitch,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from 'aifn-render'
import { rng } from '@/lib/math'

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
  const bits = useParam(3, { min: 1, max: 16, step: 1 })
  const [dither, setDither] = useState(false)

  const r = useMemo(() => {
    const { step, q } = quantiser(bits.value)
    const g = rng(11)
    // Triangular (TPDF) dither: the sum of two independent uniforms on [−Δ/2, Δ/2].
    const noise = () => (g.uniform() - 0.5) * step + (g.uniform() - 0.5) * step
    const out = X.map((v) => q(dither ? v + noise() : v))
    const err = out.map((v, n) => v - X[n])
    const power = (a: number[]) => a.reduce((s, v) => s + v * v, 0) / a.length
    const snr = 10 * Math.log10(power(X) / power(err))
    return { out, err, step, snr }
  }, [bits.value, dither])

  const theory = 6.02 * bits.value + 1.76 - (dither ? 4.77 : 0)
  const signal: XYSeries[] = [
    { name: 'x[n]', type: 'line', x: INDEX, y: X.slice(0, SHOWN), slot: 0 },
    { name: 'quantised', type: 'line', x: INDEX, y: r.out.slice(0, SHOWN), slot: 1 },
  ]
  const error: XYSeries[] = [{ name: 'error', type: 'line', x: INDEX, y: r.err.slice(0, SHOWN), slot: 2 }]
  const bound = Math.max(r.step, 1e-3)

  return (
    <Interactive
      title="Quantising a sine"
      caption="A full-scale sine is rounded to one of 2^b levels. The error stays within ±Δ/2 and, for enough bits, behaves like white noise of power Δ²/12, which gives an SNR of about 6.02b + 1.76 dB. At few bits the error is visibly correlated with the signal. Dither adds a little random noise before quantising: it decorrelates the error from the signal at the cost of about 4.8 dB."
      controls={
        <>
          <ParamSlider label="bits b" param={bits} format={(v) => String(v)} withArrows />
          <ParamSwitch label="triangular dither" checked={dither} onChange={setDither} />
        </>
      }
      readout={
        <>
          <Readout label="levels" value={2 ** bits.value} />
          <Readout label="step Δ" value={formatNumber(r.step)} />
          <Readout label="measured SNR" value={`${formatNumber(r.snr)} dB`} />
          <Readout label="theory" value={`${formatNumber(theory)} dB`} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart series={signal} xLabel="n" yLabel="amplitude" yRange={[-1.1, 1.1]} height={260} />
        <XYChart series={error} xLabel="n" yLabel="error" yRange={[-bound, bound]} height={260} />
      </div>
    </Interactive>
  )
}
