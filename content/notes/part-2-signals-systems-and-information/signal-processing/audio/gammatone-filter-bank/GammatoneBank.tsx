import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from 'aifn-render'
import { db, magnitudeSpectrum } from '@/lib/dsp'

const FS = 16000
const ORDER = 4
const LENGTH = 1024 // 64 ms of impulse response
const NFFT = 4096

const erb = (f: number) => 24.7 * (1 + (4.37 * f) / 1000)
const erbNumber = (f: number) => 21.4 * Math.log10(1 + (4.37 * f) / 1000)
const fromErbNumber = (e: number) => ((10 ** (e / 21.4) - 1) * 1000) / 4.37

/** Impulse response t^{n−1} e^{−2π b t} cos(2π f_c t), with b = 1.019 ERB(f_c), normalised to unit peak gain later. */
function gammatone(fc: number): number[] {
  const b = 1.019 * erb(fc)
  return Array.from({ length: LENGTH }, (_, i) => {
    const t = i / FS
    return t ** (ORDER - 1) * Math.exp(-2 * Math.PI * b * t) * Math.cos(2 * Math.PI * fc * t)
  })
}

/**
 * A gammatone filter bank with centres equally spaced in ERB number, and one filter's impulse response. Responses are
 * computed from the sampled impulse responses and normalised to 0 dB peak.
 */
export function GammatoneBank() {
  const count = useParam(16, { min: 4, max: 48, step: 1 })
  const probe = useParam(1000, { min: 100, max: 6000, step: 50 })

  const r = useMemo(() => {
    const [lo, hi] = [erbNumber(100), erbNumber(6000)]
    const centres = Array.from({ length: count.value }, (_, i) =>
      fromErbNumber(lo + ((hi - lo) * i) / Math.max(1, count.value - 1)),
    )
    const freqs = Array.from({ length: NFFT / 2 + 1 }, (_, k) => (k * FS) / NFFT)
    const responses = centres.map((fc) => {
      const mag = magnitudeSpectrum(gammatone(fc), NFFT)
      const peak = Math.max(...mag)
      return Array.from(mag, (m) => db(m / peak, -60))
    })
    const ir = gammatone(probe.value)
    const irPeak = Math.max(...ir.map(Math.abs))
    return { centres, freqs, responses, ir: ir.map((v) => v / irPeak) }
  }, [count.value, probe.value])

  const bank: XYSeries[] = r.responses.map((y, i) => ({
    name: i === 0 || i === r.responses.length - 1 ? `f_c = ${Math.round(r.centres[i])} Hz` : 'other filters',
    type: 'line',
    x: r.freqs,
    y,
    ...(i === 0 ? { slot: 0 } : i === r.responses.length - 1 ? { slot: 1 } : { muted: true }),
  }))
  const t = r.ir.map((_, i) => (1000 * i) / FS)
  const impulse: XYSeries[] = [
    { name: `impulse response, f_c = ${probe.value} Hz`, type: 'line', x: t, y: r.ir, slot: 2 },
  ]

  return (
    <Interactive
      title="A gammatone filter bank"
      caption="Fourth-order gammatone filters with bandwidth parameter b = 1.019 ERB(f_c), centred at frequencies equally spaced in ERB number between 100 Hz and 6 kHz, each normalised to 0 dB at its peak. The filters are narrow and densely packed at low frequencies, wide at high ones, like the cochlea's. Right: one filter's impulse response, a tone at f_c inside a gamma-distribution envelope that rises and then decays; low-frequency filters ring for longer."
      controls={
        <>
          <ParamSlider label="number of filters" param={count} withArrows />
          <ParamSlider label="impulse-response f_c (Hz)" param={probe} />
        </>
      }
      readout={
        <>
          <Readout label="ERB at probe" value={`${formatNumber(erb(probe.value))} Hz`} />
          <Readout label="b at probe" value={`${formatNumber(1.019 * erb(probe.value))} Hz`} />
          <Readout
            label="envelope peak at"
            value={`${formatNumber((1000 * (ORDER - 1)) / (2 * Math.PI * 1.019 * erb(probe.value)))} ms`}
          />
        </>
      }
    >
      <div className="grid gap-4 lg:grid-cols-2">
        <XYChart
          series={bank}
          xLabel="frequency (Hz)"
          yLabel="gain (dB)"
          xRange={[0, 7000]}
          yRange={[-60, 3]}
          height={300}
        />
        <XYChart series={impulse} xLabel="time (ms)" yLabel="normalised amplitude" xRange={[0, 30]} height={300} />
      </div>
    </Interactive>
  )
}
