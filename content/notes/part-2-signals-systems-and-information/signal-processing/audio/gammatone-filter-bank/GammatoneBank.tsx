import { useMemo } from 'react'
import {
  Curve,
  Figure,
  formatNumber,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { magnitudeSpectrum } from '../_shared/audio'

/** Decibels, 20 log₁₀ of a magnitude, floored at `floor` dB. */
const db = (m: number, floor: number) => Math.max(floor, 20 * Math.log10(Math.max(m, 1e-300)))

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
  const state = useFigureState({
    count: int(16, { min: 4, max: 48, step: 1, label: 'number of filters' }),
    probe: int(1000, { min: 100, max: 6000, step: 50, label: 'impulse-response f_c (Hz)' }),
  })

  const r = useMemo(() => {
    const [lo, hi] = [erbNumber(100), erbNumber(6000)]
    const centres = Array.from({ length: state.count }, (_, i) =>
      fromErbNumber(lo + ((hi - lo) * i) / Math.max(1, state.count - 1)),
    )
    const freqs = Array.from({ length: NFFT / 2 + 1 }, (_, k) => (k * FS) / NFFT)
    const responses = centres.map((fc) => {
      const mag = magnitudeSpectrum(gammatone(fc), NFFT)
      const peak = Math.max(...mag)
      return Array.from(mag, (m) => db(m / peak, -60))
    })
    const ir = gammatone(state.probe)
    const irPeak = Math.max(...ir.map(Math.abs))
    return { centres, freqs, responses, ir: ir.map((v) => v / irPeak) }
  }, [state.count, state.probe])

  const bank: SeriesSpec[] = r.responses.map((y, i) => ({
    name: i === 0 || i === r.responses.length - 1 ? `f_c = ${Math.round(r.centres[i])} Hz` : 'other filters',
    type: 'line',
    x: r.freqs,
    y,
    ...(i === 0 ? { slot: 0 } : i === r.responses.length - 1 ? { slot: 1 } : { muted: true }),
  }))
  const t = r.ir.map((_, i) => (1000 * i) / FS)
  const impulse = [{ name: `impulse response, f_c = ${state.probe} Hz`, x: t, y: r.ir, slot: 2 }] as const

  const xAxis = useAxis({ label: 'frequency (Hz)', range: [0, 7000] })
  const yAxis = useAxis({ label: 'gain (dB)', range: [-60, 3] })
  const xAxis2 = useAxis({ label: 'time (ms)', range: [0, 30] })
  const yAxis2 = useAxis({ label: 'normalised amplitude', hold: 'union' })
  return (
    <Figure
      title="A gammatone filter bank"
      state={state}
      caption="Fourth-order gammatone filters with bandwidth parameter b = 1.019 ERB(f_c), centred at frequencies equally spaced in ERB number between 100 Hz and 6 kHz, each normalised to 0 dB at its peak. The filters are narrow and densely packed at low frequencies, wide at high ones, like the cochlea's. Right: one filter's impulse response, a tone at f_c inside a gamma-distribution envelope that rises and then decays; low-frequency filters ring for longer."

      readouts={
        <>
          <Readout label="ERB at probe" value={`${formatNumber(erb(state.probe))} Hz`} />
          <Readout label="b at probe" value={`${formatNumber(1.019 * erb(state.probe))} Hz`} />
          <Readout
            label="envelope peak at"
            value={`${formatNumber((1000 * (ORDER - 1)) / (2 * Math.PI * 1.019 * erb(state.probe)))} ms`}
          />
        </>
      }
    >
      <div className="grid gap-4 lg:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={300}>
          {seriesLayers(bank)}
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={300}>
          <Curve {...impulse[0]} />
        </Plot>
      </div>
    </Figure>
  )
}
