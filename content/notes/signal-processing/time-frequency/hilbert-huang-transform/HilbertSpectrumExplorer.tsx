import { useMemo, useState } from 'react'
import { Heatmap, Interactive, ParamChoice, Readout, XYChart, formatNumber } from 'aifn-render'
import { magnitudeSpectrum } from '@/lib/dsp'
import { emd } from '../_shared/emd'
import { hilbertSpectrum } from '../_shared/hht'
import { chirp } from '../_shared/tf'

const N = 512
const F_MAX = 0.35
const TIME_BINS = 128
const FREQ_BINS = 70
const TIME = Array.from({ length: N }, (_, n) => n)

type Choice = 'chirp' | 'amfm' | 'wave' | 'close'

/** Test signals, frequencies in cycles per sample. */
function makeSignal(choice: Choice): Float64Array {
  const tau = 2 * Math.PI
  if (choice === 'chirp') {
    const c = chirp(N, 1, 0.02, 0.1)
    return Float64Array.from(TIME, (n) => c[n] + 0.6 * Math.cos(tau * 0.25 * n))
  }
  if (choice === 'amfm') {
    // Amplitude 1 ± 0.5 over 256 samples; frequency 0.12 ± 0.04 over 128 samples.
    return Float64Array.from(
      TIME,
      (n) =>
        (1 + 0.5 * Math.cos((tau * n) / 256)) *
        Math.cos(tau * 0.12 * n + (0.04 / (1 / 128)) * Math.sin((tau * n) / 128)),
    )
  }
  if (choice === 'wave') {
    // A Stokes-like nonlinear wave: sharp crests, flat troughs, one oscillation per period of 32 samples.
    const w = tau / 32
    return Float64Array.from(TIME, (n) => Math.cos(w * n + 0.5 * Math.sin(w * n)))
  }
  return Float64Array.from(TIME, (n) => Math.cos(tau * 0.1 * n) + 0.8 * Math.cos(tau * 0.12 * n))
}

/**
 * The Hilbert spectrum of a signal's IMFs as a heatmap, beside its marginal spectrum and the Fourier amplitude
 * spectrum. Components with drifting frequency stay single lines; a nonlinear wave shows intra-wave modulation.
 */
export function HilbertSpectrumExplorer() {
  const [choice, setChoice] = useState<Choice>('chirp')

  const r = useMemo(() => {
    const x = makeSignal(choice)
    const d = emd(x)
    const hs = hilbertSpectrum(d.imfs, TIME_BINS, FREQ_BINS, F_MAX)
    const peak = Math.max(...hs.z.map((row) => Math.max(...row)))
    // Fourier amplitude spectrum on the same frequency axis, scaled to the marginal spectrum's peak for comparison.
    const mag = magnitudeSpectrum(x, 4096)
    const fourier = hs.freqs.map((f) => mag[Math.round(f * 4096)])
    const mPeak = Math.max(...hs.marginal)
    const fPeak = Math.max(...fourier)
    return {
      x,
      d,
      hs,
      peak,
      marginal: hs.marginal.map((v) => v / mPeak),
      fourier: fourier.map((v) => v / fPeak),
    }
  }, [choice])

  return (
    <Interactive
      title="Hilbert spectrum and marginal spectrum"
      caption="Left: the Hilbert spectrum, each IMF's instantaneous amplitude placed at its instantaneous frequency at every sample, summed into cells of 4 samples by 0.005 cycles per sample. Right: the marginal spectrum (the Hilbert spectrum summed over time) against the Fourier amplitude spectrum, both scaled to peak 1. A chirp is a line that climbs; the Fourier spectrum spreads it into a plateau. The nonlinear wave is one IMF whose frequency oscillates within every period, where the Fourier spectrum shows harmonics at 2 and 3 times the fundamental. Two tones only 20% apart are not separated, and the instantaneous frequency of their sum swings at the beat rate."
      controls={
        <ParamChoice
          label="signal"
          value={choice}
          onChange={setChoice}
          options={[
            { value: 'chirp', label: 'chirp + tone' },
            { value: 'amfm', label: 'AM-FM tone' },
            { value: 'wave', label: 'nonlinear wave' },
            { value: 'close', label: 'close tones' },
          ]}
        />
      }
      readout={
        <>
          <Readout label="IMFs" value={r.d.imfs.length} />
          <Readout label="peak cell amplitude" value={formatNumber(r.peak)} />
        </>
      }
    >
      <div className="min-w-0 space-y-1">
        <div className="text-center text-xs text-muted-foreground">signal</div>
        <XYChart series={[{ name: 'x', type: 'line', x: TIME, y: Array.from(r.x) }]} xRange={[0, N - 1]} height={150} />
      </div>
      <div className="grid gap-4 md:grid-cols-[2fr_1fr]">
        <div className="min-w-0 space-y-1">
          <div className="text-center text-xs text-muted-foreground">Hilbert spectrum H(t, f)</div>
          <Heatmap
            x={r.hs.times}
            y={r.hs.freqs}
            z={r.hs.z}
            range={[0, r.peak]}
            xLabel="time (samples)"
            yLabel="frequency (cycles/sample)"
            valueLabel="amplitude"
            height={320}
          />
        </div>
        <div className="min-w-0 space-y-1">
          <div className="text-center text-xs text-muted-foreground">marginal vs Fourier spectrum</div>
          <XYChart
            series={[
              { name: 'marginal h(f)', type: 'line', x: r.hs.freqs, y: r.marginal, slot: 0 },
              { name: 'Fourier |X(f)|', type: 'line', x: r.hs.freqs, y: r.fourier, slot: 1, dashed: true },
            ]}
            xRange={[0, F_MAX]}
            yRange={[0, 1.05]}
            xLabel="frequency (cycles/sample)"
            height={320}
          />
        </div>
      </div>
    </Interactive>
  )
}
