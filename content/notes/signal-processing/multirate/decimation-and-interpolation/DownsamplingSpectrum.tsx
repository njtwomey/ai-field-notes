import { useMemo, useState } from 'react'
import { Interactive, ParamSlider, ParamSwitch, Readout, XYChart, useParam, type XYSeries } from '@/components/viz'
import { convolve, db, magnitudeSpectrum, makeWindow } from '@/lib/dsp'

const N = 2048
const NFFT = 4096
// Ten tones at ω/π = 0.05, 0.15, …, 0.95 with amplitudes falling from 1 to 0.1, so each can be told apart after folding.
const TONES = Array.from({ length: 10 }, (_, j) => ({ w: (0.05 + 0.1 * j) * Math.PI, a: 1 - 0.1 * j }))
const TAPS = 121

const signal = (() => {
  const x = new Float64Array(N)
  for (let n = 0; n < N; n++) for (const { w, a } of TONES) x[n] += a * Math.cos(w * n + 0.3 * a)
  return x
})()

/** Blackman-windowed sinc lowpass with cutoff ω_c (rad/sample) and unit DC gain. */
function lowpass(cutoff: number): Float64Array {
  const w = makeWindow('blackman', TAPS)
  const h = new Float64Array(TAPS)
  const mid = (TAPS - 1) / 2
  for (let i = 0; i < TAPS; i++) {
    const m = i - mid
    h[i] = (m === 0 ? cutoff / Math.PI : Math.sin(cutoff * m) / (Math.PI * m)) * w[i]
  }
  const sum = h.reduce((a, b) => a + b, 0)
  return h.map((v) => v / sum)
}

/** One-sided spectrum in dB (Hann window), against ω/π of the sequence's own rate. */
function spectrumDb(x: ArrayLike<number>): { x: number[]; y: number[] } {
  const w = makeWindow('hann', x.length, true)
  const xw = Float64Array.from(x, (v, i) => v * w[i])
  const mag = magnitudeSpectrum(xw, NFFT)
  const scale = x.length / 4 // a unit-amplitude tone peaks at N/4 through a Hann window
  return { x: Array.from(mag, (_, k) => (2 * k) / NFFT), y: Array.from(mag, (m) => db(m / scale, -80)) }
}

/**
 * Downsampling by M folds the spectrum: every tone above π/M reappears at a new frequency after compression. An
 * anti-aliasing lowpass before the downsampler removes those tones instead of letting them fold.
 */
export function DownsamplingSpectrum() {
  const factor = useParam(3, { min: 1, max: 6, step: 1 })
  const [filtered, setFiltered] = useState(false)
  const M = factor.value

  const r = useMemo(() => {
    let x: ArrayLike<number> = signal
    if (filtered && M > 1) {
      const full = convolve(signal, lowpass(Math.PI / M))
      const mid = (TAPS - 1) / 2
      x = full.slice(mid, mid + N)
    }
    const y = Array.from({ length: Math.floor(N / M) }, (_, n) => x[n * M])
    // Where each tone lands after downsampling: Mω folded into [0, π].
    const landed = TONES.map(({ w }) => {
      const wm = (((M * w) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)
      return (wm > Math.PI ? 2 * Math.PI - wm : wm) / Math.PI
    })
    return { input: spectrumDb(x), output: spectrumDb(y), landed }
  }, [M, filtered])

  const aliased = TONES.filter(({ w }) => w > Math.PI / M).length
  const inputSeries: XYSeries[] = [
    { name: filtered ? 'after anti-aliasing filter' : 'input x[n]', type: 'line', ...r.input, slot: 0 },
  ]
  const outputSeries: XYSeries[] = [{ name: `y[n] = x[${M}n]`, type: 'line', ...r.output, slot: 1 }]

  return (
    <Interactive
      title="Downsampling folds the spectrum"
      caption="The input is ten tones at ω/π = 0.05, 0.15, …, 0.95, with amplitudes falling from 1 to 0.1. Keeping every M-th sample stretches the band [0, π/M] across the whole output band, and every tone above π/M folds back onto a new frequency: a tone at ω lands at Mω reduced into [0, π]. Switch on the anti-aliasing filter, a lowpass with cutoff π/M before the downsampler, and the high tones are removed instead of folded."
      controls={
        <>
          <ParamSlider label="downsampling factor M" param={factor} format={(v) => String(v)} withArrows />
          <ParamSwitch label="anti-aliasing lowpass before ↓M" checked={filtered} onChange={setFiltered} />
        </>
      }
      readout={
        <>
          <Readout label="output Nyquist, in input units" value={`π/${M}`} />
          <Readout label="tones above π/M" value={`${aliased} of ${TONES.length}`} />
          <Readout
            label="where the tones land (ω/π of the output)"
            value={r.landed.map((v) => v.toFixed(2)).join(', ')}
          />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <div className="min-w-0 space-y-1">
          <div className="text-center text-xs text-muted-foreground">before downsampling</div>
          <XYChart
            series={inputSeries}
            xLabel="ω/π (input rate)"
            yLabel="dB"
            xRange={[0, 1]}
            yRange={[-80, 5]}
            height={280}
          />
        </div>
        <div className="min-w-0 space-y-1">
          <div className="text-center text-xs text-muted-foreground">after downsampling by {M}</div>
          <XYChart
            series={outputSeries}
            xLabel="ω/π (output rate)"
            yLabel="dB"
            xRange={[0, 1]}
            yRange={[-80, 5]}
            height={280}
          />
        </div>
      </div>
    </Interactive>
  )
}
