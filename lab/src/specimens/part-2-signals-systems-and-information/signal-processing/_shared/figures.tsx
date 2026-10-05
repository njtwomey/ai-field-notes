import { useMemo, useState } from 'react'
import { butter, cheby1, cheby2, filtfilt, firwin, freqz, lfilter } from 'aifn-compute/signal/filters'
import { chirp, tones, uniformTimes } from 'aifn-methods/data/signals'
import { cwt } from 'aifn-compute/signal/wavelets'
import { decibels } from 'aifn-compute/foundation/fourier'
import { siftSteps } from 'aifn-compute/signal/decompositions'
import { spectrogram, welch } from 'aifn-compute/signal/spectral'
import { type WindowSpec } from 'aifn-compute/signal/windows'
import { normals, stream } from 'aifn-compute/foundation/random'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { spectrumDecibels } from 'aifn-compute/signal'
import { trace } from 'aifn-compute/foundation/trace'
import { Player } from 'aifn-render/controls'
import { Figure } from 'aifn-render/layout'
import { choice, row, slider, toggle, useFigureState } from 'aifn-render/state'
import { Curve, Handle, Plot, Plots, Raster, Readout, useAxis } from 'aifn-render/viz'

const rowsOf = (t: Tensor) => {
  const [r, c] = t.shape
  const v = toFlat(t)
  return Array.from({ length: r }, (_, i) => v.slice(i * c, (i + 1) * c))
}

// ---------------------------------------------------------------------------------------------------------------------
// 1. Spectrogram of a chirp.

const FS = 1000
const N = 4000

export function ChirpSpectrogramSpecimen() {
  const state = useFigureState({
    signal: row('1 · signal', { f1: slider(50, 490, 400, { label: 'end frequency f₁ (Hz)', step: 10 }) }),
    analysis: row('2 · analysis', {
      nperseg: choice([32, 64, 128, 256, 512], 128, { label: 'segment length' }),
      win: choice(['hann', 'boxcar', 'blackman'], 'hann', { label: 'window' }),
    }),
    reveal: row('3 · reveal', { sweep: toggle(false, 'true sweep f(t)') }),
  })
  const { f1 } = state.signal
  const { nperseg, win } = state.analysis
  const x = useMemo(() => {
    const t = uniformTimes(N, FS)
    const c = toFlat(chirp(t, 10, N / FS, f1))
    const noise = toFlat(normals(stream('chirp-noise'), N, 0, 0.3))
    return c.map((v, i) => v + noise[i])
  }, [f1])
  const sg = useMemo(
    () => spectrogram(x, { fs: FS, nperseg, noverlap: Math.floor((nperseg * 3) / 4), window: win as WindowSpec }),
    [x, nperseg, win],
  )
  const view = useMemo(() => {
    const db = rowsOf(decibels(sg.values))
    // Display only: clip 60 dB below the peak so the noise floor does not take the colour scale.
    const floor = Math.max(...db.flat()) - 60
    return { t: toFlat(sg.t), f: toFlat(sg.f), z: db.map((r) => r.map((v) => Math.max(v, floor))) }
  }, [sg])
  const psd = useMemo(() => {
    const w = welch(x, { fs: FS, nperseg: 256 })
    return { f: toFlat(w.f), db: toFlat(decibels(w.values)) }
  }, [x])
  const df = FS / nperseg
  const dt = nperseg / 4 / FS
  const time = useAxis({ label: 'time (s)' })
  const freq = useAxis({ label: 'frequency (Hz)', range: [0, FS / 2] })
  const power = useAxis({ label: 'PSD (dB/Hz)', hold: 'union' })
  return (
    <Figure
      title="Spectrogram of a linear chirp"
      purpose="A short-time spectrum trades time resolution for frequency resolution: long segments give narrow frequency bins but smear the sweep in time; the Welch average (right) loses the time axis altogether."
      defaultSize="L"
      state={state}
      readouts={{
        resolution: (
          <>
            <Readout label="frequency bin" value={`${df.toFixed(2)} Hz`} />
            <Readout label="hop" value={`${(dt * 1000).toFixed(0)} ms`} />
            <Readout label="segments" value={sg.t.shape[0]} />
          </>
        ),
      }}
      caption="aifn-compute/signal chirp (10 Hz → f₁ over 4 s at 1 kHz) plus white noise of sd 0.3; spectrogram with 75% overlap, in dB, clipped 60 dB below the peak for display. Try segment length 32 against 512: the sweep blurs along frequency or along time. The Welch PSD (right) shares the frequency axis."
    >
      <Plots cols={2} widths={[2.2, 1]}>
        <Plot x={time} y={freq}>
          <Raster x={view.t} y={view.f} z={view.z} valueLabel="power (dB)" />
          {state.reveal.sweep && <Curve name="true sweep" x={[0, N / FS]} y={[10, f1]} emphasis dashed />}
        </Plot>
        <Plot x={power} y={freq}>
          <Curve name="Welch PSD" x={psd.db} y={psd.f} slot={0} />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 2. Filter design and filtering.

const FAMILIES = [
  { value: 'butter', label: 'Butterworth' },
  { value: 'cheby1', label: 'Chebyshev I (1 dB ripple)' },
  { value: 'cheby2', label: 'Chebyshev II (40 dB stop)' },
  { value: 'fir', label: 'FIR, Hamming window' },
] as const

export function FilterSpecimen() {
  const state = useFigureState({
    design: row('1 · design', {
      family: choice(FAMILIES, 'butter', { label: 'family' }),
      order: slider(1, 10, 4, { label: 'order (FIR: taps / 8)', step: 1 }),
    }),
    cutoff: slider(5, 200, 40, { onChart: true, label: 'cutoff (Hz)' }),
    reveal: row('2 · reveal', { zero: toggle(true, 'filtfilt (zero phase)') }),
  })
  const { family, order } = state.design
  const cutoff = state.cutoff
  const fs = 500
  const design = useMemo(() => {
    // Every design is an LTI system; the IIR ones in transfer-function form so their coefficients can be counted.
    if (family === 'fir') return firwin(8 * order + 1, cutoff, { fs })
    if (family === 'butter') return butter(order, cutoff, { fs, output: 'tf' })
    if (family === 'cheby1') return cheby1(order, 1, cutoff, { fs, output: 'tf' })
    return cheby2(order, 40, cutoff, { fs, output: 'tf' })
  }, [family, order, cutoff])
  const response = useMemo(() => {
    // The response is a complex128 spectrum over f in Hz (fs comes from the design, dt = 1/fs); a response is an
    // amplitude, so its level is 20 log₁₀ |H|.
    const r = freqz(design, { n: 512 })
    return { f: toFlat(r.f), db: toFlat(spectrumDecibels(r)).map((v) => Math.max(v, -100)) }
  }, [design])
  const signal = useMemo(() => {
    const t = uniformTimes(500, fs)
    const clean = toFlat(tones(t, [{ frequency: 5 }]))
    const noisy = toFlat(tones(t, [{ frequency: 5 }, { frequency: 80, amplitude: 0.5 }]))
    const noise = toFlat(normals(stream('filter-noise'), 500, 0, 0.2))
    return { t: toFlat(t), clean, x: noisy.map((v, i) => v + noise[i]) }
  }, [])
  const causal = useMemo(() => toFlat(lfilter(design, signal.x).y as Tensor), [design, signal])
  const zero = useMemo(() => toFlat(filtfilt(design, signal.x) as Tensor), [design, signal])
  const f = useAxis({ label: 'frequency (Hz)', range: [0, fs / 2] })
  const gain = useAxis({ label: 'gain (dB)', range: [-100, 5] })
  const t = useAxis({ label: 'time (s)' })
  const y = useAxis({ label: 'signal', hold: 'initial' })
  return (
    <Figure
      title="Low-pass filters and zero-phase filtering"
      purpose="Filtering once delays the output (lfilter's curve lags the 5 Hz tone); filtering forwards and backwards cancels the delay at the cost of squaring the magnitude response."
      defaultSize="L"
      state={state}
      readouts={{
        design: (
          <>
            <Readout label="cutoff" value={`${cutoff.toFixed(1)} Hz`} />
            <Readout
              label="coefficients b, a"
              value={design.repr.form === 'tf' ? `${design.repr.b.shape[0]}, ${design.repr.a.shape[0]}` : '—'}
            />
          </>
        ),
      }}
      caption="aifn-compute/signal butter, cheby1, cheby2 and firwin at fs = 500 Hz; the input is a 5 Hz tone plus a 0.5-amplitude 80 Hz tone and white noise. Drag the cutoff line on the response; raise the order and watch the causal output's lag grow while the zero-phase one stays on the tone."
    >
      <Plots rows={2} heights={[1, 1.3]}>
        <Plot x={f} y={gain}>
          <Curve name="|H(f)| (dB)" x={response.f} y={response.db} slot={0} />
          <Handle {...state.handle('cutoff', { label: 'cutoff' })} />
        </Plot>
        <Plot x={t} y={y}>
          <Curve name="input" x={signal.t} y={signal.x} thin muted />
          <Curve name="lfilter (causal)" x={signal.t} y={causal} slot={1} />
          {state.reveal.zero && <Curve name="filtfilt (zero phase)" x={signal.t} y={zero} slot={2} />}
          <Curve name="5 Hz tone" x={signal.t} y={signal.clean} dashed emphasis />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 3. Wavelet scalogram.

const SIGNALS = [
  { value: 'switch', label: '5 Hz then 25 Hz' },
  { value: 'chirp', label: 'chirp 2 → 40 Hz' },
  { value: 'impulse', label: '8 Hz tone + impulse' },
] as const
const FREQS = Array.from({ length: 60 }, (_, i) => 1 + i)

export function ScalogramSpecimen() {
  const state = useFigureState({
    signal: row('1 · signal', { which: choice(SIGNALS, 'switch', { label: 'signal' }) }),
    wavelet: row('2 · wavelet', { omega0: slider(3, 16, 6, { label: 'Morlet ω₀ (centre frequency)' }) }),
  })
  const { which } = state.signal
  const { omega0 } = state.wavelet
  const fs = 200
  const n = 800
  const x = useMemo(() => {
    const t = toFlat(uniformTimes(n, fs))
    if (which === 'chirp') return toFlat(chirp(t, 2, n / fs, 40))
    if (which === 'impulse') return t.map((_, i) => (i === 400 ? 10 : 0) + Math.sin(2 * Math.PI * 8 * t[i]))
    return t.map((s) => (s < 2 ? Math.sin(2 * Math.PI * 5 * s) : Math.sin(2 * Math.PI * 25 * s)))
  }, [which])
  const c = useMemo(() => cwt(x, FREQS, { fs, omega0 }), [x, omega0])
  const t = useMemo(() => toFlat(c.t), [c])
  const z = useMemo(() => rowsOf(c.magnitude), [c])
  const time = useAxis({ label: 'time (s)' })
  const xa = useAxis({ label: 'x', key: which, hold: 'initial' })
  const freq = useAxis({ label: 'frequency (Hz)' })
  return (
    <Figure
      title="Morlet wavelet scalogram"
      purpose="A wavelet transform uses short windows at high frequencies and long ones at low frequencies, so it locates the frequency switch and the impulse in time while resolving low frequencies finely; ω₀ trades time for frequency resolution."
      defaultSize="L"
      state={state}
      caption="aifn-compute/signal cwt at frequencies 1, 2, …, 60 Hz, fs = 200 Hz. Pick the impulse: its cone narrows towards high frequencies. Raise ω₀: the bands sharpen in frequency and spread in time."
    >
      <Plots rows={2} heights={[1, 2.5]} hoverGroup>
        <Plot x={time} y={xa}>
          <Curve name="x(t)" x={t} y={x} slot={0} />
        </Plot>
        <Plot x={time} y={freq}>
          <Raster x={t} y={FREQS} z={z} valueLabel="|W|" />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 4. EMD sifting, step by step.

export function SiftSpecimen() {
  const x = useMemo(() => {
    const t = toFlat(uniformTimes(512, 512))
    return t.map((s) => Math.sin(2 * Math.PI * 30 * s) + 1.2 * Math.sin(2 * Math.PI * 4 * s) + s)
  }, [])
  const run = useMemo(() => trace(siftSteps(x, { rule: { kind: 'fixed', sifts: 12 } }), undefined, 12), [x])
  const [k, setK] = useState(0)
  const at = Math.min(k, run.steps.length - 1)
  const s = run.steps[at]
  const prev = run.steps[Math.max(0, at - 1)]
  const idx = useMemo(() => x.map((_, i) => i), [x])
  const flat = useMemo(
    () => ({
      h: toFlat(s.h),
      before: toFlat(prev.h),
      upper: toFlat(s.upper),
      lower: toFlat(s.lower),
      mean: toFlat(s.mean),
    }),
    [s, prev],
  )
  const xs = useAxis({ label: 'sample' })
  const ys = useAxis({ label: 'value', hold: 'union' })
  return (
    <Figure
      title="Sifting one intrinsic mode function"
      purpose="Each sift fits cubic splines through the maxima and minima and subtracts their mean; after a few sifts the mean is near zero and h is the fastest oscillation, the first IMF."
      defaultSize="L"
      controls={<Player label="sift" value={at} onChange={setK} count={run.steps.length} />}
      readouts={{
        'this sift': (
          <>
            <Readout label="maxima" value={s.maxima.shape[0]} />
            <Readout label="minima" value={s.minima.shape[0]} />
            <Readout
              label="max |mean|"
              value={s.mean.shape[0] ? Math.max(...flat.mean.map(Math.abs)).toFixed(4) : '—'}
            />
          </>
        ),
      }}
      caption="aifn-compute/signal siftSteps on sin(2π·30t) + 1.2 sin(2π·4t) + t, twelve fixed sifts. Sift 0 is the signal; play to watch the envelopes' mean (ink) flatten towards zero as the slow tone and the trend are removed."
    >
      <Plot x={xs} y={ys}>
        {at === 0 ? (
          <Curve name="h" x={idx} y={flat.h} slot={0} />
        ) : (
          <>
            <Curve name="h before" x={idx} y={flat.before} slot={0} />
            <Curve name="upper envelope" x={idx} y={flat.upper} slot={1} dashed />
            <Curve name="lower envelope" x={idx} y={flat.lower} slot={2} dashed />
            <Curve name="mean" x={idx} y={flat.mean} emphasis />
          </>
        )}
      </Plot>
    </Figure>
  )
}
