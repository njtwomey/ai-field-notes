import { useMemo, useState } from 'react'
import { butter, cheby1, cheby2, filtfilt, firwin, freqz, lfilter } from 'aifn/signal/filters'
import { chirp, tones, uniformTimes } from 'aifn-applied/data/signals'
import { complex } from 'aifn/systems'
import { cwt } from 'aifn/signal/wavelets'
import { decibels, magnitude } from 'aifn/foundation/fourier'
import { siftSteps } from 'aifn/signal/decompositions'
import { spectrogram, welch } from 'aifn/signal/spectral'
import { type WindowSpec } from 'aifn/signal/windows'
import { normals, stream } from 'aifn/foundation/random'
import { toFlat, type Tensor } from 'aifn/foundation/tensor'
import { trace } from 'aifn/foundation/trace'
import { Player, Select, Slider } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { Heatmap, Panel, Readout, Subplots, XYChart, type XYSeries } from '@lab/viz'

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
  const [f1, setF1] = useState(400)
  const [nperseg, setNperseg] = useState(128)
  const [win, setWin] = useState<'hann' | 'boxcar' | 'blackman'>('hann')
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
  const db = rowsOf(decibels(sg.values))
  const floor = Math.max(...db.flat()) - 60
  const z = db.map((row) => row.map((v) => Math.max(v, floor)))
  const psd = welch(x, { fs: FS, nperseg: 256 })
  const psdSeries: XYSeries[] = [
    { name: 'Welch PSD', type: 'line', x: toFlat(psd.f), y: toFlat(decibels(psd.values)), slot: 0 },
  ]
  const df = FS / nperseg
  const dt = nperseg / 4 / FS
  return (
    <Figure
      title="Spectrogram of a linear chirp"
      description="A short-time spectrum trades time resolution for frequency resolution: long segments give narrow frequency bins but smear the sweep in time; the Welch average (right) loses the time axis altogether."
      defaultSize="L"
      controls={
        <>
          <ControlRow label="1 · signal">
            <Slider label="end frequency f₁ (Hz)" value={f1} onChange={setF1} min={50} max={490} step={10} />
          </ControlRow>
          <ControlRow label="2 · analysis">
            <Select
              label="segment length"
              value={String(nperseg)}
              onChange={(v) => setNperseg(Number(v))}
              options={['32', '64', '128', '256', '512']}
            />
            <Select label="window" value={win} onChange={setWin} options={['hann', 'boxcar', 'blackman']} />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout label="frequency bin" value={`${df.toFixed(2)} Hz`} />
          <Readout label="hop" value={`${(dt * 1000).toFixed(0)} ms`} />
          <Readout label="segments" value={sg.t.shape[0]} />
        </>
      }
      caption="aifn/dsp chirp (10 Hz → f₁ over 4 s at 1 kHz) plus white noise of sd 0.3; spectrogram with 75% overlap, in dB, clipped 60 dB below the peak for display."
    >
      <Subplots cols={2} widthRatios={[2.2, 1]}>
        <Panel>
          <Heatmap
            x={toFlat(sg.t)}
            y={toFlat(sg.f)}
            z={z}
            xLabel="time (s)"
            yLabel="frequency (Hz)"
            valueLabel="power (dB)"
          />
        </Panel>
        <Panel>
          <XYChart series={psdSeries} xLabel="frequency (Hz)" yLabel="PSD (dB/Hz)" />
        </Panel>
      </Subplots>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 2. Filter design and filtering.

type Family = 'butter' | 'cheby1' | 'cheby2' | 'fir'

export function FilterSpecimen() {
  const [family, setFamily] = useState<Family>('butter')
  const [order, setOrder] = useState(4)
  const [cutoff, setCutoff] = useState(40)
  const fs = 500
  const design = useMemo(() => {
    const wn = cutoff
    // Every design is an LTI system; the IIR ones in transfer-function form so their coefficients can be counted.
    if (family === 'fir') return firwin(8 * order + 1, wn, { fs })
    if (family === 'butter') return butter(order, wn, { fs, output: 'tf' })
    if (family === 'cheby1') return cheby1(order, 1, wn, { fs, output: 'tf' })
    return cheby2(order, 40, wn, { fs, output: 'tf' })
  }, [family, order, cutoff])
  const response = useMemo(() => {
    // The response is a complex spectrum: [k, 2] (re, im) values over f in Hz (fs comes from the design, dt = 1/fs).
    const r = freqz(design, { n: 512 })
    const h = magnitude(complex.parts(r.values))
    return { f: toFlat(r.f), db: toFlat(decibels(h, { power: false })).map((v) => Math.max(v, -100)) }
  }, [design])
  const signal = useMemo(() => {
    const t = uniformTimes(500, fs)
    const clean = toFlat(tones(t, [{ frequency: 5 }]))
    const noisy = toFlat(tones(t, [{ frequency: 5 }, { frequency: 80, amplitude: 0.5 }]))
    const noise = toFlat(normals(stream('filter-noise'), 500, 0, 0.2))
    return { t: toFlat(t), clean, x: noisy.map((v, i) => v + noise[i]) }
  }, [])
  const causal = toFlat(lfilter(design, signal.x).y.data)
  const zero = toFlat(filtfilt(design, signal.x).data)
  const respSeries: XYSeries[] = [{ name: '|H(f)| (dB)', type: 'line', x: response.f, y: response.db, slot: 0 }]
  const sigSeries: XYSeries[] = [
    { name: 'input', type: 'line', x: signal.t, y: signal.x, thin: true, slot: 7 },
    { name: 'lfilter (causal)', type: 'line', x: signal.t, y: causal, slot: 1 },
    { name: 'filtfilt (zero phase)', type: 'line', x: signal.t, y: zero, slot: 2 },
    { name: '5 Hz tone', type: 'line', x: signal.t, y: signal.clean, dashed: true, emphasis: true },
  ]
  return (
    <Figure
      title="Low-pass filters and zero-phase filtering"
      description="Filtering once delays the output (lfilter's curve lags the 5 Hz tone); filtering forwards and backwards cancels the delay at the cost of squaring the magnitude response."
      defaultSize="L"
      controls={
        <>
          <ControlRow label="1 · design">
            <Select
              label="family"
              value={family}
              onChange={setFamily}
              options={[
                { value: 'butter', label: 'Butterworth' },
                { value: 'cheby1', label: 'Chebyshev I (1 dB ripple)' },
                { value: 'cheby2', label: 'Chebyshev II (40 dB stop)' },
                { value: 'fir', label: 'FIR, Hamming window' },
              ]}
            />
            <Slider
              label={family === 'fir' ? 'taps / 8' : 'order'}
              value={order}
              onChange={(v) => setOrder(Math.round(v))}
              min={1}
              max={10}
              step={1}
            />
            <Slider label="cutoff (Hz)" value={cutoff} onChange={setCutoff} min={5} max={200} />
          </ControlRow>
        </>
      }
      readouts={
        <Readout
          label="coefficients b, a"
          value={design.repr.form === 'tf' ? `${design.repr.b.shape[0]}, ${design.repr.a.shape[0]}` : '—'}
        />
      }
      caption="aifn/dsp butter, cheby1, cheby2 and firwin at fs = 500 Hz; the input is a 5 Hz tone plus a 0.5-amplitude 80 Hz tone and white noise. Drag the cutoff line on the response."
    >
      <Subplots rows={2} heightRatios={[1, 1.3]}>
        <Panel>
          <XYChart
            series={respSeries}
            xLabel="frequency (Hz)"
            yLabel="gain (dB)"
            yRange={[-100, 5]}
            handles={[
              { kind: 'x', at: cutoff, label: 'cutoff', onDrag: (v) => setCutoff(Math.min(200, Math.max(5, v))) },
            ]}
          />
        </Panel>
        <Panel>
          <XYChart series={sigSeries} xLabel="time (s)" yLabel="signal" />
        </Panel>
      </Subplots>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 3. Wavelet scalogram.

type SignalId = 'chirp' | 'switch' | 'impulse'

export function ScalogramSpecimen() {
  const [which, setWhich] = useState<SignalId>('switch')
  const [omega0, setOmega0] = useState(6)
  const fs = 200
  const n = 800
  const x = useMemo(() => {
    const t = toFlat(uniformTimes(n, fs))
    if (which === 'chirp') return toFlat(chirp(t, 2, n / fs, 40))
    if (which === 'impulse') return t.map((_, i) => (i === 400 ? 10 : 0) + Math.sin(2 * Math.PI * 8 * t[i]))
    return t.map((s) => (s < 2 ? Math.sin(2 * Math.PI * 5 * s) : Math.sin(2 * Math.PI * 25 * s)))
  }, [which])
  const freqs = useMemo(() => Array.from({ length: 60 }, (_, i) => 1 + i), [])
  const c = useMemo(() => cwt(x, freqs, { fs, omega0 }), [x, freqs, omega0])
  const t = toFlat(c.t)
  return (
    <Figure
      title="Morlet wavelet scalogram"
      description="A wavelet transform uses short windows at high frequencies and long ones at low frequencies, so it locates the frequency switch and the impulse in time while resolving low frequencies finely; ω₀ trades time for frequency resolution."
      defaultSize="L"
      controls={
        <ControlRow label="signal and wavelet">
          <Select
            label="signal"
            value={which}
            onChange={setWhich}
            options={[
              { value: 'switch', label: '5 Hz then 25 Hz' },
              { value: 'chirp', label: 'chirp 2 → 40 Hz' },
              { value: 'impulse', label: '8 Hz tone + impulse' },
            ]}
          />
          <Slider label="ω₀" value={omega0} onChange={setOmega0} min={3} max={16} />
        </ControlRow>
      }
      caption="aifn/dsp cwt at frequencies 1, 2, …, 60 Hz, fs = 200 Hz."
    >
      <Subplots rows={2} sharex heightRatios={[1, 2.5]}>
        <Panel>
          <XYChart series={[{ name: 'x(t)', type: 'line', x: t, y: x, slot: 0 }]} xLabel="time (s)" yLabel="x" />
        </Panel>
        <Panel>
          <Heatmap x={t} y={freqs} z={rowsOf(c.magnitude)} xLabel="time (s)" yLabel="frequency (Hz)" valueLabel="|W|" />
        </Panel>
      </Subplots>
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
  const [k, setK] = useState(1)
  const s = run.steps[Math.min(k, run.steps.length - 1)]
  const prev = run.steps[Math.max(0, Math.min(k, run.steps.length - 1) - 1)]
  const idx = x.map((_, i) => i)
  const series: XYSeries[] =
    k === 0
      ? [{ name: 'h', type: 'line', x: idx, y: toFlat(s.h), slot: 0 }]
      : [
          { name: 'h before', type: 'line', x: idx, y: toFlat(prev.h), slot: 0 },
          { name: 'upper envelope', type: 'line', x: idx, y: toFlat(s.upper), slot: 1, dashed: true },
          { name: 'lower envelope', type: 'line', x: idx, y: toFlat(s.lower), slot: 2, dashed: true },
          { name: 'mean', type: 'line', x: idx, y: toFlat(s.mean), emphasis: true },
        ]
  return (
    <Figure
      title="Sifting one intrinsic mode function"
      description="Each sift fits cubic splines through the maxima and minima and subtracts their mean; after a few sifts the mean is near zero and h is the fastest oscillation, the first IMF."
      defaultSize="L"
      controls={<Player label="sift" value={k} onChange={setK} count={run.steps.length} defaultSpeed={2} />}
      readouts={
        <>
          <Readout label="maxima" value={s.maxima.shape[0]} />
          <Readout label="minima" value={s.minima.shape[0]} />
          <Readout
            label="max |mean|"
            value={s.mean.shape[0] ? Math.max(...toFlat(s.mean).map(Math.abs)).toFixed(4) : '—'}
          />
        </>
      }
      caption="aifn/dsp siftSteps on sin(2π·30t) + 1.2 sin(2π·4t) + t, twelve fixed sifts."
    >
      <XYChart series={series} xLabel="sample" yLabel="value" rescaleOnChange={false} holdFit="union" />
    </Figure>
  )
}
