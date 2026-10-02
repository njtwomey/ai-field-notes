/**
 * "Sampling, aliasing and multirate": sampling and sinc reconstruction, decimation with and without the anti-alias
 * filter, interpolation by zero insertion and low-pass filtering, and the polyphase form of a decimating filter.
 */
import { useMemo } from 'react'
import { convolve, upfirdn } from 'aifn/foundation/convolution'
import { toFlat, type Tensor } from 'aifn/foundation/tensor'
import { signal } from 'aifn/signal'
import { firwin } from 'aifn/signal/filters'
import { decimateSignal, polyphase, resamplePoly, sincInterpolate } from 'aifn/signal/multirate'
import { Diagram, type DiagramSpec } from '@lab/diagram'
import { Figure } from '@lab/layout'
import { choice, row, slider, toggle, useFigureState } from '@lab/state'
import { Annotation, Curve, Handle, Plot, Plots, Points, Readout, Segments, useAxis } from '@lab/viz'
import { amplitudeSpectrum, db, fmt, range, rows, stems } from './common'

const TAU = 2 * Math.PI
const LINES = [
  { f: 1.3, a: 1, phase: 0 },
  { f: 3.1, a: 0.6, phase: 1 },
]
const x0 = (t: number) => LINES.reduce((s, l) => s + l.a * Math.sin(TAU * l.f * t + l.phase), 0)

// ── 1. Sampling and sinc reconstruction ───────────────────────────────────────────────────────────────────────────

export function SamplingFigure() {
  const state = useFigureState({
    fs: slider(2, 16, 9, { label: 'sampling rate fs (Hz)', step: 0.1 }),
    reveal: row('reveal', { recon: toggle(true, 'sinc reconstruction'), images: toggle(true, 'spectral images') }),
  })
  const fs = state.fs
  const T = 4
  const fine = useMemo(() => range(801).map((i) => (i / 800) * T), [])
  const truth = useMemo(() => fine.map(x0), [fine])
  const samples = useMemo(() => {
    const n = Math.floor(T * fs) + 1
    const t = range(n).map((i) => i / fs)
    return { t, y: t.map(x0) }
  }, [fs])
  const recon = useMemo(() => toFlat(sincInterpolate(signal(samples.y, { fs }), fine)), [samples, fs, fine])
  const interior = fine.map((t, i) => [t, i] as const).filter(([t]) => t > 1 && t < 3)
  const rms = Math.sqrt(interior.reduce((s, [, i]) => s + (recon[i] - truth[i]) ** 2, 0) / interior.length)
  const spectrum = useMemo(() => {
    const images: { f: number; a: number }[] = []
    for (const l of LINES)
      for (let k = -4; k <= 4; k++)
        for (const f of [k * fs + l.f, k * fs - l.f])
          if (f > 0 && f < 20 && Math.abs(f - l.f) > 1e-9) images.push({ f, a: l.a })
    return images
  }, [fs])
  const time = useAxis({ label: 'time (s)', range: [0, T] })
  const v = useAxis({ label: 'x(t)', range: [-2.2, 2.2] })
  const f = useAxis({ label: 'frequency (Hz)', range: [0, 20] })
  const a = useAxis({ label: 'amplitude', range: [0, 1.2] })
  return (
    <Figure
      title="Sampling and sinc reconstruction"
      purpose="Samples at fs determine a signal whose spectrum lies below fs/2: the sinc series Σ x[n] sinc(fs t − n) rebuilds it between the samples. Below twice the highest frequency, a spectral image falls inside [0, fs/2] and the reconstruction is a different signal."
      defaultSize="L"
      state={state}
      readouts={
        <>
          <Readout label="Nyquist fs/2" value={`${fmt(fs / 2, 2)} Hz`} />
          <Readout label="needed: fs > 2 × 3.1" value="6.2 Hz" />
          <Readout label="rms error on [1, 3] s" value={fmt(rms, 3)} />
        </>
      }
      caption="x(t) = sin(2π·1.3t) + 0.6 sin(2π·3.1t + 1) sampled for 4 s; reconstruction by aifn sincInterpolate (Whittaker–Shannon). Drag the Nyquist line on the spectrum below 3.1 Hz: the image of the 3.1 Hz line at fs − 3.1 Hz enters the band and the reconstruction follows it. The finite record leaves errors near both ends even at high fs."
    >
      <Plots rows={2} heights={[1.3, 1]}>
        <Plot x={time} y={v}>
          <Curve name="x(t)" x={fine} y={truth} muted />
          {state.reveal.recon && <Curve name="sinc reconstruction" x={fine} y={recon} slot={0} />}
          <Points name="samples" x={samples.t} y={samples.y} emphasis size={6} />
        </Plot>
        <Plot x={f} y={a}>
          <Segments
            name="spectrum of x"
            segments={stems(
              LINES.map((l) => l.f),
              LINES.map((l) => l.a),
            )}
            slot={0}
            width={3}
          />
          {state.reveal.images && (
            <Segments
              name="images k fs ± f"
              segments={stems(
                spectrum.map((s) => s.f),
                spectrum.map((s) => s.a),
              )}
              slot={1}
              dashed
            />
          )}
          <Annotation x={fs} text="fs" dashed />
          <Handle kind="x" at={fs / 2} label="fs/2" onDrag={(v) => state.set('fs', 2 * v)} />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── 2. Decimation ─────────────────────────────────────────────────────────────────────────────────────────────────

const DECIMATE: DiagramSpec = {
  nodes: [
    { id: 'x', x: 0, y: 0, shape: 'circle', label: '$x[n]$' },
    { id: 'h', x: 2.5, y: 0, label: 'low-pass\n$H(z)$, edge $0.8 f_s/2q$' },
    { id: 'd', x: 5, y: 0, label: '$\\downarrow q$' },
    { id: 'y', x: 7, y: 0, shape: 'circle', label: '$y[m]$' },
  ],
  edges: [
    { from: 'x', to: 'h' },
    { from: 'h', to: 'd' },
    { from: 'd', to: 'y' },
  ],
}

export function DecimationFigure() {
  const state = useFigureState({
    q: slider(2, 8, 4, { label: 'decimation factor q', step: 1 }),
    filter: toggle(true, 'anti-alias filter'),
  })
  const { q, filter } = state
  const fs = 1000
  const n = 1024
  const x = useMemo(
    () => range(n).map((i) => Math.sin((TAU * 40 * i) / fs) + 0.8 * Math.sin((TAU * 310 * i) / fs + 0.5)),
    [],
  )
  const y = useMemo(
    () => (filter ? toFlat(decimateSignal(signal(x, { fs }), q).data) : x.filter((_, i) => i % q === 0)),
    [x, q, filter],
  )
  const sx = useMemo(() => amplitudeSpectrum(x, 'blackmanharris', 1), [x])
  const sy = useMemo(() => amplitudeSpectrum(y, 'blackmanharris', q), [y, q])
  const fsy = fs / q
  const alias = Math.abs(310 - fsy * Math.round(310 / fsy))
  const f = useAxis({ label: 'frequency (Hz)', range: [0, fs / 2] })
  const a = useAxis({ label: 'amplitude (dB)', range: [-100, 5] })
  const time = useAxis({ label: 'time (ms)', range: [0, 100] })
  const v = useAxis({ label: 'value', range: [-2, 2] })
  return (
    <Figure
      title="Decimation, with and without the anti-alias filter"
      purpose="Keeping every q-th sample lowers the Nyquist frequency to fs/2q; anything above it folds down unless a low-pass removes it first."
      defaultSize="L"
      state={state}
      readouts={
        <>
          <Readout label="new rate" value={`${fmt(fsy, 1)} Hz`} />
          <Readout label="new Nyquist" value={`${fmt(fsy / 2, 1)} Hz`} />
          <Readout label="310 Hz would alias to" value={310 < fsy / 2 ? 'no alias' : `${fmt(alias, 1)} Hz`} />
        </>
      }
      caption="A 40 Hz tone and a 310 Hz tone at 1 kHz, decimated by q with aifn decimateSignal (Chebyshev I, zero phase) or by keeping every q-th sample. With the filter off and q ≥ 2, the 310 Hz line reappears at its alias; with it on, it is removed before it can fold."
    >
      <Diagram spec={DECIMATE} ariaLabel="x through a low-pass filter, then downsampled by q" height={70} />
      <Plots rows={2} heights={[1.3, 1]}>
        <Plot x={f} y={a}>
          <Curve name="input" x={sx.f.map((v) => v * fs)} y={sx.amp.map((v) => db(v))} muted />
          <Curve name={`output (fs/${q})`} x={sy.f.map((v) => v * fsy)} y={sy.amp.map((v) => db(v))} slot={0} />
          <Annotation x={fsy / 2} dashed text="new Nyquist" />
        </Plot>
        <Plot x={time} y={v}>
          <Curve name="input" x={range(n).map((i) => (1000 * i) / fs)} y={x} muted thin />
          <Points name="output samples" x={range(y.length).map((i) => (1000 * i * q) / fs)} y={y} slot={0} size={5} />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── 3. Interpolation ──────────────────────────────────────────────────────────────────────────────────────────────

export function InterpolationFigure() {
  const state = useFigureState({ L: slider(2, 6, 3, { label: 'interpolation factor L', step: 1 }) })
  const L = state.L
  const n = 64
  const x = useMemo(() => range(n).map((i) => Math.sin((TAU * 3 * i) / n) + 0.4 * Math.sin((TAU * 11 * i) / n)), [])
  const stuffed = useMemo(() => range(n * L).map((j) => (j % L === 0 ? x[j / L] : 0)), [x, L])
  const interp = useMemo(() => toFlat(resamplePoly(x, L, 1).data), [x, L])
  const s1 = useMemo(() => amplitudeSpectrum(stuffed, 'rectangular', 1), [stuffed])
  const s2 = useMemo(() => amplitudeSpectrum(interp, 'rectangular', 1), [interp])
  const t = useAxis({ label: 'input sample index', range: [0, 32] })
  const v = useAxis({ label: 'value', range: [-1.6, 1.6] })
  const f = useAxis({ label: 'frequency (cycles per input sample)', range: [0, L / 2] })
  const a = useAxis({ label: 'amplitude (dB)', range: [-80, 5] })
  const tt = range(n * L).map((j) => j / L)
  return (
    <Figure
      title="Interpolation: zero insertion, then a low-pass"
      purpose="Inserting L − 1 zeros between samples raises the rate but copies the spectrum L − 1 times (images); a low-pass at the old Nyquist removes the images and fills the zeros with interpolated values."
      defaultSize="L"
      state={state}
      caption="64 samples of two tones, upsampled by L. Stems: the zero-stuffed sequence (scaled by L in the spectrum); curve: aifn resamplePoly(x, L, 1), whose polyphase FIR does both steps at once. The spectrum of the zero-stuffed signal repeats every input-rate cycle; the interpolated one keeps only the first copy."
    >
      <Plots rows={2}>
        <Plot x={t} y={v}>
          <Segments name="zero-stuffed" segments={stems(tt, stuffed)} muted live />
          <Curve name="interpolated" x={tt} y={interp} slot={0} showPoints />
          <Points name="input" x={range(n)} y={x} emphasis size={7} />
        </Plot>
        <Plot x={f} y={a}>
          <Curve name="zero-stuffed (images)" x={s1.f.map((v) => v * L)} y={s1.amp.map((v) => db(v * L))} slot={1} />
          <Curve name="interpolated" x={s2.f.map((v) => v * L)} y={s2.amp.map((v) => db(v))} slot={0} />
          <Annotation x={0.5} dashed text="input Nyquist" />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── 4. Polyphase decimation ───────────────────────────────────────────────────────────────────────────────────────

export function PolyphaseFigure() {
  const state = useFigureState({
    M: choice([2, 3, 4], 3, { label: 'branches M (decimation factor)' }),
    taps: slider(12, 60, 36, { label: 'prototype taps', step: 1 }),
  })
  const { M, taps } = state
  const h = useMemo(() => toFlat(firwin(taps, 1 / M).repr.b as Tensor), [taps, M])
  const E = useMemo(() => rows(polyphase(h, M)), [h, M])
  const x = useMemo(
    () => range(240).map((i) => Math.sin(i * 0.07) + 0.5 * Math.sin(i * 0.9 + 1) + 0.3 * Math.cos(i * 2.1)),
    [],
  )
  const check = useMemo(() => {
    const direct = toFlat(upfirdn(h, x, { down: M }) as Tensor)
    // Branch k filters the input delayed by k and kept every M-th sample: uₖ[m] = x[mM − k].
    const out = new Float64Array(direct.length)
    E.forEach((e, k) => {
      const u = range(Math.ceil((x.length + k) / M)).map((m) => x[m * M - k] ?? 0)
      const yk = toFlat(convolve(u, e) as Tensor)
      for (let m = 0; m < out.length; m++) out[m] += yk[m] ?? 0
    })
    return { gap: Math.max(...direct.map((v, i) => Math.abs(v - out[i]))), direct }
  }, [h, x, E, M])
  const k = useAxis({ label: 'tap index n', range: [-1, taps] })
  const hv = useAxis({ label: 'h[n]', hold: 'union', key: M })
  return (
    <Figure
      title="Polyphase decimation"
      purpose="A filter followed by ↓M computes M − 1 outputs it then throws away; splitting h into M branches Eₖ[p] = h[pM + k], each run at the low rate on its own phase of the input, gives the same outputs for 1/M of the multiplications."
      defaultSize="M"
      state={state}
      readouts={
        <>
          <Readout label="multiplies per output, direct" value={taps * M} />
          <Readout label="polyphase" value={taps} />
          <Readout label="max |direct − polyphase|" value={check.gap.toExponential(1)} />
        </>
      }
      caption="The prototype is aifn firwin(taps, 1/M); colours mark the branch each tap belongs to (aifn polyphase). The output of upfirdn (filter, then keep every M-th sample) and the sum of the M low-rate branch filters agree to rounding."
    >
      <Plot x={k} y={hv}>
        {E.map((e, b) => {
          const n = e.map((_, p) => p * M + b)
          return [
            <Segments key={`s${b}`} name={`E${b}`} segments={stems(n, e)} slot={b} live />,
            <Points key={`p${b}`} name={`E${b}`} x={n} y={e} slot={b} size={5} />,
          ]
        })}
      </Plot>
    </Figure>
  )
}
