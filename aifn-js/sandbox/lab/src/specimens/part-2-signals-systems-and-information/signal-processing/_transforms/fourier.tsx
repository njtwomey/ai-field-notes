/**
 * "Fourier: from sums of sinusoids to the DFT": build a record from sinusoids and read its DFT; leakage and windows;
 * zero-padding (interpolation, not resolution); aliasing; and reconstruction by the inverse DFT.
 */
import { useMemo } from 'react'
import { fft, ifft } from 'aifn/foundation/fourier'
import { fromData, toComplexFlat, type Tensor } from 'aifn/foundation/tensor'
import type { WindowSpec } from 'aifn/signal/windows'
import { Figure } from '@lab/layout'
import { choice, row, setting, slider, toggle, useFigureState } from '@lab/state'
import { Annotation, Curve, Handle, Plot, Plots, Points, Readout, Segments, useAxis } from '@lab/viz'
import { amplitudeSpectrum, db, fmt, range, stems } from './common'

const TAU = 2 * Math.PI
const WINDOWS = [
  { value: 'rectangular', label: 'rectangular' },
  { value: 'hann', label: 'Hann' },
  { value: 'hamming', label: 'Hamming' },
  { value: 'blackman', label: 'Blackman' },
  { value: 'flattop', label: 'flat top' },
] as const

type Tone = { amp: number; freq: number; phase: number }
const tone = (amp: number, freq: number, phase: number) =>
  row(`sinusoid`, {
    amp: slider(0, 1.5, amp, { label: 'amplitude A', step: 0.01 }),
    freq: slider(0, 0.5, freq, { label: 'frequency f (cycles/sample)', step: 0.001 }),
    phase: slider(-Math.PI, Math.PI, phase, { label: 'phase φ (rad)', step: 0.01 }),
  })

/** x[n] = Σ A cos(2πfn + φ) at real-valued n. */
const sumAt = (tones: readonly Tone[], t: number) =>
  tones.reduce((s, c) => s + c.amp * Math.cos(TAU * c.freq * t + c.phase), 0)

// ── 1. A record built from sinusoids, and its DFT ─────────────────────────────────────────────────────────────────

export function SinusoidsToDftFigure() {
  const state = useFigureState({
    record: row('1 · record', {
      n: choice([32, 64, 128], 64, { label: 'samples N' }),
      window: choice(WINDOWS, 'rectangular', { label: 'window' }),
      pad: choice([1, 2, 4, 8, 16], 1, { label: 'zero-padding ×' }),
      snap: setting(false, 'snap frequencies to DFT bins'),
    }),
    a: { ...tone(1, 0.086, 0), label: '2 · sinusoid 1' },
    b: { ...tone(0.6, 0.203, 1.2), label: '3 · sinusoid 2' },
    c: { ...tone(0, 0.35, -0.8), label: '4 · sinusoid 3' },
  })
  const { n, window, pad, snap } = state.record
  const raw = [state.a, state.b, state.c]
  const tones = raw.map((c) => ({ ...c, freq: snap ? Math.round(c.freq * n) / n : c.freq }))
  const key = JSON.stringify(tones)
  const x = useMemo(() => {
    const ts = JSON.parse(key) as Tone[]
    return range(n).map((i) => sumAt(ts, i))
  }, [key, n])
  const fine = useMemo(() => {
    const ts = JSON.parse(key) as Tone[]
    const t = Array.from({ length: 8 * n + 1 }, (_, i) => i / 8)
    return { t, y: t.map((s) => sumAt(ts, s)) }
  }, [key, n])
  const dft = useMemo(() => amplitudeSpectrum(x, window as WindowSpec, 1), [x, window])
  const padded = useMemo(() => amplitudeSpectrum(x, window as WindowSpec, pad), [x, window, pad])
  const shown = useMemo(() => {
    const top = Math.max(...dft.amp)
    const keep = dft.amp.map((a) => a > 1e-3 * top)
    return {
      f: dft.f.filter((_, k) => keep[k]),
      phase: dft.phase.filter((_, k) => keep[k]),
    }
  }, [dft])
  const parseval = useMemo(() => {
    const time = x.reduce((s, v) => s + v * v, 0)
    const X = toComplexFlat(fft(x) as Tensor)
    const freq = X.reduce((s, z) => s + z.re * z.re + z.im * z.im, 0) / n
    return { time, freq }
  }, [x, n])
  const time = useAxis({ label: 'sample n', range: [0, n] })
  const value = useAxis({ label: 'x[n]', range: [-3, 3] })
  const freq = useAxis({ label: 'frequency (cycles/sample)', range: [0, 0.5] })
  const amp = useAxis({ label: 'amplitude', range: [0, 1.7] })
  const ph = useAxis({ label: 'phase (rad)', range: [-3.5, 3.5] })
  const labels = ['1', '2', '3']
  return (
    <Figure
      title="A sum of sinusoids and its DFT"
      purpose="The DFT of N samples measures the record against N/2 + 1 sinusoids whose frequencies are whole numbers of cycles per record; a component between two of them leaks into every bin."
      defaultSize="L"
      state={state}
      readouts={{
        'cycles per record (f·N)': (
          <>
            {tones.map((c, i) => (
              <Readout key={i} label={`sinusoid ${labels[i]}`} value={fmt(c.freq * n, 2)} />
            ))}
          </>
        ),
        Parseval: (
          <>
            <Readout label="Σ x²" value={fmt(parseval.time, 4)} />
            <Readout label="Σ |X|² / N" value={fmt(parseval.freq, 4)} />
          </>
        ),
      }}
      caption="x[n] = Σ A cos(2πfn + φ). Drag a dot on the amplitude spectrum to move a sinusoid in frequency and amplitude, or on the phase panel to set its phase; the sliders hold the same values. Stems are the N-point DFT scaled so a sinusoid on a bin reads A; the curve is the zero-padded DFT, samples of the DTFT between the bins. Turn on snapping, or set f·N to a whole number: the leakage disappears. A window trades it for a wider peak."
    >
      <Plots rows={3} heights={[1, 1.2, 0.9]}>
        <Plot x={time} y={value}>
          <Curve name="Σ A cos(2πft + φ)" x={fine.t} y={fine.y} muted thin />
          <Segments name="x[n]" segments={stems(range(n), x)} slot={0} />
          <Points name="x[n]" x={range(n)} y={x} slot={0} size={5} />
        </Plot>
        <Plot x={freq} y={amp}>
          <Curve name="zero-padded DFT" x={padded.f} y={padded.amp} slot={1} />
          <Segments name="|X_k|" segments={stems(dft.f, dft.amp)} slot={0} />
          <Points name="|X_k|" x={dft.f} y={dft.amp} slot={0} size={5} />
          {(['a', 'b', 'c'] as const).map((k, i) => (
            <Handle key={k} {...state.handle([`${k}.freq`, `${k}.amp`], { label: labels[i] })} />
          ))}
        </Plot>
        <Plot x={freq} y={ph}>
          <Segments name="arg X_k" segments={stems(shown.f, shown.phase)} slot={0} />
          <Points name="arg X_k" x={shown.f} y={shown.phase} slot={0} size={5} />
          {(['a', 'b', 'c'] as const).map((k, i) => (
            <Handle key={k} {...state.handle([`${k}.freq`, `${k}.phase`], { label: labels[i] })} />
          ))}
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── 2. Leakage and the window ─────────────────────────────────────────────────────────────────────────────────────

const ALL_WINDOWS: readonly { value: string; label: string; spec: WindowSpec }[] = [
  { value: 'rectangular', label: 'rectangular', spec: 'rectangular' },
  { value: 'hann', label: 'Hann', spec: 'hann' },
  { value: 'hamming', label: 'Hamming', spec: 'hamming' },
  { value: 'blackman', label: 'Blackman', spec: 'blackman' },
  { value: 'blackmanharris', label: 'Blackman–Harris', spec: 'blackmanharris' },
  { value: 'flattop', label: 'flat top', spec: 'flattop' },
  { value: 'kaiser', label: 'Kaiser β = 8.6', spec: { name: 'kaiser', beta: 8.6 } },
]

export function LeakageFigure() {
  const state = useFigureState({
    tones: row('1 · tones', {
      offset: slider(0, 1, 0.5, { label: 'strong tone: bins past k = 10', step: 0.01 }),
      gap: slider(1.5, 12, 6, { label: 'weak tone: bins above the strong one', step: 0.1 }),
      level: slider(-90, -10, -50, { label: 'weak tone level (dB)', step: 1 }),
    }),
    window: choice(
      ALL_WINDOWS.map(({ value, label }) => ({ value, label })),
      'rectangular',
      { label: 'window' },
    ),
  })
  const N = 64
  const { offset, gap, level } = state.tones
  const spec = ALL_WINDOWS.find((w) => w.value === state.window)!.spec
  const k1 = 10 + offset
  const k2 = k1 + gap
  const x = useMemo(
    () => range(N).map((i) => Math.cos((TAU * k1 * i) / N) + 10 ** (level / 20) * Math.cos((TAU * k2 * i) / N + 0.7)),
    [k1, k2, level],
  )
  const dft = useMemo(() => amplitudeSpectrum(x, spec, 1), [x, spec])
  const dtft = useMemo(() => amplitudeSpectrum(x, spec, 32), [x, spec])
  const view = useMemo(
    () => ({
      bins: dft.f.map((f) => f * N),
      db: dft.amp.map((a) => db(a)),
      fineBins: dtft.f.map((f) => f * N),
      fineDb: dtft.amp.map((a) => db(a)),
    }),
    [dft, dtft],
  )
  const peak = Math.max(...view.db.slice(8, 14))
  const nearWeak = Math.round(k2)
  const bins = useAxis({ label: 'frequency (DFT bins, k = fN)', range: [0, 32] })
  const level_ = useAxis({ label: 'amplitude (dB)', range: [-120, 5] })
  return (
    <Figure
      title="Leakage and the window"
      purpose="A tone that does not complete a whole number of cycles in the record spreads into every bin; the window sets how fast that leakage falls off, and with it whether a weak tone nearby is visible."
      defaultSize="L"
      state={state}
      readouts={{
        'strong tone': (
          <>
            <Readout label="cycles in record" value={fmt(k1, 2)} />
            <Readout label="largest DFT sample" value={`${fmt(peak, 2)} dB`} />
            <Readout label="scalloping loss" value={`${fmt(-peak, 2)} dB`} />
          </>
        ),
        'weak tone': (
          <>
            <Readout label="true level" value={`${level} dB`} />
            <Readout label={`DFT at bin ${nearWeak}`} value={`${fmt(view.db[nearWeak] ?? NaN, 1)} dB`} />
          </>
        ),
      }}
      caption="N = 64 samples of a unit tone at 10 + offset bins plus a weak tone higher up. Dots: the 64-point DFT; curve: the same record zero-padded ×32 (the DTFT). Drag the line to move the strong tone between bins: with the rectangular window, a half-bin offset drops its largest DFT sample by 3.9 dB and leakage at −40 dB buries a −50 dB neighbour; Blackman–Harris or Kaiser show it at the cost of a wider main lobe."
    >
      <Plot x={bins} y={level_}>
        <Curve name="DTFT (zero-padded)" x={view.fineBins} y={view.fineDb} slot={1} />
        <Points name="DFT samples" x={view.bins} y={view.db} slot={0} size={5} />
        <Annotation x={k2} text="weak tone" dashed />
        <Handle kind="x" at={k1} label="strong tone" onDrag={(v) => state.set('tones.offset', v - 10)} />
      </Plot>
    </Figure>
  )
}

// ── 3. Zero-padding interpolates; a longer record resolves ────────────────────────────────────────────────────────

export function ZeroPaddingFigure() {
  const state = useFigureState({
    tones: row('1 · two tones', {
      sep: slider(0.002, 0.05, 0.012, { label: 'separation Δf (cycles/sample)', step: 0.001 }),
    }),
    analysis: row('2 · analysis', {
      n: choice([32, 64, 128, 256], 64, { label: 'record length N' }),
      pad: choice([1, 2, 4, 8, 16, 32], 1, { label: 'zero-padding ×' }),
    }),
  })
  const { sep } = state.tones
  const { n, pad } = state.analysis
  const f1 = 0.2
  const f2 = f1 + sep
  const x = useMemo(() => range(n).map((i) => Math.cos(TAU * f1 * i) + Math.cos(TAU * f2 * i + 1)), [n, f2])
  const s = useMemo(() => amplitudeSpectrum(x, 'hann', pad), [x, pad])
  const dft = useMemo(() => amplitudeSpectrum(x, 'hann', 1), [x])
  // Resolved: a dip of at least 3 dB between the two peaks of the finely padded spectrum.
  const resolved = useMemo(() => {
    const fine = amplitudeSpectrum(x, 'hann', 64)
    const between = fine.f.map((f, k) => [f, fine.amp[k]] as const).filter(([f]) => f >= f1 && f <= f2)
    if (between.length < 3) return false
    const ends = Math.min(between[0][1], between[between.length - 1][1])
    const lowest = Math.min(...between.map(([, a]) => a))
    return db(lowest) < db(ends) - 3
  }, [x, f2])
  const freq = useAxis({ label: 'frequency (cycles/sample)', range: [0.12, 0.3] })
  const amp = useAxis({ label: 'amplitude (Hann)', range: [0, 1.2] })
  return (
    <Figure
      title="Zero-padding interpolates; a longer record resolves"
      purpose="Padding the record with zeros samples the same DTFT more finely, so peaks look smoother but never separate; only more data (a larger N) narrows the main lobe, to about 2/N cycles per sample under a Hann window."
      defaultSize="M"
      state={state}
      readouts={
        <>
          <Readout label="Hann main lobe 4/N" value={fmt(4 / n, 4)} />
          <Readout label="separation" value={fmt(sep, 4)} />
          <Readout label="resolved (3 dB dip)" value={resolved ? 'yes' : 'no'} />
        </>
      }
      caption="Two unit tones at 0.2 and 0.2 + Δf cycles/sample, Hann window. Raise the padding at N = 64 with Δf = 0.012: the curve gets smoother but keeps one peak. Double N instead: two peaks appear once Δf exceeds about 2/N."
    >
      <Plot x={freq} y={amp}>
        <Curve name={`DFT, ${n * pad} points`} x={s.f} y={s.amp} slot={1} showPoints={pad <= 4} />
        <Points name="N-point DFT" x={dft.f} y={dft.amp} slot={0} size={6} />
        <Annotation x={f1} dashed />
        <Annotation x={f2} dashed />
      </Plot>
    </Figure>
  )
}

// ── 4. Aliasing ───────────────────────────────────────────────────────────────────────────────────────────────────

const FS = 10

/** The frequency in [0, fs/2] that a sinusoid at f is indistinguishable from after sampling at fs. */
const alias = (f: number, fs: number) => Math.abs(f - fs * Math.round(f / fs))

export function AliasingFigure() {
  const state = useFigureState({
    f: slider(0, 25, 7, { label: 'true frequency f (Hz)', step: 0.05 }),
    reveal: row('reveal', { alias: toggle(true, 'the alias f_a') }),
  })
  const f = state.f
  const fa = alias(f, FS)
  const fine = useMemo(() => {
    const t = Array.from({ length: 1501 }, (_, i) => (i / 1500) * 1.5)
    // f = m fs + r with |r| ≤ fs/2: the samples are those of cos(2π|r|t ± φ), the sign of φ that of r.
    const r = f - FS * Math.round(f / FS)
    const sign = r < 0 ? -1 : 1
    return { t, y: t.map((s) => Math.cos(TAU * f * s + 0.4)), a: t.map((s) => Math.cos(TAU * fa * s + 0.4 * sign)) }
  }, [f, fa])
  const samples = useMemo(() => {
    const t = range(16).map((i) => i / FS)
    return { t, y: t.map((s) => Math.cos(TAU * f * s + 0.4)) }
  }, [f])
  const folding = useMemo(() => {
    const g = Array.from({ length: 501 }, (_, i) => (i / 500) * 25)
    return { f: g, a: g.map((v) => alias(v, FS)) }
  }, [])
  const time = useAxis({ label: 'time (s)', range: [0, 1.5] })
  const value = useAxis({ label: 'x(t)', range: [-1.3, 1.3] })
  const trueF = useAxis({ label: 'true frequency f (Hz)', range: [0, 25] })
  const seen = useAxis({ label: 'apparent frequency (Hz)', range: [0, 5.5] })
  return (
    <Figure
      title="Aliasing: frequencies fold at Nyquist"
      purpose="Sampled at fs, the sinusoids at f and at |f − k fs| for every integer k give the same samples, so a frequency above fs/2 shows up folded back into [0, fs/2]."
      defaultSize="L"
      state={state}
      readouts={
        <>
          <Readout label="fs" value={`${FS} Hz`} />
          <Readout label="Nyquist fs/2" value={`${FS / 2} Hz`} />
          <Readout label="apparent frequency" value={`${fmt(fa, 2)} Hz`} />
        </>
      }
      caption="cos(2πft + 0.4) sampled at 10 Hz. Drag the line on the lower panel past 5 Hz: the samples (dots) stop following the fast curve and trace the slower alias (dashed) instead, at the frequency the sawtooth reads off. At f = 10 Hz the samples are constant."
    >
      <Plots rows={2} heights={[1.2, 1]}>
        <Plot x={time} y={value}>
          <Curve name="x(t)" x={fine.t} y={fine.y} slot={0} thin />
          {state.reveal.alias && <Curve name="alias" x={fine.t} y={fine.a} slot={1} dashed />}
          <Points name="samples x(n/fs)" x={samples.t} y={samples.y} emphasis size={7} />
        </Plot>
        <Plot x={trueF} y={seen}>
          <Curve name="apparent frequency" x={folding.f} y={folding.a} slot={2} />
          {[5, 10, 15, 20, 25].map((v) => (
            <Annotation key={v} x={v} dashed text={v === 5 ? 'fs/2' : v % 10 === 5 ? `${v / 5}fs/2` : `${v / 10}fs`} />
          ))}
          <Points name="this sinusoid" x={[f]} y={[fa]} emphasis size={9} live />
          <Handle {...state.handle('f', { label: 'f' })} />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── 5. The inverse DFT, term by term ──────────────────────────────────────────────────────────────────────────────

const SHAPES = [
  { value: 'square', label: 'square wave' },
  { value: 'saw', label: 'sawtooth' },
  { value: 'pulse', label: 'narrow pulse' },
  { value: 'mix', label: 'three sinusoids' },
] as const

export function InverseDftFigure() {
  const state = useFigureState({
    signal: row('1 · signal', { shape: choice(SHAPES, 'square', { label: 'signal' }) }),
    keep: row('2 · inverse DFT', {
      order: choice(
        [
          { value: 'low', label: 'lowest frequencies first' },
          { value: 'large', label: 'largest coefficients first' },
        ],
        'low',
        { label: 'which bins' },
      ),
      k: slider(0, 64, 4, { label: 'bins kept (each with its mirror)', step: 1 }),
    }),
  })
  const N = 128
  const { shape } = state.signal
  const { order, k } = state.keep
  const x = useMemo(
    () =>
      range(N).map((i) => {
        const u = i / N
        if (shape === 'square') return u < 0.5 ? 1 : -1
        if (shape === 'saw') return 2 * u - 1
        if (shape === 'pulse') return Math.abs(u - 0.5) < 0.03 ? 1 : 0
        return Math.cos(TAU * 3 * u) + 0.5 * Math.cos(TAU * 11 * u + 1) + 0.25 * Math.cos(TAU * 30 * u)
      }),
    [shape],
  )
  const X = useMemo(() => toComplexFlat(fft(x) as Tensor), [x])
  const kept = useMemo(() => {
    // Bins 0 … N/2 ranked; each kept bin brings its mirror N − k so the inverse stays real.
    const half = range(N / 2 + 1)
    const ranked =
      order === 'low' ? half : [...half].sort((a, b) => Math.hypot(X[b].re, X[b].im) - Math.hypot(X[a].re, X[a].im))
    const set = new Set(ranked.slice(0, k + 1))
    const masked = new Float64Array(2 * N)
    for (let j = 0; j < N; j++) {
      const m = j <= N / 2 ? j : N - j
      if (!set.has(m)) continue
      masked[2 * j] = X[j].re
      masked[2 * j + 1] = X[j].im
    }
    const y = toComplexFlat(ifft(fromData(masked, [N], 'complex128')) as Tensor).map((z) => z.re)
    const energy = X.reduce((s, z) => s + z.re ** 2 + z.im ** 2, 0)
    const keptEnergy = X.reduce((s, z, j) => s + (set.has(j <= N / 2 ? j : N - j) ? z.re ** 2 + z.im ** 2 : 0), 0)
    return { set, y, fraction: keptEnergy / energy }
  }, [X, order, k])
  const mags = useMemo(() => range(N / 2 + 1).map((j) => (Math.hypot(X[j].re, X[j].im) * (j === 0 ? 1 : 2)) / N), [X])
  const keptBins = range(N / 2 + 1).filter((j) => kept.set.has(j))
  const droppedBins = range(N / 2 + 1).filter((j) => !kept.set.has(j))
  const rmse = Math.sqrt(x.reduce((s, v, i) => s + (v - kept.y[i]) ** 2, 0) / N)
  const n = useAxis({ label: 'sample n' })
  const v = useAxis({ label: 'x[n]', key: shape, hold: 'initial' })
  const bin = useAxis({ label: 'bin k (cycles per record)', range: [0, 64] })
  const a = useAxis({ label: 'amplitude', key: shape, hold: 'initial' })
  return (
    <Figure
      title="The inverse DFT, term by term"
      purpose="The inverse DFT writes the record as a sum of its N sinusoids; keeping only some bins gives the best approximation from those frequencies, and the overshoot at a jump (Gibbs) does not shrink as more are added, it only narrows."
      defaultSize="L"
      state={state}
      readouts={
        <>
          <Readout label="bins kept" value={`${kept.set.size} of ${N / 2 + 1}`} />
          <Readout label="energy kept" value={`${fmt(100 * kept.fraction, 2)} %`} />
          <Readout label="rms error" value={fmt(rmse, 4)} />
        </>
      }
      caption="N = 128. Step the bins kept with the arrows: the square wave's partial sums ring at the jumps with an overshoot near 9 % of the jump however many bins are kept. Order by size on the three sinusoids: three bins reconstruct them exactly. With every bin kept the inverse DFT returns x to rounding."
    >
      <Plots rows={2} heights={[1.3, 1]}>
        <Plot x={n} y={v}>
          <Curve name="x[n]" x={range(N)} y={x} muted />
          <Curve name="inverse DFT of the kept bins" x={range(N)} y={kept.y} slot={0} />
        </Plot>
        <Plot x={bin} y={a}>
          <Segments
            name="dropped"
            segments={stems(
              droppedBins,
              droppedBins.map((j) => mags[j]),
            )}
            muted
          />
          <Segments
            name="kept"
            segments={stems(
              keptBins,
              keptBins.map((j) => mags[j]),
            )}
            slot={0}
          />
          <Points name="kept" x={keptBins} y={keptBins.map((j) => mags[j])} slot={0} size={5} />
        </Plot>
      </Plots>
    </Figure>
  )
}
