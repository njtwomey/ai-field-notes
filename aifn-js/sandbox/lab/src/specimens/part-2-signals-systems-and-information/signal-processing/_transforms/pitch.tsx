/**
 * "Pitch and the cepstrum": a synthetic voiced sound with a known f₀ through formant resonators; its log spectrum,
 * liftered envelope, real cepstrum and YIN difference function; and pitch tracks against the true f₀ under vibrato.
 */
import { useMemo } from 'react'
import { rfft } from 'aifn/foundation/fourier'
import { toComplexFlat, toFlat, type Tensor } from 'aifn/foundation/tensor'
import { voicedSound } from 'aifn-methods/data/signals'
import { cepstralEnvelope, cepstralPitch, yin, yinPitch } from 'aifn/signal/cepstrum'
import { getWindow } from 'aifn/signal/windows'
import { Figure } from '@lab/layout'
import { row, slider, useFigureState } from '@lab/state'
import { Annotation, Curve, Handle, Plot, Plots, Points, Readout, useAxis } from '@lab/viz'
import { fmt, noise, range } from './common'

const FS = 8000
const FRAME = 1024

export function CepstrumPitchFigure() {
  const state = useFigureState({
    voice: row('1 · voice', {
      f0: slider(70, 400, 140, { label: 'true f₀ (Hz)', step: 0.5 }),
      f1: slider(250, 1000, 730, { label: 'formant F₁ (Hz)', step: 10 }),
      f2: slider(800, 2500, 1090, { label: 'formant F₂ (Hz)', step: 10 }),
      sigma: slider(0, 0.3, 0.02, { label: 'noise sd', step: 0.005 }),
    }),
    analysis: row('2 · analysis', {
      lifter: slider(10, 60, 30, { label: 'lifter (samples)', step: 1 }),
      threshold: slider(0.02, 0.5, 0.1, { label: 'YIN threshold', step: 0.01, onChart: true }),
    }),
  })
  const { f0, f1, f2, sigma } = state.voice
  const { lifter, threshold } = state.analysis
  const x = useMemo(() => {
    const v = toFlat(
      voicedSound(FRAME, {
        fs: FS,
        f0,
        formants: [
          { frequency: f1, bandwidth: 90 },
          { frequency: f2, bandwidth: 110 },
          { frequency: 2440, bandwidth: 160 },
        ],
      }).x,
    )
    const w = noise('pitch', FRAME, sigma)
    return v.map((u, i) => u + w[i])
  }, [f0, f1, f2, sigma])
  const spec = useMemo(() => {
    const w = toFlat(getWindow('hann', FRAME))
    const X = toComplexFlat(rfft(x.map((v, i) => v * w[i])) as Tensor)
    const env = cepstralEnvelope(
      x.map((v, i) => v * w[i]),
      { fs: FS, lifter },
    )
    return {
      f: X.map((_, k) => (k * FS) / FRAME),
      db: X.map((z) => 20 * Math.log10(Math.max(Math.hypot(z.re, z.im), 1e-12))),
      envF: toFlat(env.f),
      // Natural log of |X| to dB.
      envDb: toFlat(env.logMagnitude).map((v) => (20 * v) / Math.LN10),
    }
  }, [x, lifter])
  const cep = useMemo(() => cepstralPitch(x, { fs: FS, fmin: 60, fmax: 500 }), [x])
  const y = useMemo(() => yinPitch(x, { fs: FS, fmin: 60, fmax: 500, threshold }), [x, threshold])
  const q = useMemo(() => toFlat(cep.q).map((v) => v * 1000), [cep])
  const c = useMemo(() => toFlat(cep.cepstrum), [cep])
  const lags = useMemo(() => toFlat(y.lags).map((v) => v * 1000), [y])
  const dp = useMemo(() => toFlat(y.dPrime), [y])
  const period = 1000 / f0
  const t = useAxis({ label: 'time (ms)', range: [0, (1000 * FRAME) / FS] })
  const amp = useAxis({ label: 'x', range: [-1.4, 1.4] })
  const f = useAxis({ label: 'frequency (Hz)', range: [0, FS / 2] })
  const level = useAxis({ label: 'log |X| (dB)', hold: 'union' })
  const quef = useAxis({ label: 'quefrency / lag (ms)', range: [0, 18] })
  const cv = useAxis({ label: 'cepstrum', range: [-0.3, 0.6] })
  const yv = useAxis({ label: 'YIN d′(τ)', range: [0, 1.6] })
  return (
    <Figure
      title="Pitch from the cepstrum and from YIN"
      purpose="A voiced sound's spectrum is a comb of harmonics f₀ apart under a smooth formant envelope; the cepstrum separates the two, the envelope at low quefrency and the comb as a peak at one period, 1/f₀. YIN reads the period from where the signal best matches a delayed copy of itself."
      defaultSize="XL"
      state={state}
      readouts={{
        'f₀ (Hz)': (
          <>
            <Readout label="true" value={fmt(f0, 1)} />
            <Readout label="cepstral peak" value={fmt(cep.f0, 1)} />
            <Readout label="YIN" value={`${fmt(y.f0, 1)}${y.voiced ? '' : ' (unvoiced)'}`} />
          </>
        ),
        errors: (
          <>
            <Readout label="cepstrum" value={`${fmt(100 * (cep.f0 / f0 - 1), 2)} %`} />
            <Readout label="YIN" value={`${fmt(100 * (y.f0 / f0 - 1), 2)} %`} />
            <Readout label="YIN aperiodicity d′(τ⋆)" value={fmt(y.aperiodicity, 3)} />
          </>
        ),
      }}
      caption="A 128 ms frame at 8 kHz of aifn-methods voicedSound: glottal pulses at f₀ through three formant resonators, plus white noise. Top right: the Hann-windowed log spectrum and its liftered envelope (aifn cepstralEnvelope; shorten the lifter to smooth it, lengthen it past one period and the harmonics leak back in). Bottom left: the real cepstrum (cepstralPitch), whose peak sits at the true period (dashed). Bottom right: YIN's cumulative-mean-normalised difference d′(τ) (yinPitch); drag the threshold line: YIN takes the first dip below it. Raise f₀ above 300 Hz with a low threshold and the cepstral peak weakens first; with heavy noise both fall back to unvoiced or octave errors."
    >
      <Plots rows={2} cols={2}>
        <Plot x={t} y={amp}>
          <Curve name="x" x={range(FRAME).map((i) => (1000 * i) / FS)} y={x} slot={0} />
        </Plot>
        <Plot x={f} y={level}>
          <Curve name="log |X|" x={spec.f} y={spec.db} muted />
          <Curve name="liftered envelope" x={spec.envF} y={spec.envDb} slot={1} />
        </Plot>
        <Plot x={quef} y={cv}>
          <Curve name="real cepstrum" x={q} y={c} slot={0} />
          <Points name="cepstral peak" x={[1000 * cep.quefrency]} y={[cep.peak]} emphasis size={9} />
          <Annotation x={period} dashed text="true period" />
        </Plot>
        <Plot x={quef} y={yv}>
          <Curve name="d′(τ)" x={lags} y={dp} slot={2} />
          <Points name="YIN dip" x={[1000 * y.period]} y={[y.aperiodicity]} emphasis size={9} />
          <Annotation x={period} dashed text="true period" />
          <Handle {...state.handle('analysis.threshold', { axis: 'y', label: 'threshold' })} />
        </Plot>
      </Plots>
    </Figure>
  )
}

export function PitchTrackFigure() {
  const state = useFigureState({
    f0: slider(80, 300, 150, { label: 'mean f₀ (Hz)', step: 1 }),
    depth: slider(0, 0.3, 0.12, { label: 'vibrato depth (fraction of f₀)', step: 0.01 }),
    rate: slider(1, 8, 4, { label: 'vibrato rate (Hz)', step: 0.1 }),
    sigma: slider(0, 0.3, 0.05, { label: 'noise sd', step: 0.005 }),
  })
  const { f0, depth, rate, sigma } = state
  const n = 2 * FS
  const sound = useMemo(() => voicedSound(n, { fs: FS, f0, vibrato: { depth, rate } }), [f0, depth, rate, n])
  const x = useMemo(() => {
    const w = noise('pitch-track', n, sigma)
    return toFlat(sound.x).map((v, i) => v + w[i])
  }, [sound, sigma, n])
  const truth = useMemo(() => {
    const p = toFlat(sound.f0)
    const idx = range(200).map((i) => Math.floor((i * n) / 200))
    return { t: idx.map((i) => i / FS), f: idx.map((i) => p[i]) }
  }, [sound, n])
  const track = useMemo(() => {
    const r = yin(x, { fs: FS, fmin: 60, fmax: 500, frameLength: 400, hop: 160 })
    return { t: toFlat(r.t), f: toFlat(r.f0) }
  }, [x])
  const cepTrack = useMemo(() => {
    const L = 512
    const t: number[] = []
    const f: number[] = []
    for (let s = 0; s + L <= n; s += 160) {
      t.push((s + L / 2) / FS)
      f.push(cepstralPitch(x.slice(s, s + L), { fs: FS, fmin: 60, fmax: 500 }).f0)
    }
    return { t, f }
  }, [x, n])
  const err = (tr: { t: number[]; f: number[] }) => {
    const p = toFlat(sound.f0)
    const e = tr.t
      .map((tt, i) => Math.abs(tr.f[i] / p[Math.min(n - 1, Math.round(tt * FS))] - 1))
      .filter(Number.isFinite)
    return e.length ? (100 * e.reduce((a, b) => a + b, 0)) / e.length : NaN
  }
  const time = useAxis({ label: 'time (s)', range: [0, 2] })
  const freq = useAxis({ label: 'f₀ (Hz)', range: [40, 450] })
  return (
    <Figure
      title="Pitch tracks under vibrato"
      purpose="Frame by frame, YIN follows a moving f₀ to a fraction of a percent while the cepstral peak, read from longer frames at whole-sample quefrencies, is coarser and more prone to octave jumps."
      defaultSize="L"
      state={state}
      readouts={
        <>
          <Readout label="mean |error|, YIN" value={`${fmt(err(track), 2)} %`} />
          <Readout label="mean |error|, cepstrum" value={`${fmt(err(cepTrack), 2)} %`} />
        </>
      }
      caption="Two seconds of aifn-methods voicedSound with sinusoidal vibrato, at 8 kHz. YIN (aifn yin, 50 ms frames, 20 ms hop) and the cepstral peak (cepstralPitch on 64 ms frames) against the true f₀ (ink). Raise the noise: YIN marks frames unvoiced (gaps) before it errs; the cepstrum jumps to octaves."
    >
      <Plot x={time} y={freq}>
        <Curve name="true f₀" x={truth.t} y={truth.f} emphasis />
        <Points name="YIN" x={track.t} y={track.f} slot={0} size={5} />
        <Points name="cepstral peak" x={cepTrack.t} y={cepTrack.f} slot={1} size={5} shape={2} />
      </Plot>
    </Figure>
  )
}
