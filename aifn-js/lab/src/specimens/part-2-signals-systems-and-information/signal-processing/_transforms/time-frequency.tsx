/**
 * "Time–frequency: STFT, wavelets, Wigner–Ville": one test signal (a chirp, two tones and a click) under the STFT with
 * its window-length trade-off, the Morlet CWT, the Wigner–Ville distribution and its smoothed forms, the reassigned
 * spectrogram and the synchrosqueezed STFT; and the Wigner–Ville cross-term between two draggable atoms.
 */
import { useMemo } from 'react'
import { toFlat, type Tensor } from 'aifn/foundation/tensor'
import { spectrogram } from 'aifn/signal/spectral'
import {
  pseudoWignerVille,
  reassignedSpectrogram,
  smoothedPseudoWignerVille,
  synchrosqueeze,
  wignerVille,
} from 'aifn/signal/time-frequency'
import { cwt } from 'aifn/signal/wavelets'
import { Figure } from '@lab/layout'
import { choice, row, slider, useFigureState } from '@lab/state'
import { Curve, Handle, Plot, Plots, Raster, Readout, useAxis } from '@lab/viz'
import { fmt, range, rows } from './common'

const FS = 512
const N = 512
const TAU = 2 * Math.PI

/** The test signal: a linear chirp 30 → 220 Hz, a 60 Hz tone (first half), a 160 Hz tone (from 0.35 s), a click at 0.75 s. */
function testSignal(): number[] {
  return range(N).map((i) => {
    const t = i / FS
    const chirp = Math.cos(TAU * (30 * t + 95 * t * t))
    const low = t < 0.5 ? 0.7 * Math.cos(TAU * 60 * t) : 0
    const high = t > 0.35 ? 0.6 * Math.cos(TAU * 160 * t + 0.3) : 0
    const click = i === Math.round(0.75 * FS) ? 4 : 0
    return chirp + low + high + click
  })
}

/** A raster in dB clipped `range` dB below its peak, for display. */
function decibelRows(values: Tensor, rangeDb = 50, power = true): number[][] {
  const r = rows(values)
  const k = power ? 10 : 20
  const peak = Math.max(...r.flat().map(Math.abs))
  return r.map((row) => row.map((v) => Math.max(k * Math.log10(Math.max(Math.abs(v), 1e-300) / peak), -rangeDb)))
}

const TRUTH = {
  chirp: { t: [0, 1], f: [30, 220] },
}

// ── 1. The STFT's window-length trade-off ─────────────────────────────────────────────────────────────────────────

export function StftTradeOffFigure() {
  const state = useFigureState({
    nperseg: choice([16, 32, 64, 128, 256], 64, { label: 'window length (samples)' }),
  })
  const L = state.nperseg
  const x = useMemo(() => testSignal(), [])
  const sg = useMemo(
    () => spectrogram(x, { fs: FS, nperseg: L, noverlap: L - Math.max(1, L / 16), window: 'hann' }),
    [x, L],
  )
  const view = useMemo(() => ({ t: toFlat(sg.t), f: toFlat(sg.f), z: decibelRows(sg.values) }), [sg])
  const time = useAxis({ label: 'time (s)', range: [0, 1] })
  const freq = useAxis({ label: 'frequency (Hz)', range: [0, FS / 2] })
  return (
    <Figure
      title="The STFT's window-length trade-off"
      purpose="A short window locates the click in time but smears every tone across frequency; a long one resolves the tones and blurs the click and the sweep. Δt · Δf stays fixed: the uncertainty principle."
      defaultSize="L"
      state={state}
      readouts={
        <>
          <Readout label="window Δt = L/fs" value={`${fmt((1000 * L) / FS, 1)} ms`} />
          <Readout label="bin spacing Δf = fs/L" value={`${fmt(FS / L, 1)} Hz`} />
          <Readout label="Δt · Δf" value="1" />
        </>
      }
      caption="A linear chirp from 30 to 220 Hz, a 60 Hz tone in the first half, a 160 Hz tone from 0.35 s and a click at 0.75 s, at fs = 512 Hz. Spectrogram (aifn spectrogram, Hann window, 1/16 hop) in dB, clipped 50 dB below its peak. Step through the window lengths: at 16 samples the click is a sharp vertical line and the tones are wide bands; at 256 the tones are thin and the click spreads over half a second."
    >
      <Plot x={time} y={freq}>
        <Raster x={view.t} y={view.f} z={view.z} valueLabel="power (dB)" />
        <Curve name="true sweep" x={TRUTH.chirp.t} y={TRUTH.chirp.f} emphasis dashed />
      </Plot>
    </Figure>
  )
}

// ── 2. Six views of one signal ────────────────────────────────────────────────────────────────────────────────────

const CWT_FREQS = range(60).map((i) => 10 + i * 4.1)

export function TimeFrequencyViewsFigure() {
  const state = useFigureState({
    stft: row('1 · STFT family', {
      nperseg: choice([32, 64, 128], 64, { label: 'window length (samples)' }),
    }),
    wvd: row('2 · Wigner–Ville smoothing', {
      lag: choice([15, 31, 63, 127], 63, { label: 'lag window h (samples)' }),
      time: choice([3, 7, 15, 31], 15, { label: 'time window g (samples)' }),
    }),
    cwt: row('3 · wavelet', { omega0: slider(4, 16, 8, { label: 'Morlet ω₀', step: 0.5 }) }),
  })
  const x = useMemo(() => testSignal(), [])
  const L = state.stft.nperseg
  const { lag, time: tw } = state.wvd
  const views = useMemo(() => {
    const sg = spectrogram(x, { fs: FS, nperseg: L, noverlap: L - L / 8, window: 'hann' })
    const opts = { fs: FS, nperseg: L, hop: 4, nfft: 2 * L }
    const re = reassignedSpectrogram(x, opts)
    const sq = synchrosqueeze(x, opts)
    return {
      stft: { t: toFlat(sg.t), f: toFlat(sg.f), z: decibelRows(sg.values) },
      re: { t: toFlat(re.t), f: toFlat(re.f), z: decibelRows(re.values) },
      sq: { t: toFlat(sq.t), f: toFlat(sq.f), z: decibelRows(sq.values, 50, false) },
    }
  }, [x, L])
  const wv = useMemo(() => {
    const plain = wignerVille(x, { fs: FS, nfft: 256, hop: 2 })
    const smooth = smoothedPseudoWignerVille(x, { fs: FS, nfft: 256, hop: 2, lagLength: lag, timeLength: tw })
    const scale = (v: Tensor) => {
      const r = rows(v)
      // Normalised to a third of the peak and clipped, so the weaker components and their cross-terms show.
      const peak = Math.max(...r.flat().map(Math.abs)) / 3
      return r.map((row) => row.map((u) => Math.max(-1, Math.min(1, u / peak))))
    }
    return {
      t: toFlat(plain.t),
      f: toFlat(plain.f),
      plain: scale(plain.values),
      smooth: scale(smooth.values),
    }
  }, [x, lag, tw])
  const omega0 = state.cwt.omega0
  const scal = useMemo(() => {
    const c = cwt(x, CWT_FREQS, { fs: FS, omega0 })
    return { t: toFlat(c.t), z: decibelRows(c.magnitude, 40, false) }
  }, [x, omega0])
  const time = useAxis({ label: 'time (s)', range: [0, 1] })
  const freq = useAxis({ label: 'frequency (Hz)', range: [0, FS / 2] })
  return (
    <Figure
      title="One signal, six time–frequency views"
      purpose="Each representation spends the same uncertainty differently: the STFT evenly, the wavelet transform by scale, the Wigner–Ville distribution not at all (perfect lines, plus interference between every pair of components), and reassignment and synchrosqueezing by moving energy onto the instantaneous frequency."
      defaultSize="XL"
      state={state}
      caption="The signal of the figure above. Top: spectrogram and Morlet CWT (dB, |W|, aifn cwt). Middle: the Wigner–Ville distribution of the analytic signal and its smoothed pseudo form (aifn wignerVille, smoothedPseudoWignerVille), signed, normalised to a third of their peak and clipped: the plain one draws the chirp as a thin line but fills the plane between components with oscillating cross-terms; widen the windows to wash them out. Bottom: reassigned spectrogram and synchrosqueezed STFT (aifn reassignedSpectrogram, synchrosqueeze), sharp lines from the same windows as the spectrogram."
    >
      <Plots rows={3} cols={2}>
        <Plot x={time} y={freq} title="STFT (spectrogram)">
          <Raster x={views.stft.t} y={views.stft.f} z={views.stft.z} valueLabel="dB" colorBar={false} />
        </Plot>
        <Plot x={time} y={freq} title="Morlet CWT">
          <Raster x={scal.t} y={CWT_FREQS} z={scal.z} valueLabel="dB" colorBar={false} />
        </Plot>
        <Plot x={time} y={freq} title="Wigner–Ville">
          <Raster x={wv.t} y={wv.f} z={wv.plain} scale="diverging" valueLabel="W / max" colorBar={false} />
        </Plot>
        <Plot x={time} y={freq} title="smoothed pseudo Wigner–Ville">
          <Raster x={wv.t} y={wv.f} z={wv.smooth} scale="diverging" valueLabel="W / max" colorBar={false} />
        </Plot>
        <Plot x={time} y={freq} title="reassigned spectrogram">
          <Raster x={views.re.t} y={views.re.f} z={views.re.z} valueLabel="dB" colorBar={false} />
        </Plot>
        <Plot x={time} y={freq} title="synchrosqueezed STFT">
          <Raster x={views.sq.t} y={views.sq.f} z={views.sq.z} valueLabel="dB" colorBar={false} />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── 3. Cross-terms between two atoms ──────────────────────────────────────────────────────────────────────────────

export function CrossTermFigure() {
  const state = useFigureState({
    a: row('atom 1', {
      t: slider(0.1, 0.9, 0.3, { label: 'time (s)', step: 0.01 }),
      f: slider(10, 120, 40, { label: 'frequency (Hz)', step: 1 }),
    }),
    b: row('atom 2', {
      t: slider(0.1, 0.9, 0.7, { label: 'time (s)', step: 0.01 }),
      f: slider(10, 120, 90, { label: 'frequency (Hz)', step: 1 }),
    }),
    method: choice(
      [
        { value: 'wvd', label: 'Wigner–Ville' },
        { value: 'pseudo', label: 'pseudo (lag window 31)' },
        { value: 'smoothed', label: 'smoothed pseudo (31, 15)' },
      ],
      'wvd',
      { label: 'distribution' },
    ),
  })
  const fs = 256
  const n = 256
  const atoms = [state.a, state.b]
  const key = JSON.stringify(atoms)
  const result = useMemo(() => {
    const as = JSON.parse(key) as { t: number; f: number }[]
    const x = range(n).map((i) => {
      const t = i / fs
      return as.reduce((s, a) => s + Math.exp(-0.5 * ((t - a.t) / 0.05) ** 2) * Math.cos(TAU * a.f * t), 0)
    })
    const opts = { fs, nfft: 128 }
    const w =
      state.method === 'wvd'
        ? wignerVille(x, opts)
        : state.method === 'pseudo'
          ? pseudoWignerVille(x, { ...opts, lagLength: 31 })
          : smoothedPseudoWignerVille(x, { ...opts, lagLength: 31, timeLength: 15 })
    const r = rows(w.values)
    const peak = Math.max(...r.flat().map(Math.abs))
    return { t: toFlat(w.t), f: toFlat(w.f), z: r.map((row) => row.map((v) => v / peak)) }
  }, [key, state.method])
  const mid = { t: (state.a.t + state.b.t) / 2, f: (state.a.f + state.b.f) / 2 }
  const time = useAxis({ label: 'time (s)', range: [0, 1] })
  const freq = useAxis({ label: 'frequency (Hz)', range: [0, 128] })
  return (
    <Figure
      title="Wigner–Ville cross-terms"
      purpose="The Wigner–Ville distribution is quadratic, so two components produce a third term at their midpoint, oscillating across the line joining them faster the farther apart they are; smoothing in lag and time averages it away and blurs the atoms."
      defaultSize="L"
      state={state}
      readouts={
        <>
          <Readout label="cross-term at" value={`${fmt(mid.t, 2)} s, ${fmt(mid.f, 0)} Hz`} />
          <Readout
            label="separation"
            value={`${fmt(Math.abs(state.a.t - state.b.t), 2)} s, ${fmt(Math.abs(state.a.f - state.b.f), 0)} Hz`}
          />
        </>
      }
      caption="Two Gaussian atoms (width 50 ms) at fs = 256 Hz; drag either atom on the plane. Red and blue are positive and negative values. The interference term sits exactly half way between them and is as strong as the atoms; it oscillates perpendicular to the line joining them. Switch to the smoothed pseudo distribution: it vanishes, and the atoms grow."
    >
      <Plot x={time} y={freq}>
        <Raster x={result.t} y={result.f} z={result.z} scale="diverging" valueLabel="W / max" />
        <Handle {...state.handle(['a.t', 'a.f'], { label: '1' })} />
        <Handle {...state.handle(['b.t', 'b.f'], { label: '2' })} />
      </Plot>
    </Figure>
  )
}
