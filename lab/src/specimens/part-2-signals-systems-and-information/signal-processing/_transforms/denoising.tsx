/**
 * "Denoising: Wiener, Savitzky–Golay, median, wavelet thresholding": the same noisy test function cleaned by each
 * method, scored against the known clean signal.
 */
import { useMemo } from 'react'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { testFunction } from 'aifn-methods/data/signals'
import { matchedFilter, medfilt, savgolFilter, wiener, wienerDenoise } from 'aifn-compute/signal/filters'
import { waveletDenoise } from 'aifn-compute/signal/wavelets'
import { normalQuantile } from 'aifn-compute/numerics/special'
import { Figure } from 'aifn-render/layout'
import { choice, row, slider, useFigureState } from 'aifn-render/state'
import { Annotation, Curve, Handle, Plot, Plots, Points, Readout, useAxis } from 'aifn-render/viz'
import { fmt, noise, range, snrDb } from './common'

const N = 1024
const SIGNALS = [
  { value: 'doppler', label: 'Doppler' },
  { value: 'blocks', label: 'blocks' },
  { value: 'bumps', label: 'bumps' },
  { value: 'heavisine', label: 'HeaviSine' },
] as const

const METHODS = ['spectral Wiener', 'local Wiener', 'Savitzky–Golay', 'median', 'wavelet shrinkage'] as const

export function DenoisingFigure() {
  const state = useFigureState({
    data: row('1 · data', {
      signal: choice(SIGNALS, 'doppler', { label: 'clean signal (Donoho–Johnstone)' }),
      sigma: slider(0.05, 1, 0.3, { label: 'Gaussian noise sd σ', step: 0.01 }),
      impulses: slider(0, 0.1, 0, { label: 'impulse fraction (±4)', step: 0.005 }),
    }),
    methods: row('2 · methods', {
      sgHalf: slider(2, 30, 10, { label: 'Savitzky–Golay half-width h (window 2h + 1)', step: 1 }),
      order: slider(1, 6, 3, { label: 'Savitzky–Golay order', step: 1 }),
      half: slider(1, 15, 4, { label: 'median / local Wiener half-width (window 2h + 1)', step: 1 }),
      wavelet: choice(['haar', 'db2', 'db4', 'db8'], 'db4', { label: 'wavelet' }),
      mode: choice(['soft', 'hard'], 'soft', { label: 'threshold' }),
    }),
  })
  const { signal: kind, sigma, impulses } = state.data
  const { wavelet, mode } = state.methods
  const window = 2 * state.methods.sgHalf + 1
  const kernel = 2 * state.methods.half + 1
  const order = Math.min(state.methods.order, window - 1)
  const clean = useMemo(() => toFlat(testFunction(kind, N)), [kind])
  const noisy = useMemo(() => {
    const w = noise('denoising', N, 1)
    const u = noise('denoising-impulses', N, 1)
    return clean.map((v, i) => v + sigma * w[i] + (Math.abs(u[i]) > quantile(impulses) ? 4 * Math.sign(u[i]) : 0))
  }, [clean, sigma, impulses])
  const out = useMemo(
    () => [
      toFlat(wienerDenoise(noisy, sigma * sigma, { smoothing: 9 }).signal.data),
      toFlat(wiener(noisy, kernel).data),
      toFlat(savgolFilter(noisy, window, order).data),
      toFlat(medfilt(noisy, kernel, { padding: 'nearest' }).data),
      toFlat(waveletDenoise(noisy, { wavelet: wavelet as 'db4', mode: mode as 'soft' }).signal.data),
    ],
    [noisy, sigma, kernel, window, order, wavelet, mode],
  )
  const snrIn = snrDb(clean, noisy)
  const snr = out.map((y) => snrDb(clean, y))
  const t = useMemo(() => range(N).map((i) => i / N), [])
  const x = useAxis({ label: 't', range: [0, 1] })
  const y = useAxis({ label: 'value', hold: 'initial', key: kind })
  return (
    <Figure
      title="Five denoisers on one known signal"
      purpose="Linear smoothers (Wiener, Savitzky–Golay) trade noise for blur at jumps; the median keeps jumps and removes impulses; wavelet shrinkage keeps the few large coefficients that carry both jumps and smooth parts, so it adapts to the signal."
      defaultSize="XL"
      state={state}
      readouts={{
        'SNR (dB)': (
          <>
            <Readout label="input" value={fmt(snrIn, 1)} />
            {METHODS.map((name, i) => (
              <Readout key={name} label={name} value={fmt(snr[i], 1)} />
            ))}
          </>
        ),
      }}
      caption="Each Donoho–Johnstone test function (aifn-methods testFunction, unit variance, n = 1024) plus Gaussian noise of sd σ and optional ±4 impulses. Spectral Wiener: aifn wienerDenoise with the true σ²; local Wiener: wiener (scipy's); Savitzky–Golay: savgolFilter; median: medfilt; wavelet: waveletDenoise with the universal threshold. Add impulses: every linear method smears them; the median removes them. On blocks, Savitzky–Golay rounds the steps that the Haar wavelet keeps."
    >
      <Plots rows={3} cols={2} widths={[1, 1]}>
        <Plot x={x} y={y} title="noisy input">
          <Curve name="noisy" x={t} y={noisy} muted thin />
          <Curve name="clean" x={t} y={clean} emphasis />
        </Plot>
        {out.map((o, i) => (
          <Plot key={METHODS[i]} x={x} y={y} title={`${METHODS[i]}: ${fmt(snr[i], 1)} dB`}>
            <Curve name="clean" x={t} y={clean} muted />
            <Curve name={METHODS[i]} x={t} y={o} slot={i} />
          </Plot>
        ))}
      </Plots>
    </Figure>
  )
}

/** |z| above this standard-normal quantile happens with probability p (two-sided), so a fraction p of samples is hit. */
const quantile = (p: number): number => (p <= 0 ? Infinity : (normalQuantile(1 - p / 2) as number))

// ── Detection: the matched filter ─────────────────────────────────────────────────────────────────────────────────

const TEMPLATE = range(40).map((i) => Math.sin((2 * Math.PI * 3 * i) / 40) * Math.sin((Math.PI * i) / 40))
const ARRIVALS = [150, 420, 700]

export function MatchedFilterFigure() {
  const state = useFigureState({
    sigma: slider(0.2, 3, 1.2, { label: 'noise sd σ', step: 0.05 }),
    threshold: slider(0, 8, 4, { label: 'detection threshold', step: 0.1, onChart: true }),
  })
  const { sigma, threshold } = state
  const n = 900
  const x = useMemo(() => {
    const w = noise('matched', n, sigma)
    const v = [...w]
    for (const a of ARRIVALS) TEMPLATE.forEach((u, k) => (v[a + k] += u))
    return v
  }, [sigma])
  const m = useMemo(() => matchedFilter(x, TEMPLATE), [x])
  const y = useMemo(() => toFlat(m.output.data).map((v) => v / sigma), [m, sigma])
  const peaks = range(n).filter(
    (i) => y[i] > threshold && (y[i - 1] ?? -Infinity) <= y[i] && y[i] > (y[i + 1] ?? -Infinity),
  )
  const hits = peaks.filter((p) => ARRIVALS.some((a) => Math.abs(p - a) <= 3))
  const energy = TEMPLATE.reduce((a, b) => a + b * b, 0)
  const t = useAxis({ label: 'sample n', range: [0, n] })
  const v = useAxis({ label: 'x[n]', hold: 'union' })
  const z = useAxis({ label: 'output / σ', range: [-5, 9] })
  return (
    <Figure
      title="The matched filter finds a known pulse in noise"
      purpose="Correlating with the template is the linear filter with the largest output SNR in white noise: the peak height is √E/σ, E the template energy, however the noise looks sample by sample."
      defaultSize="L"
      state={state}
      readouts={
        <>
          <Readout label="expected peak √E/σ" value={fmt(Math.sqrt(energy) / sigma, 2)} />
          <Readout label="detections" value={peaks.length} />
          <Readout label="true arrivals found" value={`${hits.length} of ${ARRIVALS.length}`} />
          <Readout label="false alarms" value={peaks.length - hits.length} />
        </>
      }
      caption="A 40-sample windowed sinusoid (E ≈ 10) arrives at samples 150, 420 and 700 in white noise. The lower panel is aifn matchedFilter's output scaled by 1/σ, so the noise has unit sd. Drag the threshold line: lower catches more pulses and more false alarms; at σ = 1.2 the pulses are invisible in the input but stand at about 2.6 sd in the output."
    >
      <Plots rows={2}>
        <Plot x={t} y={v}>
          <Curve name="x[n]" x={range(n)} y={x} muted thin />
          {ARRIVALS.map((a) => (
            <Annotation key={a} x={a} dashed />
          ))}
        </Plot>
        <Plot x={t} y={z}>
          <Curve name="matched-filter output" x={range(n)} y={y} slot={0} />
          <Points name="detections" x={peaks} y={peaks.map((p) => y[p])} emphasis size={7} />
          <Handle {...state.handle('threshold', { axis: 'y', label: 'threshold' })} />
        </Plot>
      </Plots>
    </Figure>
  )
}
