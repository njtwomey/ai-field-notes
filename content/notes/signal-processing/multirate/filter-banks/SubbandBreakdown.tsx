import { useMemo, useState } from 'react'
import {
  Heatmap,
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from '@/components/viz'
import { rng } from '@/lib/math'
import { kaiser } from '@/lib/dsp'

const FS = 8000
const N = 4000
const TIME = Array.from({ length: N }, (_, n) => n / FS)

/**
 * Half a second of distinct events: a 500 Hz tone, an upward chirp, a click, then two tones at once (at band centres, 1.5 and 3 kHz). Each should light
 * up different subbands at different times.
 */
function testSignal(): number[] {
  const g = rng(11)
  return TIME.map((t, n) => {
    let v = 0
    if (t < 0.15) v += Math.sin(2 * Math.PI * 500 * t)
    if (t >= 0.15 && t < 0.35) {
      // Linear chirp from 1000 to 3500 Hz over 0.2 s: phase 2π(f0 τ + (f1 − f0) τ² / (2T)).
      const tau = t - 0.15
      v += Math.sin(2 * Math.PI * (1000 * tau + (2500 * tau * tau) / (2 * 0.2)))
    }
    if (n >= 3040 && n < 3048) v += 4 * (n % 2 === 0 ? 1 : -1)
    if (t >= 0.4) v += 0.7 * Math.sin(2 * Math.PI * 1500 * t) + 0.7 * Math.sin(2 * Math.PI * 3000 * t)
    return v + 0.02 * g.normal()
  })
}

/**
 * Prototype lowpass: a Kaiser-windowed sinc with cutoff π/K, length 8K + 1, centred at D = 4K. Its taps at D ± rK are
 * zero for r ≠ 0 (sinc zeros) and 1/K at D, the Nyquist(K) property behind exact recombination.
 */
function prototype(k: number) {
  const length = 8 * k + 1
  const centre = 4 * k
  const w = kaiser(length, 8)
  const h = Array.from({ length }, (_, j) => {
    const u = (j - centre) / k
    return (u === 0 ? 1 : Math.sin(Math.PI * u) / (Math.PI * u)) / k
  })
  return { h: h.map((v, j) => v * w[j]), centre }
}

type Mode = 'magnitude' | 'real'

export function SubbandBreakdown() {
  const channels = useParam(16, { min: 4, max: 32, step: 4 })
  const band = useParam(2, { min: 0, max: 16, step: 1 })
  const hop = useParam(8, { min: 1, max: 32, step: 1 })
  const [mode, setMode] = useState<Mode>('magnitude')

  const r = useMemo(() => {
    const K = channels.value
    const x = testSignal()
    const { h, centre } = prototype(K)
    const L = h.length
    // Undecimated analysis: y_k[n] = Σ_j h[j] x[n − j] e^{i2πkj/K}. Only bands 0..K/2 are shown; the rest mirror them
    // for a real input.
    const half = K / 2
    const re: number[][] = Array.from({ length: half + 1 }, () => new Array(N).fill(0))
    const im: number[][] = Array.from({ length: half + 1 }, () => new Array(N).fill(0))
    const cos = Array.from({ length: half + 1 }, (_, k) => h.map((_, j) => Math.cos((2 * Math.PI * k * j) / K)))
    const sin = Array.from({ length: half + 1 }, (_, k) => h.map((_, j) => Math.sin((2 * Math.PI * k * j) / K)))
    for (let n = 0; n < N; n++) {
      for (let k = 0; k <= half; k++) {
        let a = 0
        let b = 0
        for (let j = 0; j < L && j <= n; j++) {
          const v = h[j] * x[n - j]
          a += v * cos[k][j]
          b += v * sin[k][j]
        }
        re[k][n] = a
        im[k][n] = b
      }
    }
    // Recombination: Σ over all K bands. Bands k and K − k are complex conjugates for real x, so the sum is
    // y_0 + 2 Σ_{0<k<K/2} Re y_k + y_{K/2}. By the Nyquist(K) property it equals x[n − D] exactly.
    const sum = Array.from({ length: N }, (_, n) => {
      let s = re[0][n] + re[half][n]
      for (let k = 1; k < half; k++) s += 2 * re[k][n]
      return s
    })
    let err = 0
    let ref = 0
    for (let n = centre; n < N; n++) {
      err = Math.max(err, Math.abs(sum[n] - x[n - centre]))
      ref = Math.max(ref, Math.abs(x[n - centre]))
    }
    return { x, re, im, sum, centre, K, half, relError: err / ref }
  }, [channels.value])

  const k = Math.min(band.value, r.half)
  const frames = Array.from({ length: Math.floor(N / hop.value) }, (_, m) => m * hop.value)
  const bands = Array.from({ length: r.half + 1 }, (_, b) => b)
  const bandHz = (b: number) => (b * FS) / r.K
  // Band × time magnitudes in dB, decimated by `hop` for display.
  const z = bands.map((b) => frames.map((n) => 20 * Math.log10(Math.max(Math.hypot(r.re[b][n], r.im[b][n]), 1e-4))))
  const trace: XYSeries[] = [
    {
      name: mode === 'magnitude' ? `|band ${k}|` : `Re band ${k}`,
      type: 'line',
      x: TIME,
      y: TIME.map((_, n) => (mode === 'magnitude' ? Math.hypot(r.re[k][n], r.im[k][n]) : r.re[k][n])),
      slot: 0,
    },
  ]
  const shift = r.centre
  const recombination: XYSeries[] = [
    { name: 'input (delayed)', type: 'line', x: TIME.slice(shift), y: r.x.slice(0, N - shift), slot: 0 },
    { name: 'sum of all bands', type: 'line', x: TIME, y: r.sum, slot: 1, dashed: true },
  ]
  const input: XYSeries[] = [{ name: 'input', type: 'line', x: TIME, y: r.x, slot: 0 }]

  return (
    <Interactive
      title="A signal broken into bands, and put back together"
      caption="The input holds four events in turn: a 500 Hz tone, a chirp rising from 1 to 3.5 kHz, a click, and two tones at 1.5 kHz and 3 kHz together. A K-channel DFT filter bank splits it into bands of width fs/K. The map shows each band's magnitude over time: the tone lights one band, the chirp climbs through the bands, the click fills every band for an instant, and the final pair lights two bands at once. The trace below follows one band. Summing all the bands reproduces the input, delayed by the prototype's centre, to rounding error."
      controls={
        <>
          <ParamSlider label="channels K" param={channels} format={(v) => String(v)} withArrows />
          <ParamSlider label={`band shown (0 to ${r.half})`} param={band} format={(v) => String(v)} withArrows />
          <ParamSlider label="display hop (samples)" param={hop} format={(v) => String(v)} />
          <ParamChoice
            label="band trace"
            value={mode}
            onChange={setMode}
            options={[
              { value: 'magnitude', label: 'magnitude' },
              { value: 'real', label: 'real part' },
            ]}
          />
        </>
      }
      readout={
        <>
          <Readout label="band width" value={`${formatNumber(FS / r.K)} Hz`} />
          <Readout label={`band ${k} centre`} value={`${formatNumber(bandHz(k))} Hz`} />
          <Readout label="prototype length" value={`${8 * r.K + 1} taps`} />
          <Readout label="delay" value={`${r.centre} samples`} />
          <Readout label="recombination error (relative)" value={r.relError.toExponential(1)} />
        </>
      }
    >
      <div className="space-y-4">
        <XYChart series={input} xLabel="time (s)" yLabel="x" height={160} />
        <Heatmap
          x={frames.map((n) => n / FS)}
          y={bands.map(bandHz)}
          z={z}
          range={[-40, 0]}
          xLabel="time (s)"
          yLabel="band centre (Hz)"
          valueLabel="magnitude (dB)"
          height={260}
        />
        <XYChart series={trace} xLabel="time (s)" yLabel={`band ${k}`} height={160} />
        <XYChart series={recombination} xLabel="time (s)" yLabel="x" height={160} />
      </div>
    </Interactive>
  )
}
