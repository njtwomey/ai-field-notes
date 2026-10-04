import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Raster,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { toFlat } from 'aifn/foundation/tensor'
import { getWindow } from 'aifn/signal'
import { normal, stream } from 'aifn/foundation/random'

const FS = 8000
const N = 4000
const TIME = Array.from({ length: N }, (_, n) => n / FS)

/**
 * Half a second of distinct events: a 500 Hz tone, an upward chirp, a click, then two tones at once (at band centres, 1.5 and 3 kHz). Each should light
 * up different subbands at different times.
 */
function testSignal(): number[] {
  const g = stream(11)
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
    return v + 0.02 * normal(g)
  })
}

/**
 * Prototype lowpass: a Kaiser-windowed sinc with cutoff π/K, length 8K + 1, centred at D = 4K. Its taps at D ± rK are
 * zero for r ≠ 0 (sinc zeros) and 1/K at D, the Nyquist(K) property behind exact recombination.
 */
function prototype(k: number) {
  const length = 8 * k + 1
  const centre = 4 * k
  const w = toFlat(getWindow({ name: 'kaiser', beta: 8 }, length))
  const h = Array.from({ length }, (_, j) => {
    const u = (j - centre) / k
    return (u === 0 ? 1 : Math.sin(Math.PI * u) / (Math.PI * u)) / k
  })
  return { h: h.map((v, j) => v * w[j]), centre }
}

type Mode = 'magnitude' | 'real'

export function SubbandBreakdown() {
  const state = useFigureState({
    channels: int(16, { min: 4, max: 32, step: 4, suggestions: [4, 8, 16, 32], label: 'channels K' }),
    band: slider(0, 16, 2, { step: 1, label: 'band shown (0 to K/2)', format: (v) => String(v) }),
    hop: int(8, { min: 1, max: 32, step: 1, label: 'display hop (samples)' }),
    mode: choice<Mode>(
      [
        { value: 'magnitude', label: 'magnitude' },
        { value: 'real', label: 'real part' },
      ],
      'magnitude',
      { label: 'band trace' },
    ),
  })
  const { mode } = state

  const r = useMemo(() => {
    const K = state.channels
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
  }, [state.channels])

  const k = Math.min(state.band, r.half)
  const frames = Array.from({ length: Math.floor(N / state.hop) }, (_, m) => m * state.hop)
  const bands = Array.from({ length: r.half + 1 }, (_, b) => b)
  const bandHz = (b: number) => (b * FS) / r.K
  // Band × time magnitudes in dB, decimated by `hop` for display.
  const z = bands.map((b) => frames.map((n) => 20 * Math.log10(Math.max(Math.hypot(r.re[b][n], r.im[b][n]), 1e-4))))
  const trace = [
    {
      name: mode === 'magnitude' ? `|band ${k}|` : `Re band ${k}`,
      x: TIME,
      y: TIME.map((_, n) => (mode === 'magnitude' ? Math.hypot(r.re[k][n], r.im[k][n]) : r.re[k][n])),
      slot: 0,
    },
  ] as const
  const shift = r.centre
  const recombination = [
    { name: 'input (delayed)', x: TIME.slice(shift), y: r.x.slice(0, N - shift), slot: 0 },
    { name: 'sum of all bands', x: TIME, y: r.sum, slot: 1, dashed: true },
  ] as const
  const input = [{ name: 'input', x: TIME, y: r.x, slot: 0 }] as const

  const xAxis = useAxis({ label: 'time (s)', hold: 'union' })
  const yAxis = useAxis({ label: 'x', hold: 'union' })
  const xAxis2 = useAxis({ label: 'time (s)' })
  const yAxis2 = useAxis({ label: 'band centre (Hz)' })
  const xAxis3 = useAxis({ label: 'time (s)', hold: 'union' })
  const yAxis3 = useAxis({ label: `band ${k}`, hold: 'union' })
  const xAxis4 = useAxis({ label: 'time (s)', hold: 'union' })
  const yAxis4 = useAxis({ label: 'x', hold: 'union' })
  return (
    <Figure
      title="A signal broken into bands, and put back together"
      caption="The input holds four events in turn: a 500 Hz tone, a chirp rising from 1 to 3.5 kHz, a click, and two tones at 1.5 kHz and 3 kHz together. A K-channel DFT filter bank splits it into bands of width fs/K. The map shows each band's magnitude over time: the tone lights one band, the chirp climbs through the bands, the click fills every band for an instant, and the final pair lights two bands at once. The trace below follows one band; drag the line on the map to choose it. Summing all the bands reproduces the input, delayed by the prototype's centre, to rounding error."
      state={state}
      readouts={
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
        <Plot x={xAxis} y={yAxis} height={160}>
          <Curve {...input[0]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={260}>
          <Raster
            x={frames.map((n) => n / FS)}
            y={bands.map(bandHz)}
            z={z}
            range={[-40, 0]}
            valueLabel={'magnitude (dB)'}
          />
          <Handle
            kind="y"
            at={bandHz(k)}
            label={`band ${k}`}
            onDrag={(hz) => {
              const near = bands.map((b) => Math.abs(bandHz(b) - hz))
              state.set('band', near.indexOf(Math.min(...near)))
            }}
          />
        </Plot>
        <Plot x={xAxis3} y={yAxis3} height={160}>
          <Curve {...trace[0]} />
        </Plot>
        <Plot x={xAxis4} y={yAxis4} height={160}>
          <Curve {...recombination[0]} />
          <Curve {...recombination[1]} />
        </Plot>
      </div>
    </Figure>
  )
}
