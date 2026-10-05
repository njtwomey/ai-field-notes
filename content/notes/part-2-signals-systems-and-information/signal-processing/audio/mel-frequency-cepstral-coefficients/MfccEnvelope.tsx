import { useMemo } from 'react'
import {
  Bars,
  choice,
  Curve,
  Figure,
  formatNumber,
  int,
  Plot,
  Points,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { lfilter } from 'aifn-compute/signal/filters'
import { getWindow } from 'aifn-compute/signal/windows'
import { dct, magnitudeSpectrum, melFilterBank } from '../_shared/audio'

const FS = 16000
const N = 512
const BANDS = 26

/** Average adult male formant frequencies (Hz) for three vowels, after Peterson and Barney (1952). */
const VOWELS = {
  a: [730, 1090, 2440],
  i: [270, 2290, 3010],
  u: [300, 870, 2240],
} as const
type Vowel = keyof typeof VOWELS

function vowelFrame(vowel: Vowel, f0: number): number[] {
  let a = [1]
  for (const f of VOWELS[vowel]) {
    const r = Math.exp((-Math.PI * 90) / FS)
    const th = (2 * Math.PI * f) / FS
    const s = [1, -2 * r * Math.cos(th), r * r]
    const next = new Array(a.length + 2).fill(0)
    a.forEach((ai, i) => s.forEach((sj, j) => (next[i + j] += ai * sj)))
    a = next
  }
  const period = FS / f0
  const excitation = Array.from({ length: 3000 }, (_, n) =>
    Math.floor(n / period) !== Math.floor((n - 1) / period) ? 1 : 0,
  )
  const voiced = toFlat(lfilter({ b: [1], a }, excitation).y as Tensor).slice(2000, 2000 + N)
  const w = toFlat(getWindow('hamming', N))
  return Array.from(voiced, (v, n) => v * w[n])
}

/** Inverse of the orthonormal DCT-II (the orthonormal DCT-III). */
function idct(c: number[], n: number): number[] {
  return Array.from({ length: n }, (_, i) =>
    c.reduce(
      (s, ck, k) => s + ck * Math.sqrt((k === 0 ? 1 : 2) / n) * Math.cos((Math.PI * k * (2 * i + 1)) / (2 * n)),
      0,
    ),
  )
}

/**
 * Log-mel energies of one voiced frame, their MFCCs (orthonormal DCT-II), and the log-mel curve rebuilt from only
 * the first C coefficients: the low-order cepstrum is a smooth description of the spectral envelope.
 */
export function MfccEnvelope() {
  const state = useFigureState({
    vowel: choice<Vowel>(
      [
        { value: 'a', label: '/a/' },
        { value: 'i', label: '/i/' },
        { value: 'u', label: '/u/' },
      ],
      'a',
      { label: 'vowel' },
    ),
    f0: int(120, { min: 80, max: 300, step: 5, label: 'fundamental f₀ (Hz)' }),
    keep: int(13, { min: 1, max: BANDS, step: 1, label: 'coefficients kept C' }),
  })

  const r = useMemo(() => {
    const frame = vowelFrame(state.vowel, state.f0)
    const mag = magnitudeSpectrum(frame, N)
    const bank = melFilterBank(BANDS, N, FS, 0, FS / 2, 'htk')
    const logMel = bank.filters.map((w) => Math.log(1e-10 + w.reduce((s, wk, k) => s + wk * mag[k] * mag[k], 0)))
    const mfcc = dct(logMel)
    const truncated = mfcc.map((c, k) => (k < state.keep ? c : 0))
    const rebuilt = idct(truncated, BANDS)
    const error = Math.sqrt(logMel.reduce((s, v, i) => s + (v - rebuilt[i]) ** 2, 0) / BANDS)
    return { logMel, mfcc, rebuilt, error, centres: bank.edges.slice(1, -1) }
  }, [state.vowel, state.f0, state.keep])

  const idx = r.centres.map((_, m) => m + 1)
  const bands = [
    { name: 'log-mel energies', x: idx, y: r.logMel, muted: true },
    { name: 'log-mel energies (points)', x: idx, y: r.logMel, slot: 0 },
    { name: `rebuilt from c₀…c${state.keep - 1}`, x: idx, y: r.rebuilt, slot: 1 },
  ] as const
  const coefficients = [
    {
      name: 'kept',
      x: r.mfcc.map((_, k) => k).slice(0, state.keep),
      y: r.mfcc.slice(0, state.keep),
      slot: 1,
    },
    {
      name: 'discarded',
      x: r.mfcc.map((_, k) => k).slice(state.keep),
      y: r.mfcc.slice(state.keep),
      muted: true,
    },
  ] as const

  const xAxis = useAxis({ label: 'mel band', hold: 'union' })
  const yAxis = useAxis({ label: 'log energy', hold: 'union' })
  const xAxis2 = useAxis({ label: 'cepstral index k', hold: 'union' })
  const yAxis2 = useAxis({ label: 'c_k', hold: 'union' })
  return (
    <Figure
      title="MFCCs as a compact envelope"
      state={state}
      caption="One frame of a synthetic vowel through 26 mel bands. Left: the log-mel energies, and the curve rebuilt from only the first C cepstral coefficients by the inverse DCT. Right: the coefficients themselves. A dozen coefficients reproduce the formant envelope; the discarded high-order coefficients hold the fine detail, including harmonic ripple at low bands. Different vowels change the low-order coefficients most; changing f₀ barely moves them."

      readouts={
        <>
          <Readout label="c₀, c₁, c₂" value={r.mfcc.slice(0, 3).map(formatNumber).join(', ')} />
          <Readout label="rms rebuild error (log units)" value={formatNumber(r.error)} />
        </>
      }
    >
      <div className="grid gap-4 lg:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={300}>
          <Curve {...bands[0]} />
          <Points {...bands[1]} />
          <Curve {...bands[2]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={300}>
          <Bars {...coefficients[0]} />
          <Bars {...coefficients[1]} />
        </Plot>
      </div>
    </Figure>
  )
}
