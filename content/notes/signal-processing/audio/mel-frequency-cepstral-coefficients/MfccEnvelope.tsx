import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from '@/components/viz'
import { lfilter, magnitudeSpectrum, makeWindow } from '@/lib/dsp'
import { dct, melFilterBank } from '../_shared/audio'

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
  const voiced = lfilter([1], a, excitation).slice(2000, 2000 + N)
  const w = makeWindow('hamming', N)
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
  const [vowel, setVowel] = useState<Vowel>('a')
  const f0 = useParam(120, { min: 80, max: 300, step: 5 })
  const keep = useParam(13, { min: 1, max: BANDS, step: 1 })

  const r = useMemo(() => {
    const frame = vowelFrame(vowel, f0.value)
    const mag = magnitudeSpectrum(frame, N)
    const bank = melFilterBank(BANDS, N, FS, 0, FS / 2, 'htk')
    const logMel = bank.filters.map((w) => Math.log(1e-10 + w.reduce((s, wk, k) => s + wk * mag[k] * mag[k], 0)))
    const mfcc = dct(logMel)
    const truncated = mfcc.map((c, k) => (k < keep.value ? c : 0))
    const rebuilt = idct(truncated, BANDS)
    const error = Math.sqrt(logMel.reduce((s, v, i) => s + (v - rebuilt[i]) ** 2, 0) / BANDS)
    return { logMel, mfcc, rebuilt, error, centres: bank.edges.slice(1, -1) }
  }, [vowel, f0.value, keep.value])

  const idx = r.centres.map((_, m) => m + 1)
  const bands: XYSeries[] = [
    { name: 'log-mel energies', type: 'line', x: idx, y: r.logMel, muted: true },
    { name: 'log-mel energies (points)', type: 'scatter', x: idx, y: r.logMel, slot: 0 },
    { name: `rebuilt from c₀…c${keep.value - 1}`, type: 'line', x: idx, y: r.rebuilt, slot: 1 },
  ]
  const coefficients: XYSeries[] = [
    {
      name: 'kept',
      type: 'bar',
      x: r.mfcc.map((_, k) => k).slice(0, keep.value),
      y: r.mfcc.slice(0, keep.value),
      slot: 1,
    },
    {
      name: 'discarded',
      type: 'bar',
      x: r.mfcc.map((_, k) => k).slice(keep.value),
      y: r.mfcc.slice(keep.value),
      muted: true,
    },
  ]

  return (
    <Interactive
      title="MFCCs as a compact envelope"
      caption="One frame of a synthetic vowel through 26 mel bands. Left: the log-mel energies, and the curve rebuilt from only the first C cepstral coefficients by the inverse DCT. Right: the coefficients themselves. A dozen coefficients reproduce the formant envelope; the discarded high-order coefficients hold the fine detail, including harmonic ripple at low bands. Different vowels change the low-order coefficients most; changing f₀ barely moves them."
      controls={
        <>
          <ParamChoice
            label="vowel"
            value={vowel}
            onChange={setVowel}
            options={[
              { value: 'a', label: '/a/' },
              { value: 'i', label: '/i/' },
              { value: 'u', label: '/u/' },
            ]}
          />
          <ParamSlider label="fundamental f₀ (Hz)" param={f0} />
          <ParamSlider label="coefficients kept C" param={keep} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="c₀, c₁, c₂" value={r.mfcc.slice(0, 3).map(formatNumber).join(', ')} />
          <Readout label="rms rebuild error (log units)" value={formatNumber(r.error)} />
        </>
      }
    >
      <div className="grid gap-4 lg:grid-cols-2">
        <XYChart series={bands} xLabel="mel band" yLabel="log energy" height={300} />
        <XYChart series={coefficients} xLabel="cepstral index k" yLabel="c_k" height={300} />
      </div>
    </Interactive>
  )
}
