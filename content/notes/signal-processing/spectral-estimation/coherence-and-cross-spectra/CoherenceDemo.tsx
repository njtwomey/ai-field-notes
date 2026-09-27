import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamSlider,
  ParamSwitch,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from '@/components/viz'
import { fft, makeWindow } from '@/lib/dsp'
import { whiteNoise } from '../_shared/spectra'

const N = 8192
const SEG = 128

/**
 * Welch estimates of the cross-spectrum and coherence between white noise x and y = x delayed by d samples plus
 * independent noise. Coherence is 1/(1 + noise variance) at every frequency; the cross-spectrum's phase is −ωd.
 */
export function CoherenceDemo() {
  const delay = useParam(5, { min: 0, max: 20, step: 1 })
  const noise = useParam(0.5, { min: 0, max: 3, step: 0.1 })
  const overlap = useParam(0.5, { min: 0, max: 0.75, step: 0.25 })
  const [single, setSingle] = useState(false)

  const r = useMemo(() => {
    const x = whiteNoise(N + 32, 11)
    const e = whiteNoise(N + 32, 12)
    const y = x.map((_, n) => (n >= delay.value ? x[n - delay.value] : 0) + noise.value * e[n])
    const w = makeWindow('hann', SEG, true)
    const half = SEG / 2 + 1
    const sxx = new Array<number>(half).fill(0)
    const syy = new Array<number>(half).fill(0)
    const sxyRe = new Array<number>(half).fill(0)
    const sxyIm = new Array<number>(half).fill(0)
    const step = Math.max(1, Math.round(SEG * (1 - overlap.value)))
    // With "one segment", only the first segment is used: coherence is then identically 1.
    const last = single ? 0 : N - SEG
    let count = 0
    for (let s = 32; s <= 32 + last; s += step) {
      const X = fft(
        x.slice(s, s + SEG).map((v, i) => v * w[i]),
        SEG,
      )
      const Y = fft(
        y.slice(s, s + SEG).map((v, i) => v * w[i]),
        SEG,
      )
      for (let k = 0; k < half; k++) {
        sxx[k] += X.re[k] ** 2 + X.im[k] ** 2
        syy[k] += Y.re[k] ** 2 + Y.im[k] ** 2
        // S_xy = conj(X) · Y, so that y = h ∗ x gives S_xy = H S_xx.
        sxyRe[k] += X.re[k] * Y.re[k] + X.im[k] * Y.im[k]
        sxyIm[k] += X.re[k] * Y.im[k] - X.im[k] * Y.re[k]
      }
      count++
    }
    const omega = Array.from({ length: half }, (_, k) => (2 * k) / SEG)
    const coherence = omega.map((_, k) => (sxyRe[k] ** 2 + sxyIm[k] ** 2) / (sxx[k] * syy[k]))
    const phase = omega.map((_, k) => Math.atan2(sxyIm[k], sxyRe[k]))
    const mean = coherence.slice(1, -1).reduce((s, v) => s + v, 0) / (half - 2)
    return { omega, coherence, phase, mean, count }
  }, [delay.value, noise.value, overlap.value, single])

  const theory = 1 / (1 + noise.value ** 2)
  const wrap = (p: number) => Math.atan2(Math.sin(p), Math.cos(p))
  const coh: XYSeries[] = [
    { name: 'estimated coherence', type: 'line', x: r.omega, y: r.coherence, slot: 0 },
    { name: 'theory 1/(1 + σ²)', type: 'line', x: [0, 1], y: [theory, theory], slot: 1, dashed: true },
  ]
  const ph: XYSeries[] = [
    { name: 'estimated phase of S_xy', type: 'scatter', x: r.omega, y: r.phase, slot: 0 },
    {
      name: 'theory −ωd (wrapped)',
      type: 'line',
      x: r.omega,
      y: r.omega.map((w) => wrap(-w * Math.PI * delay.value)),
      slot: 1,
      dashed: true,
    },
  ]

  return (
    <Interactive
      title="Coherence and cross-phase"
      caption="White noise x and y = x delayed by d samples plus independent noise of standard deviation σ. Welch-averaged cross-spectra give a coherence near 1/(1 + σ²) at every frequency, the fraction of y's power linearly explained by x, and a cross-spectrum phase that falls linearly with slope −d, the delay. Estimate from one segment and the coherence is exactly 1 everywhere, whatever the noise: coherence needs averaging."
      controls={
        <>
          <ParamSlider label="delay d (samples)" param={delay} format={(v) => String(v)} withArrows />
          <ParamSlider label="noise standard deviation σ" param={noise} />
          <ParamSlider label="segment overlap" param={overlap} format={(v) => `${Math.round(v * 100)}%`} />
          <ParamSwitch label="estimate from a single segment" checked={single} onChange={setSingle} />
        </>
      }
      readout={
        <>
          <Readout label="segments averaged" value={r.count} />
          <Readout label="mean estimated coherence" value={formatNumber(r.mean)} />
          <Readout label="theory" value={formatNumber(theory)} />
        </>
      }
    >
      <div className="space-y-4">
        <XYChart series={coh} xLabel="ω / π" yLabel="coherence" xRange={[0, 1]} yRange={[0, 1.05]} height={240} />
        <XYChart series={ph} xLabel="ω / π" yLabel="phase (rad)" xRange={[0, 1]} yRange={[-3.3, 3.3]} height={240} />
      </div>
    </Interactive>
  )
}
