import { useMemo } from 'react'
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
import { harmonicTone } from '../_shared/audio'

const FS = 16000
const W = 512
const MAX_LAG = 400 // down to 40 Hz
const MIN_LAG = 32 // up to 500 Hz

/**
 * YIN's cumulative mean normalised difference d′(τ) for one frame of a harmonic tone, with the plain autocorrelation
 * for comparison. YIN takes the first dip below the threshold and refines it by parabolic interpolation.
 */
export function YinPitch() {
  const f0 = useParam(220, { min: 60, max: 450, step: 1 })
  const noise = useParam(0.1, { min: 0, max: 1.5, step: 0.05 })
  const threshold = useParam(0.1, { min: 0.02, max: 0.5, step: 0.01 })
  const drop = useParam(0, { min: 0, max: 1, step: 1 })

  const r = useMemo(() => {
    // Harmonics 1..10 at amplitude 1/h, optionally without the fundamental, plus white noise.
    const x = harmonicTone(f0.value, FS, W + MAX_LAG + 2, 10, noise.value, 4, (h) =>
      drop.value && h === 1 ? 0 : 1 / h,
    )
    const d = Array.from({ length: MAX_LAG }, (_, tau) => {
      let s = 0
      for (let j = 0; j < W; j++) s += (x[j] - x[j + tau]) ** 2
      return s
    })
    let running = 0
    const dPrime = d.map((v, tau) => {
      if (tau === 0) return 1
      running += v
      return (v * tau) / running
    })
    let tau = -1
    for (let t = MIN_LAG; t < MAX_LAG - 1; t++) {
      if (dPrime[t] < threshold.value) {
        while (t + 1 < MAX_LAG - 1 && dPrime[t + 1] < dPrime[t]) t++
        tau = t
        break
      }
    }
    if (tau < 0) tau = dPrime.slice(MIN_LAG).reduce((best, v, i) => (v < dPrime[best] ? i + MIN_LAG : best), MIN_LAG)
    const [a, b, c] = [dPrime[tau - 1], dPrime[tau], dPrime[tau + 1]]
    const refined = tau + (0.5 * (a - c)) / (a - 2 * b + c)
    // Plain autocorrelation over the same window, largest peak in the search range.
    const acf = Array.from({ length: MAX_LAG }, (_, t) => {
      let s = 0
      for (let j = 0; j < W; j++) s += x[j] * x[j + t]
      return s
    })
    const acfLag = acf.slice(MIN_LAG).reduce((best, v, i) => (v > acf[best] ? i + MIN_LAG : best), MIN_LAG)
    return { dPrime, tau, refined, acf: acf.map((v) => v / acf[0]), acfLag }
  }, [f0.value, noise.value, threshold.value, drop.value])

  const lags = Array.from({ length: MAX_LAG }, (_, t) => t)
  const series: XYSeries[] = [
    { name: 'YIN d′(τ)', type: 'line', x: lags, y: r.dPrime, slot: 0 },
    { name: 'normalised autocorrelation', type: 'line', x: lags, y: r.acf, slot: 1 },
    { name: 'threshold', type: 'line', x: [0, MAX_LAG], y: [threshold.value, threshold.value], dashed: true, slot: 2 },
    { name: 'YIN choice', type: 'scatter', x: [r.tau], y: [r.dPrime[r.tau]], emphasis: true },
  ]

  return (
    <Interactive
      title="YIN against the autocorrelation"
      caption="One 512-sample frame of a harmonic tone at 16 kHz with added noise. The autocorrelation (orange) has nearly equal peaks at every multiple of the period, and its largest peak in the search range is often a multiple, an octave or sub-harmonic error. YIN's cumulative mean normalised difference (blue) is 1 at lag 0 and dips towards 0 at each period; taking the first dip below the threshold picks the period itself, and parabolic interpolation refines it below one sample. Removing the fundamental does not change the period, and YIN still finds it."
      controls={
        <>
          <ParamSlider label="f₀ (Hz)" param={f0} />
          <ParamSlider label="noise level" param={noise} />
          <ParamSlider label="YIN threshold" param={threshold} />
          <ParamSwitch
            label="remove the fundamental"
            checked={drop.value === 1}
            onChange={(v) => drop.set(v ? 1 : 0)}
          />
        </>
      }
      readout={
        <>
          <Readout label="true period" value={`${formatNumber(FS / f0.value)} samples`} />
          <Readout label="YIN" value={`${formatNumber(FS / r.refined)} Hz`} />
          <Readout label="autocorrelation peak" value={`${formatNumber(FS / r.acfLag)} Hz`} />
        </>
      }
    >
      <XYChart series={series} xLabel="lag τ (samples)" yLabel="value" yRange={[-1, 2]} height={320} />
    </Interactive>
  )
}
