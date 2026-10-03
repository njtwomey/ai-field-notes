import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from 'aifn-render'
import { linspace, rng } from '@/lib/math'
import { periodogram, powerDb } from '../_shared/spectra'

const N = 240
const W1 = 0.3 * Math.PI
const W2 = 0.8 * Math.PI
const GRID = linspace(0.005 * Math.PI, 1.2 * Math.PI, 900)

/** Lomb–Scargle periodogram P(ω) = ½[(Σ y c)²/Σ c² + (Σ y s)²/Σ s²], with c, s at the time offset τ. */
function lombScargle(t: number[], y: number[], omega: number[]): number[] {
  return omega.map((w) => {
    let s2 = 0
    let c2 = 0
    for (const tj of t) {
      s2 += Math.sin(2 * w * tj)
      c2 += Math.cos(2 * w * tj)
    }
    const tau = Math.atan2(s2, c2) / (2 * w)
    let yc = 0
    let ys = 0
    let cc = 0
    let ss = 0
    t.forEach((tj, j) => {
      const c = Math.cos(w * (tj - tau))
      const s = Math.sin(w * (tj - tau))
      yc += y[j] * c
      ys += y[j] * s
      cc += c * c
      ss += s * s
    })
    return 0.5 * ((yc * yc) / cc + (ys * ys) / ss)
  })
}

/** Peak value of y within ±width of x0 on the grid x. */
const localPeak = (x: number[], y: number[], x0: number, width: number) =>
  Math.max(...y.filter((_, i) => Math.abs(x[i] - x0) < width))

/**
 * An irregularly sampled signal with a strong component at 0.3π and a half-amplitude one at 0.8π rad/s (mean spacing
 * 1 s). Lomb–Scargle fits sinusoids at the sample times; the alternative interpolates to a grid first.
 */
export function LombScargleDemo() {
  const jitter = useParam(0.45, { min: 0, max: 0.5, step: 0.05 })
  const gap = useParam(0.2, { min: 0, max: 0.5, step: 0.05 })
  const seed = useParam(2, { min: 1, max: 30, step: 1 })

  const r = useMemo(() => {
    const g = rng(seed.value)
    // Nominal times 0..N−1, jittered, then a contiguous gap removed from the middle of the record.
    const all = Array.from({ length: N }, (_, j) => j + jitter.value * (2 * g.uniform() - 1))
    const gapStart = Math.floor((N * (1 - gap.value)) / 2)
    const gapEnd = gapStart + Math.floor(N * gap.value)
    const t = all.filter((_, j) => j < gapStart || j >= gapEnd).sort((a, b) => a - b)
    const raw = t.map((tj) => Math.cos(W1 * tj) + 0.5 * Math.cos(W2 * tj) + 0.2 * g.normal())
    const mean = raw.reduce((s, v) => s + v, 0) / raw.length
    const y = raw.map((v) => v - mean)
    const ls = lombScargle(t, y, GRID)
    // Naive alternative: linear interpolation onto a unit grid, then a Hann-windowed periodogram.
    const grid: number[] = []
    const yi: number[] = []
    let k = 0
    for (let tg = Math.ceil(t[0]); tg <= t[t.length - 1]; tg++) {
      while (t[k + 1] < tg) k++
      const frac = (tg - t[k]) / (t[k + 1] - t[k])
      grid.push(tg)
      yi.push(y[k] + frac * (y[k + 1] - y[k]))
    }
    const p = periodogram(yi, 'hann', 4096)
    const lsMax = Math.max(...ls)
    const pMax = Math.max(...p.psd)
    const lsDb = ls.map((v) => powerDb(v / lsMax))
    const pDb = p.psd.map((v) => powerDb(v / pMax))
    const gridPi = GRID.map((w) => w / Math.PI)
    const pPi = p.omega.map((w) => w / Math.PI)
    const ratio = (x: number[], v: number[]) => localPeak(x, v, 0.8, 0.03) - localPeak(x, v, 0.3, 0.03)
    return { gridPi, lsDb, pPi, pDb, count: t.length, lsRatio: ratio(gridPi, lsDb), interpRatio: ratio(pPi, pDb) }
  }, [jitter.value, gap.value, seed.value])

  const series: XYSeries[] = [
    { name: 'periodogram of interpolated signal', type: 'line', x: r.pPi, y: r.pDb, slot: 2 },
    { name: 'Lomb–Scargle', type: 'line', x: r.gridPi, y: r.lsDb, slot: 0 },
  ]

  return (
    <Interactive
      title="Irregular samples: fit, don't interpolate"
      caption="Samples at jittered times with a gap, mean spacing 1 s, of cos(0.3πt) + 0.5 cos(0.8πt) plus noise; the 0.8π component should sit 6 dB below the 0.3π one. Lomb–Scargle fits sinusoids at the actual sample times and keeps that ratio. Interpolating onto a regular grid first acts as a low-pass filter, attenuating the high-frequency component, and invents data in the gap. Lomb–Scargle is also defined above the pseudo-Nyquist frequency π rad/s, where it shows aliases whose strength depends on how irregular the sampling is."
      controls={
        <>
          <ParamSlider label="timing jitter (× mean spacing)" param={jitter} withArrows />
          <ParamSlider label="gap (fraction of record)" param={gap} withArrows />
          <ParamSlider label="seed" param={seed} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="samples" value={r.count} />
          <Readout label="0.8π vs 0.3π, Lomb–Scargle" value={`${formatNumber(r.lsRatio)} dB`} />
          <Readout label="0.8π vs 0.3π, interpolated" value={`${formatNumber(r.interpRatio)} dB`} />
          <Readout label="true" value="−6.0 dB" />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="ω / π (rad/s)"
        yLabel="normalised power (dB)"
        xRange={[0, 1.2]}
        yRange={[-50, 5]}
        height={320}
      />
    </Interactive>
  )
}
