import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'
import { rng } from '@/lib/math'

const N = 250

/** Running statistic over a centred window of 2m + 1 samples, repeating the end samples at the edges. */
function running(x: number[], m: number, stat: (w: number[]) => number): number[] {
  return x.map((_, n) => stat(Array.from({ length: 2 * m + 1 }, (_, k) => x[Math.min(N - 1, Math.max(0, n + k - m))])))
}

const median = (w: number[]) => [...w].sort((a, b) => a - b)[(w.length - 1) / 2]
const mean = (w: number[]) => w.reduce((s, v) => s + v, 0) / w.length

/** Samples needed to rise from 10 % to 90 % of the step at n = 120 (step height 1, from 0 to 1). */
function riseWidth(y: number[]): number {
  const lo = y.findIndex((v, n) => n > 90 && v > 0.1)
  const hi = y.findIndex((v, n) => n > 90 && v > 0.9)
  return lo < 0 || hi < 0 ? NaN : hi - lo
}

/** A step and a ramp with small Gaussian noise and sparse impulses, smoothed by a running median and a running mean. */
export function MedianVsMean() {
  const half = useParam(3, { min: 1, max: 12, step: 1 })
  const rate = useParam(0.08, { min: 0, max: 0.3, step: 0.01 })

  const r = useMemo(() => {
    const g = rng(11)
    const clean = Array.from({ length: N }, (_, n) => (n < 120 ? 0 : n < 180 ? 1 : 1 - (n - 180) / 70))
    const x = clean.map((v) => {
      const u = g.uniform()
      const impulse = u < rate.value ? (g.uniform() < 0.5 ? -2.5 : 2.5) : 0
      return v + 0.05 * g.normal() + impulse
    })
    const med = running(x, half.value, median)
    const avg = running(x, half.value, mean)
    const rms = (y: number[]) => Math.sqrt(y.reduce((s, v, n) => s + (v - clean[n]) ** 2, 0) / N)
    return { x, clean, med, avg, rmsMed: rms(med), rmsAvg: rms(avg), riseMed: riseWidth(med), riseAvg: riseWidth(avg) }
  }, [half.value, rate.value])

  const n = Array.from({ length: N }, (_, i) => i)
  const series: XYSeries[] = [
    { name: 'input with impulses', type: 'line', x: n, y: r.x, muted: true },
    { name: `moving average (${2 * half.value + 1} points)`, type: 'line', x: n, y: r.avg, slot: 2 },
    { name: `median (${2 * half.value + 1} points)`, type: 'line', x: n, y: r.med, slot: 0 },
    { name: 'clean signal', type: 'line', x: n, y: r.clean, slot: 1, dashed: true },
  ]

  return (
    <Interactive
      title="Running median against running mean"
      caption="A step at n = 120 followed by a ramp, with small Gaussian noise and random impulses of ±2.5. The moving average spreads every impulse over its window and blurs the step. The median ignores an impulse as long as fewer than half the samples in the window are outliers, and passes the step unchanged. Raise the impulse rate until clusters of more than m impulses appear in one window: the median then fails too."
      controls={
        <>
          <ParamSlider
            label="half-width m (window 2m + 1)"
            param={half}
            format={(v) => `${v}  (${2 * v + 1} points)`}
            withArrows
          />
          <ParamSlider label="impulse probability per sample" param={rate} format={(v) => v.toFixed(2)} />
        </>
      }
      readout={
        <>
          <Readout label="RMS error, median" value={formatNumber(r.rmsMed)} />
          <Readout label="RMS error, moving average" value={formatNumber(r.rmsAvg)} />
          <Readout label="10–90 % rise of step, median / average" value={`${r.riseMed} / ${r.riseAvg} samples`} />
        </>
      }
    >
      <XYChart series={series} xLabel="n" yLabel="amplitude" xRange={[0, N - 1]} yRange={[-3, 3.5]} height={320} />
    </Interactive>
  )
}
