import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from 'aifn-render'
import { lfilter } from '@/lib/dsp'
import { rng } from '@/lib/math'
import { iirLowpass } from '../_shared/design'

const N = 300

/**
 * Forward–backward filtering: filter, reverse, filter again, reverse. Odd reflection padding at both ends, as in
 * scipy.signal.filtfilt, reduces the start-up transients.
 */
function filtfilt(b: number[], a: number[], x: number[]): number[] {
  const pad = Math.min(3 * Math.max(a.length, b.length), x.length - 1)
  const head = Array.from({ length: pad }, (_, i) => 2 * x[0] - x[pad - i])
  const tail = Array.from({ length: pad }, (_, i) => 2 * x[x.length - 1] - x[x.length - 2 - i])
  const ext = [...head, ...x, ...tail]
  const forward = Array.from(lfilter(b, a, ext)).reverse()
  const backward = Array.from(lfilter(b, a, forward)).reverse()
  return backward.slice(pad, pad + x.length)
}

/** A step and a pulse in noise, filtered causally and forward–backward with the same Butterworth filter. */
export function ZeroPhaseDemo() {
  const order = useParam(4, { min: 1, max: 8, step: 1 })
  const cutoff = useParam(0.08, { min: 0.02, max: 0.4, step: 0.01 })

  const r = useMemo(() => {
    const g = rng(5)
    const clean = Array.from({ length: N }, (_, n) => (n >= 80 ? 1 : 0) + Math.exp(-0.5 * ((n - 200) / 6) ** 2))
    const x = clean.map((v) => v + 0.15 * g.normal())
    const f = iirLowpass('butter', order.value, cutoff.value * Math.PI)
    const causal = Array.from(lfilter(f.b, f.a, x))
    const zero = filtfilt(f.b, f.a, x)
    // Where each output crosses half the step height: the causal filter's crossing is delayed.
    const cross = (y: number[]) => y.findIndex((v, n) => n > 60 && v > 0.5)
    return { x, clean, causal, zero, delay: cross(causal) - cross(zero) }
  }, [order.value, cutoff.value])

  const n = Array.from({ length: N }, (_, i) => i)
  const series: XYSeries[] = [
    { name: 'noisy input', type: 'line', x: n, y: r.x, muted: true },
    { name: 'causal (filter once)', type: 'line', x: n, y: r.causal, slot: 2 },
    { name: 'zero-phase (forward–backward)', type: 'line', x: n, y: r.zero, slot: 0 },
    { name: 'clean signal', type: 'line', x: n, y: r.clean, slot: 1, dashed: true },
  ]

  return (
    <Interactive
      title="Forward–backward filtering removes the delay"
      caption="A step at n = 80 and a pulse at n = 200 in noise. Filtering once with a Butterworth low-pass filter smooths the noise but delays and distorts the edges, because the filter's phase is not linear. Filtering forward, then backward in time, cancels the phase: the edges stay where they were and the pulse stays symmetric. The price is a squared magnitude response and a non-causal filter, usable only on recorded data."
      controls={
        <>
          <ParamSlider label="Butterworth order" param={order} format={(v) => String(v)} withArrows />
          <ParamSlider label="cutoff (×π)" param={cutoff} />
        </>
      }
      readout={
        <>
          <Readout label="step delay, causal vs zero-phase" value={`${r.delay} samples`} />
          <Readout label="effective order" value={`${2 * order.value} (magnitude squared)`} />
          <Readout label="gain at cutoff, zero-phase" value={`${formatNumber(-6.02)} dB`} />
        </>
      }
    >
      <XYChart series={series} xLabel="n" yLabel="amplitude" xRange={[0, N - 1]} yRange={[-0.5, 2]} height={320} />
    </Interactive>
  )
}
