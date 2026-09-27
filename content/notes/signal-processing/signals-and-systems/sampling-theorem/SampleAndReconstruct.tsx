import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'
import { linspace } from '@/lib/math'

const FS = 8
const T_END = 2
const DENSE = linspace(0, T_END, 801)
// Samples well beyond the plotted window, so the truncated sinc sum barely differs from the ideal one inside it.
const PAD = 40
const sinc = (u: number) => (u === 0 ? 1 : Math.sin(Math.PI * u) / (Math.PI * u))

/**
 * A cosine sampled at f_s = 8 Hz and rebuilt by ideal sinc interpolation. Below f_s/2 the rebuilt curve is the
 * original; above it, the samples are those of a lower-frequency alias, and interpolation returns the alias.
 */
export function SampleAndReconstruct() {
  const f = useParam(1.5, { min: 0.25, max: 12, step: 0.25 })
  const phase = 0.4

  const r = useMemo(() => {
    const signal = (t: number) => Math.cos(2 * Math.PI * f.value * t + phase)
    const n0 = -PAD
    const n1 = Math.ceil(T_END * FS) + PAD
    const samples = Array.from({ length: n1 - n0 + 1 }, (_, i) => {
      const n = n0 + i
      return { n, t: n / FS, x: signal(n / FS) }
    })
    const rebuilt = DENSE.map((t) => samples.reduce((acc, s) => acc + s.x * sinc(FS * t - s.n), 0))
    const shown = samples.filter((s) => s.t >= 0 && s.t <= T_END)
    // The apparent frequency: f folded into [0, f_s/2].
    const folded = Math.abs(f.value - FS * Math.round(f.value / FS))
    return { rebuilt, shown, folded, original: DENSE.map(signal) }
  }, [f.value])

  const series: XYSeries[] = [
    { name: 'x(t)', type: 'line', x: DENSE, y: r.original, slot: 0 },
    { name: 'sinc reconstruction', type: 'line', x: DENSE, y: r.rebuilt, slot: 1, dashed: true },
    { name: 'samples x[n]', type: 'scatter', x: r.shown.map((s) => s.t), y: r.shown.map((s) => s.x), emphasis: true },
  ]
  const aliased = f.value > FS / 2

  return (
    <Interactive
      title="Sampling and ideal reconstruction"
      caption="A cosine of frequency f is sampled at f_s = 8 Hz (dots) and rebuilt from its samples by sinc interpolation (dashed). Below the Nyquist frequency f_s/2 = 4 Hz the reconstruction lies on the original. Above it, the same samples belong to a slower cosine, and interpolation returns that alias instead. Exactly at 4 Hz the result depends on the phase."
      controls={<ParamSlider label="signal frequency f (Hz)" param={f} withArrows />}
      readout={
        <>
          <Readout label="f" value={`${formatNumber(f.value)} Hz`} />
          <Readout label="Nyquist f_s/2" value={`${FS / 2} Hz`} />
          <Readout label="reconstructed frequency" value={`${formatNumber(r.folded)} Hz`} />
          <Readout label="status" value={aliased ? 'aliased' : f.value === FS / 2 ? 'at Nyquist' : 'recovered'} />
        </>
      }
    >
      <XYChart series={series} xLabel="t (s)" yLabel="x(t)" xRange={[0, T_END]} yRange={[-1.6, 1.6]} height={300} />
    </Interactive>
  )
}
