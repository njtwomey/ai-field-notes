import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, useParam, type XYSeries } from '@/components/viz'
import { dft } from '@/lib/dsp'
import { linspace } from '@/lib/math'

// A short sequence: a decaying oscillation, 8 samples.
const X = Array.from({ length: 8 }, (_, n) => 0.8 ** n * Math.cos(0.9 * n))
const OMEGA = linspace(0, 2 * Math.PI, 721)
const dtftMagnitude = (w: number) => {
  let re = 0
  let im = 0
  X.forEach((v, n) => {
    re += v * Math.cos(-w * n)
    im += v * Math.sin(-w * n)
  })
  return Math.hypot(re, im)
}
const CURVE = OMEGA.map(dtftMagnitude)

/** The N-point DFT of an 8-sample sequence lands exactly on its DTFT at the frequencies 2πk/N. */
export function DftSamplesDtft() {
  const n = useParam(8, { min: 8, max: 64, step: 1 })

  const r = useMemo(() => {
    const padded = [...X, ...Array(n.value - X.length).fill(0)]
    const { re, im } = dft(padded)
    return Array.from(re, (v, k) => ({ w: (2 * Math.PI * k) / n.value, m: Math.hypot(v, im[k]) }))
  }, [n.value])

  const series: XYSeries[] = [
    { name: 'DTFT |X(e^{iω})|', type: 'line', x: OMEGA.map((w) => w / Math.PI), y: CURVE, slot: 0 },
    {
      name: `${n.value}-point DFT |X[k]|`,
      type: 'scatter',
      x: r.map((p) => p.w / Math.PI),
      y: r.map((p) => p.m),
      emphasis: true,
    },
  ]

  return (
    <Interactive
      title="The DFT samples the DTFT"
      caption="An 8-sample sequence has a continuous, 2π-periodic DTFT (line). Its N-point DFT, computed after padding to N samples with zeros, gives exactly N equally spaced samples of that curve, at ω = 2πk/N (points). A larger N samples the same curve more densely; it does not change the curve."
      controls={<ParamSlider label="DFT length N" param={n} format={(v) => String(v)} withArrows />}
      readout={
        <>
          <Readout label="sequence length" value={X.length} />
          <Readout label="bin spacing 2π/N" value={`${(2 / n.value).toFixed(3)}π`} />
        </>
      }
    >
      <XYChart series={series} xLabel="ω / π" yLabel="magnitude" xRange={[0, 2]} height={260} />
    </Interactive>
  )
}
