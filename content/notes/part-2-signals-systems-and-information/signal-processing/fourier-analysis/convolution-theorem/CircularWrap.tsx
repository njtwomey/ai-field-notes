import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from 'aifn-render'
import { convolve } from '@/lib/dsp'

const X = [1, 1, 1, 1, 1, 1]
const H = [1, 0.75, 0.5, 0.25]
const LINEAR = Array.from(convolve(X, H))

/** Circular convolution of length N: indices wrap modulo N, so the tail of the linear result folds onto the start. */
const circular = (n: number) => {
  const out = new Array(n).fill(0)
  X.forEach((xv, i) => H.forEach((hv, j) => (out[(i + j) % n] += xv * hv)))
  return out
}

/**
 * Multiplying N-point DFTs convolves circularly. With N ≥ |x| + |h| − 1 = 9 the result equals linear convolution;
 * shorter N wraps the tail around.
 */
export function CircularWrap() {
  const n = useParam(6, { min: 4, max: 12, step: 1 })
  const circ = useMemo(() => circular(n.value), [n.value])
  const error = LINEAR.reduce((s, v, i) => s + Math.abs(v - (i < n.value ? circ[i] : 0)), 0)

  const series: XYSeries[] = [
    { name: 'linear convolution', type: 'bar', x: LINEAR.map((_, i) => i), y: LINEAR, slot: 0 },
    { name: `circular, N = ${n.value}`, type: 'scatter', x: circ.map((_, i) => i), y: circ, emphasis: true },
  ]

  return (
    <Interactive
      title="Circular convolution wraps around"
      caption="A length-6 sequence convolved with a length-4 response. Linear convolution has 6 + 4 − 1 = 9 samples (bars). Multiplying N-point DFTs gives circular convolution (points): when N < 9 the samples past N fold back onto the start and add to them. From N = 9 the two agree."
      controls={<ParamSlider label="DFT length N" param={n} format={(v) => String(v)} withArrows />}
      readout={
        <>
          <Readout label="needed N ≥" value={LINEAR.length} />
          <Readout label="wrapped samples" value={Math.max(0, LINEAR.length - n.value)} />
          <Readout label="total error vs linear" value={formatNumber(error)} />
        </>
      }
    >
      <XYChart series={series} xLabel="n" yLabel="value" xRange={[-0.5, 11.5]} yRange={[0, 5]} height={260} />
    </Interactive>
  )
}
