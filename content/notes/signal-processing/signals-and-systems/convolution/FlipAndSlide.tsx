import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from 'aifn-render'
import { convolve } from '@/lib/dsp'

// A short input and a decaying impulse response, both starting at index 0.
const X = [1, 2, 1.5, 0.5, -1, -0.5, 0.25]
const H = [1, 0.6, 0.36, 0.216]
const Y = Array.from(convolve(X, H))
const K = Array.from({ length: 16 }, (_, i) => i - 4)
const at = (a: number[], i: number) => (i >= 0 && i < a.length ? a[i] : 0)

/**
 * Convolution drawn as flip and slide: h is reversed and shifted to n, multiplied pointwise with x, and the products
 * summed to give y[n]. Step n to build the output one sample at a time.
 */
export function FlipAndSlide() {
  const n = useParam(3, { min: 0, max: Y.length - 1, step: 1 })

  const r = useMemo(() => {
    const flipped = K.map((k) => at(H, n.value - k))
    const products = K.map((k, i) => at(X, k) * flipped[i])
    return { flipped, products, sum: products.reduce((s, v) => s + v, 0) }
  }, [n.value])

  const top: XYSeries[] = [
    { name: 'x[k]', type: 'bar', x: K, y: K.map((k) => at(X, k)), slot: 0 },
    { name: 'h[n − k]', type: 'scatter', x: K, y: r.flipped, slot: 1 },
    { name: 'product x[k] h[n − k]', type: 'scatter', x: K, y: r.products, emphasis: true },
  ]
  const outIndex = Y.map((_, i) => i)
  const bottom: XYSeries[] = [
    { name: 'y[m], m ≤ n', type: 'bar', x: outIndex, y: Y.map((v, i) => (i <= n.value ? v : 0)), slot: 0 },
    { name: 'y[n]', type: 'scatter', x: [n.value], y: [Y[n.value]], emphasis: true },
  ]

  return (
    <Interactive
      title="Flip and slide"
      caption="To compute y[n], reverse the impulse response and slide it so its origin sits at n. Multiply it sample by sample with x and add up the products. Step n forward: the reversed response slides right, and each sum gives the next output sample."
      controls={<ParamSlider label="output index n" param={n} format={(v) => String(v)} withArrows />}
      readout={
        <>
          <Readout label="n" value={n.value} />
          <Readout label="Σ x[k] h[n − k]" value={formatNumber(r.sum)} />
          <Readout label="output length" value={`${X.length} + ${H.length} − 1 = ${Y.length}`} />
        </>
      }
    >
      <div className="space-y-4">
        <XYChart series={top} xLabel="k" yLabel="value" xRange={[-4.5, 11.5]} height={240} />
        <XYChart series={bottom} xLabel="n" yLabel="y[n]" xRange={[-4.5, 11.5]} height={200} />
      </div>
    </Interactive>
  )
}
