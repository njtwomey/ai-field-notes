import { useMemo } from 'react'
import { Bars, Figure, formatNumber, Handle, int, Plot, Points, Readout, useAxis, useFigureState } from 'aifn-render'
import { convolve } from 'aifn/foundation/convolution'
import { tensor, toFlat } from 'aifn/foundation/tensor'

// A short input and a decaying impulse response, both starting at index 0.
const X = [1, 2, 1.5, 0.5, -1, -0.5, 0.25]
const H = [1, 0.6, 0.36, 0.216]
const Y = toFlat(convolve(tensor(X), tensor(H)))
const K = Array.from({ length: 16 }, (_, i) => i - 4)
const at = (a: number[], i: number) => (i >= 0 && i < a.length ? a[i] : 0)

/**
 * Convolution drawn as flip and slide: h is reversed and shifted to n, multiplied pointwise with x, and the products
 * summed to give y[n]. Step n to build the output one sample at a time.
 */
export function FlipAndSlide() {
  const state = useFigureState({
    n: int(3, { min: 0, max: Y.length - 1, step: 1, label: 'output index n' }),
  })

  const r = useMemo(() => {
    const flipped = K.map((k) => at(H, state.n - k))
    const products = K.map((k, i) => at(X, k) * flipped[i])
    return { flipped, products, sum: products.reduce((s, v) => s + v, 0) }
  }, [state.n])

  const top = [
    { name: 'x[k]', x: K, y: K.map((k) => at(X, k)), slot: 0 },
    { name: 'h[n − k]', x: K, y: r.flipped, slot: 1 },
    { name: 'product x[k] h[n − k]', x: K, y: r.products, emphasis: true },
  ] as const
  const outIndex = Y.map((_, i) => i)
  const bottom = [
    { name: 'y[m], m ≤ n', x: outIndex, y: Y.map((v, i) => (i <= state.n ? v : 0)), slot: 0 },
    { name: 'y[n]', x: [state.n], y: [Y[state.n]], emphasis: true },
  ] as const

  const xAxis = useAxis({ label: 'k', range: [-4.5, 11.5] })
  const yAxis = useAxis({ label: 'value', hold: 'union' })
  const xAxis2 = useAxis({ label: 'n', range: [-4.5, 11.5] })
  const yAxis2 = useAxis({ label: 'y[n]', hold: 'union' })
  return (
    <Figure
      title="Flip and slide"
      state={state}
      caption="To compute y[n], reverse the impulse response and slide it so its origin sits at n. Multiply it sample by sample with x and add up the products. Step n forward: the reversed response slides right, and each sum gives the next output sample. Drag the line labelled n in the lower chart, or step n with its field."
      readouts={
        <>
          <Readout label="n" value={state.n} />
          <Readout label="Σ x[k] h[n − k]" value={formatNumber(r.sum)} />
          <Readout label="output length" value={`${X.length} + ${H.length} − 1 = ${Y.length}`} />
        </>
      }
    >
      <div className="space-y-4">
        <Plot x={xAxis} y={yAxis} height={240}>
          <Bars {...top[0]} />
          <Points {...top[1]} />
          <Points {...top[2]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={200}>
          <Bars {...bottom[0]} />
          <Points {...bottom[1]} />
          <Handle {...state.handle('n', { label: 'n' })} />
        </Plot>
      </div>
    </Figure>
  )
}
