import { useMemo } from 'react'
import { Bars, Figure, formatNumber, int, Plot, Points, Readout, useAxis, useFigureState } from 'aifn-render'
import { convolve } from 'aifn-compute/foundation/convolution'
import { toFlat } from 'aifn-compute/foundation/tensor'

const X = [1, 1, 1, 1, 1, 1]
const H = [1, 0.75, 0.5, 0.25]
const LINEAR = toFlat(convolve(X, H))

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
  const state = useFigureState({
    n: int(6, { min: 4, max: 12, step: 1, label: 'DFT length N', format: (v) => String(v) }),
  })
  const circ = useMemo(() => circular(state.n), [state.n])
  const error = LINEAR.reduce((s, v, i) => s + Math.abs(v - (i < state.n ? circ[i] : 0)), 0)

  const series = [
    { name: 'linear convolution', x: LINEAR.map((_, i) => i), y: LINEAR, slot: 0 },
    { name: `circular, N = ${state.n}`, x: circ.map((_, i) => i), y: circ, emphasis: true },
  ] as const

  const xAxis = useAxis({ label: 'n', range: [-0.5, 11.5] })
  const yAxis = useAxis({ label: 'value', range: [0, 5] })
  return (
    <Figure
      title="Circular convolution wraps around"
      state={state}
      caption="A length-6 sequence convolved with a length-4 response. Linear convolution has 6 + 4 − 1 = 9 samples (bars). Multiplying N-point DFTs gives circular convolution (points): when N < 9 the samples past N fold back onto the start and add to them. From N = 9 the two agree."

      readouts={
        <>
          <Readout label="needed N ≥" value={LINEAR.length} />
          <Readout label="wrapped samples" value={Math.max(0, LINEAR.length - state.n)} />
          <Readout label="total error vs linear" value={formatNumber(error)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={260}>
        <Bars {...series[0]} />
        <Points {...series[1]} />
      </Plot>
    </Figure>
  )
}
