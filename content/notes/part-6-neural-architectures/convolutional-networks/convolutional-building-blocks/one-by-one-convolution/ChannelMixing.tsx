import { useMemo } from 'react'
import {
  Figure,
  formatNumber,
  Handle,
  Plot,
  Points,
  Raster,
  Readout,
  row,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { conv2d } from 'aifn-compute/nn/functional'
import { tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { FixedHeight } from '../_shared/FixedHeight'

const N = 8
const CHANNELS = ['red', 'green', 'blue'] as const

// An 8 × 8 RGB image: a red disc on a background that runs from green (left) to blue (right).
const IMAGE: number[][][] = (() => {
  const red: number[][] = []
  const green: number[][] = []
  const blue: number[][] = []
  for (let r = 0; r < N; r++) {
    red.push([])
    green.push([])
    blue.push([])
    for (let c = 0; c < N; c++) {
      const disc = (r - 3.5) ** 2 + (c - 4) ** 2 < 6.5
      const t = c / (N - 1)
      red[r].push(disc ? 0.9 : 0.1)
      green[r].push(disc ? 0.15 : 0.8 * (1 - t) + 0.1)
      blue[r].push(disc ? 0.1 : 0.8 * t + 0.1)
    }
  }
  return [red, green, blue]
})()
const X = tensor(IMAGE.flat(2), [1, 3, N, N])
const cells = Array.from({ length: N }, (_, i) => i)
const grid = (flat: number[], channel: number) => cells.map((r) => cells.map((c) => flat[channel * N * N + r * N + c]))

/**
 * A 1 × 1 convolution with three input channels and two output channels: one 2 × 3 matrix applied to the colour vector
 * of every pixel. Drag the marked pixel to read its input vector and its two outputs.
 */
export function ChannelMixing() {
  const state = useFigureState({
    first: row('output channel 1 (initially luminance)', {
      r: slider(-1, 1, 0.3, { step: 0.01, label: 'weight on red' }),
      g: slider(-1, 1, 0.59, { step: 0.01, label: 'weight on green' }),
      b: slider(-1, 1, 0.11, { step: 0.01, label: 'weight on blue' }),
    }),
    second: row('output channel 2 (initially red minus green)', {
      r: slider(-1, 1, 1, { step: 0.05, label: 'weight on red' }),
      g: slider(-1, 1, -1, { step: 0.05, label: 'weight on green' }),
      b: slider(-1, 1, 0, { step: 0.05, label: 'weight on blue' }),
    }),
    col: slider(0, N - 1, 4, { step: 1, onChart: true }),
    row: slider(0, N - 1, 3, { step: 1, onChart: true }),
  })
  const W = [
    [state.first.r, state.first.g, state.first.b],
    [state.second.r, state.second.g, state.second.b],
  ]
  const key = W.flat().join(',')

  // The layer itself: kernels [C' = 2, C = 3, 1, 1].
  const output = useMemo(() => toFlat(conv2d(X, tensor(key.split(',').map(Number), [2, 3, 1, 1])) as Tensor), [key])

  const i = state.row
  const j = state.col
  const pixel = CHANNELS.map((_, c) => IMAGE[c][i][j])
  const y = [0, 1].map((o) => output[o * N * N + i * N + j])

  const xAxis = useAxis({ label: 'column', range: [-0.5, N - 0.5], zoom: false })
  const yAxis = useAxis({ label: 'row', range: [-0.5, N - 0.5], inverse: true, zoom: false })
  const marker = <Points x={[j]} y={[i]} emphasis live />
  return (
    <Figure
      title="A 1 × 1 convolution mixes channels at every pixel"
      state={state}
      caption="Top: the three input channels of an 8 × 8 colour image. Bottom: the two output channels of a 1 × 1 convolution, each a weighted sum of the three input values at the same pixel. The six weights form a 2 × 3 matrix, shared by all 64 pixels. Drag the marked pixel on the red channel to read its input vector and its outputs; move the weight sliders to change the matrix. The first output starts as the luminance weights (0.3, 0.59, 0.11), so it is the greyscale image; the second starts as red minus green, which lights up the disc."
      readouts={
        <>
          <Readout label={`input at (${i}, ${j})`} value={`(${pixel.map((v) => formatNumber(v)).join(', ')})`} />
          <Readout label="output = W · input" value={`(${y.map((v) => formatNumber(v)).join(', ')})`} />
          <Readout label="weights" value="2 × 3 = 6, whatever the image size" />
          <Readout label="multiply-accumulates" value={`6 × ${N * N} pixels = ${6 * N * N}`} />
        </>
      }
    >
      <FixedHeight height={190}>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          {CHANNELS.map((name, c) => (
            <div key={name} className="min-w-0 space-y-1">
              <div className="text-center text-xs text-muted-foreground">input: {name}</div>
              <Plot x={xAxis} y={yAxis} ariaLabel={`Input channel ${name}`}>
                <Raster x={cells} y={cells} z={IMAGE[c]} range={[0, 1]} valueLabel={name} colorBar={c === 2} />
                {marker}
                {c === 0 ? <Handle {...state.handle(['col', 'row'], { label: 'pixel' })} /> : null}
              </Plot>
            </div>
          ))}
        </div>
        <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
          {[0, 1].map((o) => (
            <div key={o} className="min-w-0 space-y-1">
              <div className="text-center text-xs text-muted-foreground">output channel {o + 1}</div>
              <Plot x={xAxis} y={yAxis} ariaLabel={`Output channel ${o + 1}`}>
                <Raster
                  x={cells}
                  y={cells}
                  z={grid(output, o)}
                  scale="diverging"
                  range={[-1.5, 1.5]}
                  valueLabel={`output ${o + 1}`}
                />
                {marker}
              </Plot>
            </div>
          ))}
        </div>
      </FixedHeight>
    </Figure>
  )
}
