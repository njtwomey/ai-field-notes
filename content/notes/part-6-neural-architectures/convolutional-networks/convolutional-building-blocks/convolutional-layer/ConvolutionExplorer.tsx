import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Points,
  Raster,
  Readout,
  slider,
  Slider,
  useAxis,
  useFigureState,
} from 'aifn-render'

const N = 12
const K = 3
/** The largest output size, at padding 1 and stride 1. */
const MAX_OUT = N + 2 - K + 1

// A 12 × 12 binary image: a hollow square and a filled bar. Rows are counted from the bottom, as on the chart.
const IMAGE: number[][] = Array.from({ length: N }, (_, r) =>
  Array.from({ length: N }, (_, c) => {
    const square = r >= 5 && r <= 10 && c >= 1 && c <= 6 && (r === 5 || r === 10 || c === 1 || c === 6)
    const bar = r >= 1 && r <= 3 && c >= 5 && c <= 10
    return square || bar ? 1 : 0
  }),
)

type KernelName = 'blur' | 'vertical' | 'horizontal' | 'sharpen'

// Row u of each kernel multiplies input row (top-left row + u), counted upwards like the image.
const KERNELS: Record<KernelName, number[][]> = {
  blur: [
    [1 / 9, 1 / 9, 1 / 9],
    [1 / 9, 1 / 9, 1 / 9],
    [1 / 9, 1 / 9, 1 / 9],
  ],
  vertical: [
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1],
  ],
  horizontal: [
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1],
  ],
  sharpen: [
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0],
  ],
}

const pixel = (r: number, c: number) => (r >= 0 && r < N && c >= 0 && c < N ? IMAGE[r][c] : 0)

/** Cross-correlation with zero padding p and stride s, as computed by a convolution layer. */
function convolve(kernel: number[][], p: number, s: number): number[][] {
  const m = Math.floor((N + 2 * p - K) / s) + 1
  return Array.from({ length: m }, (_, i) =>
    Array.from({ length: m }, (_, j) => {
      let acc = 0
      for (let u = 0; u < K; u++) for (let v = 0; v < K; v++) acc += kernel[u][v] * pixel(i * s - p + u, j * s - p + v)
      return acc
    }),
  )
}

const range = (lo: number, hi: number) => Array.from({ length: hi - lo + 1 }, (_, i) => lo + i)

/**
 * A 3 × 3 kernel slides over a 12 × 12 image. The window on the input and the cell it produces on the output move
 * together; drag the window or set the output position with the sliders.
 */
export function ConvolutionExplorer() {
  const state = useFigureState({
    kernel: choice<KernelName>(
      [
        { value: 'vertical', label: 'vertical edge' },
        { value: 'horizontal', label: 'horizontal edge' },
        { value: 'blur', label: 'blur' },
        { value: 'sharpen', label: 'sharpen' },
      ],
      'vertical',
      { label: 'kernel' },
    ),
    padding: choice<'0' | '1'>(
      [
        { value: '0', label: '0' },
        { value: '1', label: '1' },
      ],
      '0',
      { label: 'padding p' },
    ),
    stride: choice<'1' | '2'>(
      [
        { value: '1', label: '1' },
        { value: '2', label: '2' },
      ],
      '1',
      { label: 'stride s' },
    ),
    // The output position: its range depends on padding and stride, so its sliders are placed by hand below.
    row: slider(0, MAX_OUT - 1, 4, { step: 1, onChart: true }),
    col: slider(0, MAX_OUT - 1, 2, { step: 1, onChart: true }),
  })
  const kernel = state.kernel
  const p = Number(state.padding)
  const s = Number(state.stride)
  const m = Math.floor((N + 2 * p - K) / s) + 1
  const i = Math.min(state.row, m - 1)
  const j = Math.min(state.col, m - 1)

  const output = useMemo(() => convolve(KERNELS[kernel], p, s), [kernel, p, s])
  const input = useMemo(() => range(-p, N - 1 + p).map((r) => range(-p, N - 1 + p).map((c) => pixel(r, c))), [p])

  // Bottom-left corner of the window in input coordinates, and its outline half a cell outside the cell centres.
  const r0 = i * s - p
  const c0 = j * s - p
  const window = [
    {
      name: 'window',
      x: [c0 - 0.5, c0 + K - 0.5, c0 + K - 0.5, c0 - 0.5, c0 - 0.5],
      y: [r0 - 0.5, r0 - 0.5, r0 + K - 0.5, r0 + K - 0.5, r0 - 0.5],
      emphasis: true,
    },
  ] as const
  const terms = KERNELS[kernel].flatMap((ku, u) => ku.map((kv, v) => kv * pixel(r0 + u, c0 + v)))
  const signed = kernel !== 'blur'
  const bound = kernel === 'sharpen' ? 5 : 3
  const outRange: [number, number] = signed ? [-bound, bound] : [0, 1]

  const xAxis = useAxis({ label: 'column' })
  const yAxis = useAxis({ label: 'row' })
  const xAxis2 = useAxis({ label: 'column j' })
  const yAxis2 = useAxis({ label: 'row i' })
  return (
    <Figure
      title="A 3 × 3 kernel sliding over an image"
      caption="Left: the input, with zero padding shown as the outer ring when padding is 1. Right: the output feature map. Drag the window on the input (or use the sliders) to choose an output cell; its value is the sum of the kernel times the nine pixels under the window. The same nine weights produce every output cell. Stride 2 skips every other position and roughly halves the output size."
      state={state}
      controls={
        <>
          <Slider label="output row i" value={i} onChange={(v) => state.set('row', v)} min={0} max={m - 1} step={1} />
          <Slider
            label="output column j"
            value={j}
            onChange={(v) => state.set('col', v)}
            min={0}
            max={m - 1}
            step={1}
          />
        </>
      }
      readouts={
        <>
          <Readout label="output size ⌊(n + 2p − k)/s⌋ + 1" value={`⌊(${N} + ${2 * p} − ${K})/${s}⌋ + 1 = ${m}`} />
          <Readout label={`y[${i}, ${j}]`} value={formatNumber(terms.reduce((a, b) => a + b, 0))} />
          <Readout label="weights" value={`${K * K} (shared by all ${m * m} outputs)`} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={320} ariaLabel={'Input image with the kernel window'}>
          <Raster x={range(-p, N - 1 + p)} y={range(-p, N - 1 + p)} z={input} range={[0, 1]} valueLabel={'pixel'} />
          <Curve {...window[0]} live />
          <Handle
            kind="point"
            at={[c0 + 1, r0 + 1]}
            onDrag={([x, y]) => {
              state.set('col', Math.min(Math.max(Math.round((x - 1 + p) / s), 0), m - 1))
              state.set('row', Math.min(Math.max(Math.round((y - 1 + p) / s), 0), m - 1))
            }}
            label="window"
          />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={320} ariaLabel={'Output feature map'}>
          <Raster
            x={range(0, m - 1)}
            y={range(0, m - 1)}
            z={output}
            scale={signed ? 'diverging' : 'sequential'}
            range={outRange}
            valueLabel={'output'}
          />
          <Points x={[j]} y={[i]} emphasis live />
        </Plot>
      </div>
    </Figure>
  )
}
