import { useMemo, useState } from 'react'
import {
  Heatmap,
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  formatNumber,
  useParam,
  type HeatmapOverlay,
} from 'aifn-render'

const N = 12
const K = 3

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
  const [kernel, setKernel] = useState<KernelName>('vertical')
  const [padding, setPadding] = useState<'0' | '1'>('0')
  const [stride, setStride] = useState<'1' | '2'>('1')
  const p = Number(padding)
  const s = Number(stride)
  const m = Math.floor((N + 2 * p - K) / s) + 1
  const row = useParam(4, { min: 0, max: m - 1, step: 1 })
  const col = useParam(2, { min: 0, max: m - 1, step: 1 })
  const i = Math.min(row.value, m - 1)
  const j = Math.min(col.value, m - 1)

  const output = useMemo(() => convolve(KERNELS[kernel], p, s), [kernel, p, s])
  const input = useMemo(() => range(-p, N - 1 + p).map((r) => range(-p, N - 1 + p).map((c) => pixel(r, c))), [p])

  // Bottom-left corner of the window in input coordinates, and its outline half a cell outside the cell centres.
  const r0 = i * s - p
  const c0 = j * s - p
  const window: HeatmapOverlay[] = [
    {
      name: 'window',
      type: 'line',
      x: [c0 - 0.5, c0 + K - 0.5, c0 + K - 0.5, c0 - 0.5, c0 - 0.5],
      y: [r0 - 0.5, r0 - 0.5, r0 + K - 0.5, r0 + K - 0.5, r0 - 0.5],
      emphasis: true,
    },
  ]
  const terms = KERNELS[kernel].flatMap((ku, u) => ku.map((kv, v) => kv * pixel(r0 + u, c0 + v)))
  const signed = kernel !== 'blur'
  const bound = kernel === 'sharpen' ? 5 : 3
  const outRange: [number, number] = signed ? [-bound, bound] : [0, 1]

  return (
    <Interactive
      title="A 3 × 3 kernel sliding over an image"
      caption="Left: the input, with zero padding shown as the outer ring when padding is 1. Right: the output feature map. Drag the window on the input (or use the sliders) to choose an output cell; its value is the sum of the kernel times the nine pixels under the window. The same nine weights produce every output cell. Stride 2 skips every other position and roughly halves the output size."
      controls={
        <>
          <ParamChoice
            label="kernel"
            value={kernel}
            onChange={setKernel}
            options={[
              { value: 'vertical', label: 'vertical edge' },
              { value: 'horizontal', label: 'horizontal edge' },
              { value: 'blur', label: 'blur' },
              { value: 'sharpen', label: 'sharpen' },
            ]}
          />
          <ParamChoice
            label="padding p"
            value={padding}
            onChange={setPadding}
            options={[
              { value: '0', label: '0' },
              { value: '1', label: '1' },
            ]}
          />
          <ParamChoice
            label="stride s"
            value={stride}
            onChange={setStride}
            options={[
              { value: '1', label: '1' },
              { value: '2', label: '2' },
            ]}
          />
          <ParamSlider label="output row i" param={row} format={(v) => String(v)} withArrows />
          <ParamSlider label="output column j" param={col} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="output size ⌊(n + 2p − k)/s⌋ + 1" value={`⌊(${N} + ${2 * p} − ${K})/${s}⌋ + 1 = ${m}`} />
          <Readout label={`y[${i}, ${j}]`} value={formatNumber(terms.reduce((a, b) => a + b, 0))} />
          <Readout label="weights" value={`${K * K} (shared by all ${m * m} outputs)`} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Heatmap
          x={range(-p, N - 1 + p)}
          y={range(-p, N - 1 + p)}
          z={input}
          range={[0, 1]}
          xLabel="column"
          yLabel="row"
          valueLabel="pixel"
          overlay={window}
          handles={[
            {
              kind: 'point',
              at: [c0 + 1, r0 + 1],
              onDrag: ([x, y]) => {
                col.set((x - 1 + p) / s)
                row.set((y - 1 + p) / s)
              },
              label: 'window',
            },
          ]}
          height={320}
          ariaLabel="Input image with the kernel window"
        />
        <Heatmap
          x={range(0, m - 1)}
          y={range(0, m - 1)}
          z={output}
          scale={signed ? 'diverging' : 'sequential'}
          range={outRange}
          xLabel="column j"
          yLabel="row i"
          valueLabel="output"
          marker={[j, i]}
          height={320}
          ariaLabel="Output feature map"
        />
      </div>
    </Interactive>
  )
}
