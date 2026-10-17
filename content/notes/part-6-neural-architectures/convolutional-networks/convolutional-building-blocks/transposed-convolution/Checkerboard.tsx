import { useMemo } from 'react'
import {
  choice,
  Figure,
  formatNumber,
  int,
  Plot,
  Raster,
  Readout,
  Segments,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { conv, convTranspose } from 'aifn-compute/foundation/convolution'
import { ones, shapeOfValue, tensor, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { FixedHeight } from '../_shared/FixedHeight'

/** Length of the 1-D input in the matrix panel, and side of the constant 2-D input in the overlap panel. */
const N1 = 4
const N2 = 8

type Method = 'transposed' | 'resize'

/**
 * A transposed convolution as a matrix, built column by column from one-hot inputs, and the 2-D output of a
 * transposed convolution (or of resize-then-convolve) on a constant input, which shows how many kernel taps land on
 * each output pixel.
 */
export function Checkerboard() {
  const state = useFigureState({
    kernel: int(3, { ge: 1, le: 7, label: 'kernel k' }),
    stride: int(2, { ge: 1, le: 4, label: 'stride s' }),
    method: choice<Method>(
      [
        { value: 'transposed', label: 'transposed convolution' },
        { value: 'resize', label: 'nearest-neighbour resize, then convolve' },
      ],
      'transposed',
      { label: 'upsampler (right panel)' },
    ),
  })
  const k = state.kernel
  const s = state.stride

  // Kernel taps 1, 2, …, k so each tap's place in the matrix can be read off.
  const matrix = useMemo(() => {
    const w = tensor(Array.from({ length: k }, (_, u) => u + 1))
    const columns = Array.from({ length: N1 }, (_, i) => {
      const e = Array.from({ length: N1 }, (_, j) => (j === i ? 1 : 0))
      return toFlat(convTranspose(tensor(e), w, { stride: s, flip: false }) as Tensor)
    })
    const m = columns[0].length
    return Array.from({ length: m }, (_, r) => columns.map((col) => col[r]))
  }, [k, s])

  const overlap = useMemo(() => {
    const x = ones([1, 1, N2, N2])
    const w = ones([1, 1, k, k])
    let y: Value
    if (state.method === 'transposed') {
      y = convTranspose(x, w, { stride: s, flip: false })
    } else {
      // Nearest-neighbour upsampling is itself a transposed convolution with an s × s kernel of ones and stride s.
      const up = convTranspose(x, ones([1, 1, s, s]), { stride: s, flip: false })
      y = conv(up, w, { padding: 'same', flip: false })
    }
    // The [1, 1, H, W] output as H rows of W values.
    const [, , H, W] = shapeOfValue(y)
    const flat = toFlat(y as Tensor)
    return Array.from({ length: H }, (_, r) => flat.slice(r * W, (r + 1) * W))
  }, [k, s, state.method])

  const side = overlap.length
  // Interior: away from the borders by the kernel size, where every position has its full set of contributions.
  const interior = overlap.slice(k, side - k).flatMap((row) => row.slice(k, side - k))
  const lo = interior.length ? Math.min(...interior) : NaN
  const hi = interior.length ? Math.max(...interior) : NaN
  const m = matrix.length

  const cells = (n: number) => Array.from({ length: n }, (_, i) => i)
  const gridLines = useMemo(() => {
    const segs: { from: [number, number]; to: [number, number] }[] = []
    for (let i = 0; i <= N1; i++) segs.push({ from: [i - 0.5, -0.5], to: [i - 0.5, m - 0.5] })
    for (let r = 0; r <= m; r++) segs.push({ from: [-0.5, r - 0.5], to: [N1 - 0.5, r - 0.5] })
    return segs
  }, [m])

  const mx = useAxis({ label: 'input position i', range: [-0.5, N1 - 0.5], integer: true, zoom: false })
  const my = useAxis({ label: 'output position', range: [-0.5, m - 0.5], integer: true, inverse: true, zoom: false })
  const ox = useAxis({ label: 'output column', range: [-0.5, side - 0.5], zoom: false })
  const oy = useAxis({ label: 'output row', range: [-0.5, side - 0.5], inverse: true, zoom: false })
  return (
    <Figure
      title="Transposed convolution: the matrix and the checkerboard"
      state={state}
      caption="Left: the matrix of a 1-D transposed convolution from 4 inputs, with kernel taps numbered 1 to k. Column i is the kernel, placed s rows further down than column i − 1; the transpose of this matrix is the strided convolution it inverts the shape of. Right: the output of upsampling a constant 8 × 8 image of ones with a k × k kernel of ones. With a transposed convolution, each output pixel's value is the number of kernel taps that land on it: when k is not a multiple of s, neighbouring pixels receive different counts and a checkerboard appears. Resize-then-convolve gives every interior pixel the same count. Set k = 4, s = 2 for an even transposed upsampler, or k = 3, s = 2 for the checkerboard."
      readouts={
        <>
          <Readout label="1-D output length (n − 1)s + k" value={`(${N1} − 1) · ${s} + ${k} = ${m}`} />
          <Readout label="2-D output side" value={side} />
          <Readout label="interior values, min to max" value={`${formatNumber(lo)} to ${formatNumber(hi)}`} />
          <Readout label="uneven overlap" value={hi > lo ? `yes, ratio ${formatNumber(hi / lo)}` : 'no'} />
        </>
      }
    >
      <FixedHeight height={320}>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-[2fr_3fr]">
          <Plot x={mx} y={my} ariaLabel="Matrix of the transposed convolution">
            <Raster x={cells(N1)} y={cells(m)} z={matrix} range={[0, 7]} valueLabel="weight" colorBar={false} />
            <Segments segments={gridLines} />
          </Plot>
          <Plot x={ox} y={oy} ariaLabel="Output of upsampling a constant image">
            <Raster x={cells(side)} y={cells(side)} z={overlap} valueLabel="output" />
          </Plot>
        </div>
      </FixedHeight>
    </Figure>
  )
}
