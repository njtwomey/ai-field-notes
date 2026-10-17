import { useMemo, useState } from 'react'
import {
  choice,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Raster,
  Readout,
  Segments,
  useAxis,
  useFigureState,
  Vectors,
  when,
  type Vec2,
} from 'aifn-render'
import { deformableConvolutionAt } from 'aifn-methods/neural/convolutional'

const K = 3
type KernelName = 'mean' | 'line'
/** The mean of the nine samples, and a line detector: twice the middle row minus the outer rows, over 3. */
const KERNELS: Record<KernelName, number[]> = {
  mean: new Array(K * K).fill(1 / (K * K)),
  line: [-1, -1, -1, 2, 2, 2, -1, -1, -1].map((v) => v / 3),
}
const KERNEL_OPTIONS = [
  { value: 'mean' as const, label: 'mean of the nine samples' },
  { value: 'line' as const, label: 'line detector (middle row minus outer rows)' },
]
/** Regular offsets p_n = (u − 1, v − 1), row-major over the kernel. */
const GRID: Vec2[] = Array.from({ length: K * K }, (_, n) => [Math.floor(n / K) - 1, (n % K) - 1])
const ZEROS: Vec2[] = GRID.map(() => [0, 0])

/** The worked example: a 5 × 5 image with a bright anti-diagonal ridge. */
const RIDGE = { size: 5, pixels: [0, 0, 1, 4, 8, 0, 1, 4, 8, 4, 1, 4, 8, 4, 1, 4, 8, 4, 1, 0, 8, 4, 1, 0, 0] }
const RIDGE_OFFSETS: Vec2[] = [
  [0.4, 0.8],
  [0, 0.5],
  [0, 0],
  [0.5, 0],
  [0, 0],
  [-0.5, 0],
  [0, 0],
  [0, -0.5],
  [-0.8, -0.4],
]

/** A 15 × 15 image of a bright ring of radius 4√2 about (7, 7), so it passes through (11, 11). */
const RING_SIZE = 15
const RING_CENTRE: Vec2 = [7, 7]
const RING_RADIUS = 4 * Math.SQRT2
const RING = {
  size: RING_SIZE,
  pixels: Array.from({ length: RING_SIZE * RING_SIZE }, (_, q) => {
    const rho = Math.hypot(Math.floor(q / RING_SIZE) - RING_CENTRE[0], (q % RING_SIZE) - RING_CENTRE[1])
    return Math.exp(-((rho - RING_RADIUS) ** 2) / (2 * 0.6 ** 2))
  }),
}

type Preset = 'regular' | 'worked' | 'dilated' | 'scaled' | 'rotated' | 'bent'

/** The offsets Δp_n of a preset at output position p0. */
function presetOffsets(preset: Preset, p0: Vec2): Vec2[] {
  const rotate = (q: Vec2, a: number): Vec2 => [
    q[0] * Math.cos(a) + q[1] * Math.sin(a),
    -q[0] * Math.sin(a) + q[1] * Math.cos(a),
  ]
  // p0 relative to the ring's centre, in (row, column), and its distance ρ0 from it.
  const rel: Vec2 = [p0[0] - RING_CENTRE[0], p0[1] - RING_CENTRE[1]]
  const rho = Math.hypot(rel[0], rel[1]) || 1
  if (preset === 'bent') {
    // Kernel column b runs along the ring (angle φ0 ± b/ρ0) and kernel row a across it (radius ρ0 ± a), so the
    // middle row follows the ring. Of the four sign choices, take the one closest to the regular grid.
    const options: Vec2[][] = []
    for (const sa of [1, -1]) {
      for (const sb of [1, -1]) {
        options.push(
          GRID.map(([a, b]): Vec2 => {
            const angle = Math.atan2(rel[0], rel[1]) + (sb * b) / rho
            const radius = rho + sa * a
            return [
              RING_CENTRE[0] + radius * Math.sin(angle) - p0[0] - a,
              RING_CENTRE[1] + radius * Math.cos(angle) - p0[1] - b,
            ]
          }),
        )
      }
    }
    const length = (o: Vec2[]) => o.reduce((acc, d) => acc + Math.hypot(d[0], d[1]), 0)
    return options.reduce((best, o) => (length(o) < length(best) ? o : best))
  }
  return GRID.map((q, n): Vec2 => {
    let target: Vec2 = q
    if (preset === 'worked') return RIDGE_OFFSETS[n]
    if (preset === 'dilated') target = [2 * q[0], 2 * q[1]]
    if (preset === 'scaled') target = [1.5 * q[0], 1.5 * q[1]]
    if (preset === 'rotated') target = rotate(q, -Math.PI / 4)
    return [target[0] - q[0], target[1] - q[1]]
  })
}

const range = (n: number) => Array.from({ length: n }, (_, i) => i)

/**
 * A 3 × 3 kernel at one output position, its regular sampling grid and its deformed one, with every tap
 * draggable; the tap last dragged shows its four bilinear neighbours and weights.
 */
export function DeformableSampling() {
  const state = useFigureState({
    image: choice<'ridge' | 'ring'>(
      [
        { value: 'ridge', label: 'diagonal ridge, 5 × 5 (worked example)' },
        { value: 'ring', label: 'ring, 15 × 15' },
      ],
      'ridge',
      { label: 'image' },
    ),
    ridgePreset: choice<Preset>(
      [
        { value: 'regular', label: 'regular grid' },
        { value: 'worked', label: 'worked-example offsets' },
        { value: 'dilated', label: 'dilation 2' },
        { value: 'scaled', label: 'scaled by 1.5' },
        { value: 'rotated', label: 'rotated by 45°' },
      ],
      'worked',
      { label: 'offsets', when: when('image', 'ridge') },
    ),
    ringPreset: choice<Preset>(
      [
        { value: 'regular', label: 'regular grid' },
        { value: 'dilated', label: 'dilation 2' },
        { value: 'scaled', label: 'scaled by 1.5' },
        { value: 'rotated', label: 'rotated by 45°' },
        { value: 'bent', label: 'bent along the ring' },
      ],
      'bent',
      { label: 'offsets', when: when('image', 'ring') },
    ),
    ridgeKernel: choice<KernelName>(KERNEL_OPTIONS, 'mean', { label: 'kernel', when: when('image', 'ridge') }),
    ringKernel: choice<KernelName>(KERNEL_OPTIONS, 'line', { label: 'kernel', when: when('image', 'ring') }),
    row: int(11, { ge: 1, le: RING_SIZE - 2, label: 'output row', when: when('image', 'ring') }),
    col: int(11, { ge: 1, le: RING_SIZE - 2, label: 'output column', when: when('image', 'ring') }),
  })
  const ridge = state.image === 'ridge'
  const image = ridge ? RIDGE : RING
  const preset = ridge ? state.ridgePreset : state.ringPreset
  const kernel = KERNELS[ridge ? state.ridgeKernel : state.ringKernel]
  const p0: Vec2 = ridge ? [2, 2] : [state.row, state.col]
  const key = `${state.image}:${preset}:${p0[0]}:${p0[1]}`

  // Dragged offsets belong to one image, preset and position; any change of those opens at the preset again.
  const [dragged, setDragged] = useState<{ key: string; offsets: Vec2[] } | null>(null)
  const [selected, setSelected] = useState(0)
  const [r0, c0] = p0
  const offsets = useMemo(
    () => (dragged !== null && dragged.key === key ? dragged.offsets : presetOffsets(preset, [r0, c0])),
    [dragged, key, preset, r0, c0],
  )

  const n = image.size
  const z = useMemo(() => range(n).map((r) => range(n).map((c) => image.pixels[r * n + c])), [image, n])
  const regular = useMemo(
    () => deformableConvolutionAt(image.pixels, n, n, kernel, K, [r0, c0], ZEROS),
    [image, n, kernel, r0, c0],
  )
  const deformed = useMemo(
    () => deformableConvolutionAt(image.pixels, n, n, kernel, K, [r0, c0], offsets),
    [image, n, kernel, r0, c0, offsets],
  )

  const tap = deformed.taps[selected]
  // Effective dilation (Dai et al., table 2): the mean distance between horizontally or vertically adjacent taps.
  const pairs: [number, number][] = []
  for (let u = 0; u < K; u++) {
    for (let v = 0; v < K; v++) {
      if (v + 1 < K) pairs.push([u * K + v, u * K + v + 1])
      if (u + 1 < K) pairs.push([u * K + v, (u + 1) * K + v])
    }
  }
  const dilation =
    pairs.reduce((acc, [a, b]) => {
      const pa = deformed.taps[a].position
      const pb = deformed.taps[b].position
      return acc + Math.hypot(pa[0] - pb[0], pa[1] - pb[1])
    }, 0) / pairs.length

  const drag = (index: number, [x, y]: Vec2) => {
    const clamp = (v: number) => Math.min(Math.max(v, -0.5), n - 0.5)
    const regularPosition = deformed.taps[index].regular
    const next = offsets.map((o, m): Vec2 =>
      m === index ? [clamp(y) - regularPosition[0], clamp(x) - regularPosition[1]] : o,
    )
    setDragged({ key, offsets: next })
    setSelected(index)
  }

  const xAxis = useAxis({ label: 'column', range: [-0.5, n - 0.5], key: n })
  const yAxis = useAxis({ label: 'row', range: [-0.5, n - 0.5], inverse: true, equal: xAxis, key: n })
  // Plot coordinates are (column, row).
  const xy = (p: readonly [number, number]): Vec2 => [p[1], p[0]]
  const neighbours = tap.sample.taps.filter((t) => t.row >= 0 && t.row < n && t.col >= 0 && t.col < n)
  return (
    <Figure
      title="Sampling positions of a deformable 3 × 3 kernel"
      state={state}
      caption="The image with the regular 3 × 3 grid around one output position (grey) and the deformed sampling positions (coloured), each joined to its regular position by its offset. Drag any coloured point to change its offset; the tap you dragged last shows its four integer neighbours with their bilinear weights. With the mean kernel the output is the average brightness the taps see: on the diagonal ridge the worked-example offsets pull six taps towards the ridge and raise the output from 4.67 to 6.43. The line detector responds when its middle row lies on a bright line and its outer rows do not: on the ridge, rotating the grid by 45° raises it from 2.0 to 8.2. On the ring, type other output positions on the ring, such as (3, 11) or (11, 3): the fixed rotation fits only at (3, 3) and (11, 11), where the ring runs along the rotated rows, while the grid bent along the ring keeps its middle row on the ring everywhere. A deformable layer predicts a different set of offsets at every position, which is what the bent grid needs."
      readouts={
        <>
          <Readout label="regular output y(p0)" value={formatNumber(regular.value)} />
          <Readout label="deformable output y(p0)" value={formatNumber(deformed.value)} />
          <Readout label="effective dilation" value={formatNumber(dilation)} />
          <Readout
            label={`tap ${selected + 1}: offset Δp`}
            value={`(${formatNumber(offsets[selected][0])}, ${formatNumber(offsets[selected][1])})`}
          />
          <Readout
            label={`tap ${selected + 1}: x at (${formatNumber(tap.position[0])}, ${formatNumber(tap.position[1])})`}
            value={formatNumber(tap.sample.value)}
          />
          <Readout label="bilinear weights" value={tap.sample.taps.map((t) => formatNumber(t.weight)).join(', ')} />
          <Readout
            label="∂y / ∂Δp (row, column)"
            value={`(${formatNumber(deformed.offsetGradient[selected][0])}, ${formatNumber(deformed.offsetGradient[selected][1])})`}
          />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} ariaLabel="Image with regular and deformed sampling positions">
        <Raster x={range(n)} y={range(n)} z={z} valueLabel="pixel" />
        <Points
          name="regular grid p0 + p_n"
          x={deformed.taps.map((t) => t.regular[1])}
          y={deformed.taps.map((t) => t.regular[0])}
          size={7}
          muted
        />
        <Vectors
          vectors={deformed.taps
            .filter((t) => Math.hypot(t.position[0] - t.regular[0], t.position[1] - t.regular[1]) > 0.05)
            .map((t) => ({ from: xy(t.regular), to: xy(t.position), head: 6, muted: true }))}
        />
        {neighbours.length > 0 && (
          <Segments
            segments={neighbours.map((t) => ({ from: xy(tap.position), to: [t.col, t.row] as [number, number] }))}
            dashed
          />
        )}
        {neighbours.length > 0 && (
          <Points
            name="bilinear neighbours of the selected tap"
            x={neighbours.map((t) => t.col)}
            y={neighbours.map((t) => t.row)}
            labels={neighbours.map((t) => formatNumber(t.weight))}
            size={6}
            emphasis
          />
        )}
        <Points
          name="sampled positions p0 + p_n + Δp_n"
          x={deformed.taps.map((t) => t.position[1])}
          y={deformed.taps.map((t) => t.position[0])}
          size={9}
          slot={1}
        />
        {deformed.taps.map((t) => (
          <Handle key={t.index} kind="point" at={xy(t.position)} onDrag={(p) => drag(t.index, p)} />
        ))}
      </Plot>
    </Figure>
  )
}
