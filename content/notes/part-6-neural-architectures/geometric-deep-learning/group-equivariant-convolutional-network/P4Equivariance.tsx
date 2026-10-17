import { useMemo } from 'react'
import { choice, Figure, formatNumber, int, Plot, Raster, Readout, useAxis, useFigureState } from 'aifn-render'
import { normals, stream } from 'aifn-compute/foundation/random'
import { abs, max, sub, tensor, toArray, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { conv2d } from 'aifn-compute/nn/functional'
import { orientationPool, p4Act, p4GroupConv, p4Lift, rot90 } from 'aifn-compute/nn/equivariant'
import { FixedHeight } from '../_shared/figure-helpers'

type Pattern = 'gamma' | 'arrow' | 'random'

/** The worked example's 5 × 5 image, and two 9 × 9 patterns with no rotational symmetry. */
const PATTERNS: Record<Exclude<Pattern, 'random'>, number[][]> = {
  gamma: [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0],
  ],
  arrow: [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
  ],
}
const EDGE = [
  [1, 1, 1],
  [0, 0, 0],
  [-1, -1, -1],
]
const ORIENTATIONS = ['0°', '90°', '180°', '270°']

const image4 = (t: Value) => (toArray(t as Tensor) as number[][][][])[0][0]
const maps5 = (t: Value) => (toArray(t as Tensor) as number[][][][][])[0][0]
const err = (a: Value, b: Value) => max(abs(sub(a, b))) as number
const bound = (rows: number[][][]) => Math.max(1e-9, ...rows.flat(2).map(Math.abs))

/**
 * A p4 lifting layer and one group-convolution layer on a small image: rotating the input by a quarter turn rotates
 * every orientation map and moves it one orientation along.
 */
export function P4Equivariance() {
  const state = useFigureState({
    pattern: choice(
      [
        { value: 'gamma', label: 'Γ, 5 × 5 (worked example)' },
        { value: 'arrow', label: 'bent arrow, 9 × 9' },
        { value: 'random', label: 'random, 9 × 9' },
      ] as const,
      'gamma',
      { label: 'image' },
    ),
    filter: choice(
      [
        { value: 'edge', label: 'edge detector (worked example)' },
        { value: 'random', label: 'random 3 × 3' },
      ] as const,
      'edge',
      { label: 'lifting filter' },
    ),
    turns: choice(
      [
        { value: 0, label: '0°' },
        { value: 1, label: '90°' },
        { value: 2, label: '180°' },
        { value: 3, label: '270°' },
      ] as const,
      0,
      { label: 'rotate the input by' },
    ),
    seed: int(1, { ge: 1, le: 999, label: 'seed (random image, filters)' }),
  })
  const { pattern, filter, turns, seed } = state

  const r = useMemo(() => {
    const image =
      pattern === 'random'
        ? tensor([
            [
              (toArray(normals(stream(`p4/image/${seed}`), [9, 9])) as number[][]).map((row) =>
                row.map((v) => (v > 0.6 ? 1 : 0)),
              ),
            ],
          ])
        : tensor([[PATTERNS[pattern]]])
    const w1 = filter === 'edge' ? tensor([[EDGE]]) : normals(stream(`p4/lift/${seed}`), [1, 1, 3, 3])
    const w2 = normals(stream(`p4/group/${seed}`), [1, 1, 4, 3, 3], 0, 0.5)
    const input = rot90(image, turns)
    const lift = (v: Value) => p4Lift(v, w1, { padding: 1 })
    const group = (v: Value) => p4GroupConv(lift(v), w2, { padding: 1 })
    const plain = (v: Value) => conv2d(v, w1, { padding: 1 })
    return {
      input: image4(input),
      lifted: maps5(lift(input)),
      grouped: maps5(group(input)),
      pooled: image4(orientationPool(group(input))),
      bounds: [bound(maps5(lift(image))), bound(maps5(group(image)))],
      liftError: err(lift(input), p4Act(lift(image), turns)),
      groupError: err(group(input), p4Act(group(image), turns)),
      plainError: err(plain(input), rot90(plain(image), turns)),
    }
  }, [pattern, filter, turns, seed])

  const n = r.input.length
  const idx = Array.from({ length: n }, (_, i) => i)
  const ax = useAxis({ label: '', range: [-0.5, n - 0.5], nice: false, zoom: false })
  const ay = useAxis({ label: '', range: [-0.5, n - 0.5], nice: false, zoom: false, inverse: true, equal: ax })
  const panel = (
    key: string,
    title: string,
    z: number[][],
    range: [number, number],
    scale: 'diverging' | 'sequential',
  ) => (
    <Plot key={key} x={ax} y={ay} title={title} bare>
      <Raster x={idx} y={idx} z={z} scale={scale} range={range} colorBar={false} valueLabel="value" />
    </Plot>
  )
  const [b1, b2] = r.bounds

  return (
    <Figure
      title="Rotating the input rotates and shifts the orientation maps"
      state={state}
      defaultSize="L"
      caption="Top row: the input, rotated by the chosen angle, and the four orientation maps of a p4 lifting layer, which correlates the image with its filter turned by 0°, 90°, 180° and 270° (zero padding of 1 keeps the size). Bottom row: the maximum over orientations of one output channel of a p4 group-convolution layer applied to the lifted maps, with random filters on the group, then that channel's four orientation maps. Red is positive, blue negative, on a scale fixed for each row. Rotate the input by 90°: every map turns by 90° and moves one column to the right, the last column wrapping to the first. The orientation-pooled map only turns. A plain convolution with the same filter does not follow the rotation unless the filter has the image's rotational symmetry; the readouts give the three equivariance errors."
      readouts={
        <>
          <Readout label="lifting: max |L(rot x) − rot·L(x)|" value={formatNumber(r.liftError)} />
          <Readout label="group conv: max |G(rot x) − rot·G(x)|" value={formatNumber(r.groupError)} />
          <Readout label="plain conv: max |conv(rot x) − rot conv(x)|" value={formatNumber(r.plainError)} />
        </>
      }
    >
      <FixedHeight height={140}>
        <div className="grid grid-cols-3 gap-2 md:grid-cols-5">
          {panel('input', 'input', r.input, [0, 1], 'sequential')}
          {r.lifted.map((z, k) => panel(`lift-${k}`, `lifted, ${ORIENTATIONS[k]}`, z, [-b1, b1], 'diverging'))}
          {panel('pooled', 'max over orientations', r.pooled, [-b2, b2], 'diverging')}
          {r.grouped.map((z, k) => panel(`group-${k}`, `group conv, ${ORIENTATIONS[k]}`, z, [-b2, b2], 'diverging'))}
        </div>
      </FixedHeight>
    </Figure>
  )
}
