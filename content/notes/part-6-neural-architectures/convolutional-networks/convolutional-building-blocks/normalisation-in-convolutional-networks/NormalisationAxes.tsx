import { useMemo } from 'react'
import {
  choice,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Points,
  Raster,
  Readout,
  Segments,
  setting,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { batchNorm, groupNorm } from 'aifn-compute/nn/layers'
import { tensor, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { FixedHeight } from '../_shared/FixedHeight'

// The [N, C, H, W] = [2, 4, 2, 2] tensor of the note's worked example, channel by channel, pixels in row order.
const EXAMPLE = [
  [1, 3, 5, 7],
  [2, 2, 4, 4],
  [0, 4, 0, 4],
  [6, 6, 6, 10],
  [9, 11, 13, 15],
  [0, 0, 2, 2],
  [3, 3, 5, 5],
  [2, 4, 6, 8],
]
const N = 2
const C = 4
const EPS = 1e-5

type Method = 'batch' | 'layer' | 'instance' | 'group'
const LABELS: Record<Method, string> = {
  batch: 'batch norm: one channel, all examples',
  layer: 'layer norm: one example, all channels',
  instance: 'instance norm: one example, one channel',
  group: 'group norm, G = 2: one example, two channels',
}

/** Grid layout: column 2c + w, row 2n + h, so each example is a strip and each channel a 2 × 2 block. */
const index = (n: number, c: number, h: number, w: number) => ((n * C + c) * 2 + h) * 2 + w
const cell = (x: number, y: number) => ({ n: Math.floor(y / 2), h: y % 2, c: Math.floor(x / 2), w: x % 2 })

function shares(method: Method, a: { n: number; c: number }, b: { n: number; c: number }) {
  if (method === 'batch') return a.c === b.c
  if (method === 'layer') return a.n === b.n
  if (method === 'instance') return a.n === b.n && a.c === b.c
  return a.n === b.n && Math.floor(a.c / 2) === Math.floor(b.c / 2)
}

/**
 * The four normalisations of a convolutional activation tensor differ only in which values share a mean and a
 * variance. Pick a value; the values normalised together with it are marked.
 */
export function NormalisationAxes() {
  const state = useFigureState({
    method: choice<Method>(
      (Object.keys(LABELS) as Method[]).map((value) => ({ value, label: LABELS[value] })),
      'group',
      { label: 'normalisation' },
    ),
    copy: setting(false, 'second example replaced by a copy of the first'),
    x: slider(0, 2 * C - 1, 0, { step: 1, onChart: true }),
    y: slider(0, 2 * N - 1, 0, { step: 1, onChart: true }),
  })
  const method = state.method

  const values = useMemo(() => {
    const rows = state.copy ? [...EXAMPLE.slice(0, C), ...EXAMPLE.slice(0, C)] : EXAMPLE
    return rows.flat()
  }, [state.copy])

  const normalised = useMemo(() => {
    const x = tensor(values, [N, C, 2, 2])
    const groups = { layer: 1, instance: C, group: 2 } as const
    const y: Value =
      method === 'batch'
        ? batchNorm(x, undefined, undefined, { eps: EPS })
        : groupNorm(x, groups[method], undefined, undefined, EPS)
    return toFlat(y as Tensor)
  }, [values, method])

  const grid = (flat: number[]) =>
    Array.from({ length: 2 * N }, (_, y) =>
      Array.from({ length: 2 * C }, (_, x) => {
        const { n, c, h, w } = cell(x, y)
        return flat[index(n, c, h, w)]
      }),
    )

  const pick = cell(state.x, state.y)
  const members = { x: [] as number[], y: [] as number[], v: [] as number[] }
  for (let y = 0; y < 2 * N; y++) {
    for (let x = 0; x < 2 * C; x++) {
      const at = cell(x, y)
      if (shares(method, pick, at)) {
        members.x.push(x)
        members.y.push(y)
        members.v.push(values[index(at.n, at.c, at.h, at.w)])
      }
    }
  }
  const mu = members.v.reduce((a, v) => a + v, 0) / members.v.length
  const variance = members.v.reduce((a, v) => a + (v - mu) ** 2, 0) / members.v.length
  const selected = index(pick.n, pick.c, pick.h, pick.w)

  // Channel blocks and the line between the two examples, drawn over the opaque rasters.
  const lines = useMemo(() => {
    const s: { from: [number, number]; to: [number, number] }[] = []
    for (let c = 0; c <= C; c++) s.push({ from: [2 * c - 0.5, -0.5], to: [2 * c - 0.5, 2 * N - 0.5] })
    for (let n = 0; n <= N; n++) s.push({ from: [-0.5, 2 * n - 0.5], to: [2 * C - 0.5, 2 * n - 0.5] })
    return s
  }, [])

  const xs = Array.from({ length: 2 * C }, (_, i) => i)
  const ys = Array.from({ length: 2 * N }, (_, i) => i)
  const xAxis = useAxis({
    label: 'channel c (2 × 2 pixels each)',
    range: [-0.5, 2 * C - 0.5],
    zoom: false,
    format: (v) => (Number.isInteger(v) && v % 2 === 0 ? `c = ${v / 2}` : ''),
  })
  const yAxis = useAxis({
    label: 'example n (2 rows each)',
    range: [-0.5, 2 * N - 0.5],
    inverse: true,
    zoom: false,
    format: (v) => (Number.isInteger(v) && v % 2 === 0 ? `n = ${v / 2}` : ''),
  })
  return (
    <Figure
      title="Which values share a mean and a variance"
      state={state}
      caption="A [2, 4, 2, 2] activation tensor drawn as a grid: the two examples are the two strips, the four channels the 2 × 2 blocks across. Top: the values, with the set normalised together with the chosen value marked. Bottom: the tensor after normalisation, without scale or shift. Drag the marker to choose a value; switch the method to change the set. Batch normalisation's set runs across both examples, so replacing the second example changes the first example's output; the other three methods do not look at the other example."
      readouts={
        <>
          <Readout label="chosen value" value={`x[${pick.n}, ${pick.c}, ${pick.h}, ${pick.w}] = ${values[selected]}`} />
          <Readout label="values in its set" value={members.v.length} />
          <Readout label="set mean, variance" value={`${formatNumber(mu)}, ${formatNumber(variance)}`} />
          <Readout label="normalised value" value={formatNumber(normalised[selected])} />
        </>
      }
    >
      <FixedHeight height={180}>
        <div className="space-y-3">
          <Plot x={xAxis} y={yAxis} ariaLabel="Activation values with the normalisation set marked">
            <Raster x={xs} y={ys} z={grid(values)} range={[0, 15]} valueLabel="x" />
            <Segments segments={lines} />
            <Points name="normalised together" x={members.x} y={members.y} emphasis size={10} />
            <Handle {...state.handle(['x', 'y'], { label: 'value' })} />
          </Plot>
          <Plot x={xAxis} y={yAxis} ariaLabel="Normalised activations">
            <Raster x={xs} y={ys} z={grid(normalised)} scale="diverging" range={[-2.5, 2.5]} valueLabel="normalised" />
            <Segments segments={lines} />
          </Plot>
        </div>
      </FixedHeight>
    </Figure>
  )
}
