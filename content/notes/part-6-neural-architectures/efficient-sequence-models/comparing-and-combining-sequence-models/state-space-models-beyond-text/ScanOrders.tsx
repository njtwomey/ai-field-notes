import { useMemo } from 'react'
import {
  Bars,
  choice,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Plots,
  Points,
  Raster,
  Readout,
  Segments,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import {
  neighbourPairs,
  routePositions,
  scanOrder,
  sequenceDistance,
  visibleTo,
  type ScanRoute,
} from 'aifn-compute/nn/sequence'

const N = 8
const CENTRES = Array.from({ length: N }, (_, i) => i + 0.5)

type SchemeId = 'row' | 'both' | 'snake' | 'cross' | 'hilbert'
const SCHEMES: Record<SchemeId, { label: string; routes: ScanRoute[] }> = {
  row: { label: 'raster, one direction', routes: [{ order: 'row' }] },
  both: {
    label: 'raster, both directions (Vision Mamba)',
    routes: [{ order: 'row' }, { order: 'row', reverse: true }],
  },
  snake: { label: 'snake (boustrophedon)', routes: [{ order: 'snake' }] },
  cross: {
    label: 'rows and columns, both directions (VMamba)',
    routes: [
      { order: 'row' },
      { order: 'row', reverse: true },
      { order: 'column' },
      { order: 'column', reverse: true },
    ],
  },
  hilbert: { label: 'Hilbert curve', routes: [{ order: 'hilbert' }] },
}
const OPTIONS = (Object.keys(SCHEMES) as SchemeId[]).map((id) => ({ value: id, label: SCHEMES[id].label }))
const PAIRS = neighbourPairs(N, N)
const GRID = Array.from({ length: N + 1 }, (_, k) => [
  { from: [k, 0] as const, to: [k, N] as const },
  { from: [0, k] as const, to: [N, k] as const },
]).flat()

/** Scan orders over an 8 × 8 grid of patches: how far apart neighbouring patches land in the sequence. */
export function ScanOrders() {
  const state = useFigureState({
    scheme: choice<SchemeId>(OPTIONS, 'row', { label: 'scan' }),
    col: slider(0, N - 1, 3, { step: 1, onChart: true }),
    row: slider(0, N - 1, 4, { step: 1, onChart: true }),
  })
  const scheme = SCHEMES[state.scheme]
  const col = Math.round(state.col)
  const row = Math.round(state.row)
  const cell = row * N + col

  const pos = useMemo(() => routePositions(scheme.routes, N, N), [scheme])
  const paths = useMemo(() => {
    const kinds = [...new Set(scheme.routes.map((r) => r.order))]
    return kinds.flatMap((kind) => {
      const o = scanOrder(kind, N, N)
      const xy = (c: number) => [(c % N) + 0.5, Math.floor(c / N) + 0.5] as const
      return o.slice(1).map((c, i) => ({ from: xy(o[i]), to: xy(c) }))
    })
  }, [scheme])
  const distances = useMemo(() => PAIRS.map(([a, b]) => sequenceDistance(pos, a, b)), [pos])
  const histogram = useMemo(() => {
    const max = Math.max(...distances)
    const x = Array.from({ length: max }, (_, i) => i + 1)
    return { x, y: x.map((d) => distances.filter((v) => v === d).length) }
  }, [distances])

  const z = useMemo(
    () =>
      CENTRES.map((_, r) =>
        CENTRES.map((_, c) => {
          const other = r * N + c
          return visibleTo(pos, cell, other) ? sequenceDistance(pos, cell, other) : NaN
        }),
      ),
    [pos, cell],
  )
  const visible = z.flat().filter((v) => Number.isFinite(v)).length
  const neighbours = [
    [row - 1, col],
    [row + 1, col],
    [row, col - 1],
    [row, col + 1],
  ]
    .filter(([r, c]) => r >= 0 && r < N && c >= 0 && c < N)
    .map(([r, c]) => sequenceDistance(pos, cell, r * N + c))
  const mean = distances.reduce((a, d) => a + d, 0) / distances.length

  const xAxis = useAxis({ label: 'patch column', range: [0, N] })
  const yAxis = useAxis({ label: 'patch row', range: [0, N], inverse: true, equal: xAxis })
  const dAxis = useAxis({ label: 'sequence distance between neighbouring patches', integer: true })
  const cAxis = useAxis({ label: 'neighbouring pairs', integer: true })

  return (
    <Figure
      title="Scan orders over a grid of patches"
      state={state}
      caption="An 8 × 8 grid of image patches is read as a sequence. Left: the scan path (several overlaid for the four-route scan) and, coloured by sequence distance, the patches that the output at the marked patch can see; blank patches come later on every route and are invisible to it. Drag the marked patch. Right: the sequence distance of all 112 pairs of horizontally or vertically neighbouring patches, the smallest over the scan's routes. The raster order puts every vertical neighbour 8 steps away and lets a patch see only the patches above and to its left. The snake order makes the path continuous but keeps the average distance at 4.5. Running both directions makes every patch visible to every output. The four-route scan of VMamba puts every neighbour 1 step away on one of its routes. The Hilbert curve keeps small blocks together but separates a few neighbours by dozens of steps."
      readouts={
        <>
          <Readout label="mean neighbour distance" value={formatNumber(mean)} />
          <Readout label="largest neighbour distance" value={String(Math.max(...distances))} />
          <Readout label="neighbours at distance 1" value={`${distances.filter((d) => d === 1).length} of 112`} />
          <Readout label="patches the marked output sees" value={`${visible} of 64`} />
          <Readout label="distances to its neighbours" value={neighbours.join(', ')} />
        </>
      }
    >
      <Plots cols={2}>
        <Plot x={xAxis} y={yAxis}>
          <Raster x={CENTRES} y={CENTRES} z={z} scale="sequential" range={[0, 63]} valueLabel="sequence distance" />
          <Segments segments={GRID} muted width={1} />
          <Segments segments={paths} emphasis width={1.5} />
          <Points x={[col + 0.5]} y={[row + 0.5]} emphasis size={10} />
          <Handle
            kind="point"
            at={[col + 0.5, row + 0.5]}
            label="patch"
            onDrag={([x, y]) => {
              state.set('col', Math.floor(x))
              state.set('row', Math.floor(y))
            }}
          />
        </Plot>
        <Plot x={dAxis} y={cAxis}>
          <Bars name="pairs" x={histogram.x} y={histogram.y} slot={0} />
        </Plot>
      </Plots>
    </Figure>
  )
}
