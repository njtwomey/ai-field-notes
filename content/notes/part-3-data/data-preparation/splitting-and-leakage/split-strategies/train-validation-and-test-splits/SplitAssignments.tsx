import { useContext, useMemo, type ReactNode } from 'react'
import { choice, Figure, FrameContext, int, Plot, Raster, Readout, useAxis, useFigureState } from 'aifn-render'
import { bernoulli, child, integers, stream } from 'aifn-compute/foundation/random'
import { toFlat, toRows } from 'aifn-compute/foundation/tensor'
import {
  assignment,
  expandingWindow,
  groupKFold,
  kFold,
  stratifiedKFold,
  type Splitter,
} from 'aifn-compute/learning/validate'

const N = 48
const ENTITIES = 8
const ROWS = Array.from({ length: N }, (_, i) => i)

const SPLITTERS = [
  { value: 'random', label: 'random k-fold' },
  { value: 'stratified', label: 'stratified k-fold' },
  { value: 'grouped', label: 'grouped k-fold' },
  { value: 'temporal', label: 'expanding window' },
] as const
type SplitterName = (typeof SPLITTERS)[number]['value']

function makeSplitter(name: SplitterName, k: number, gap: number): Splitter {
  switch (name) {
    case 'random':
      return kFold({ k, shuffle: true })
    case 'stratified':
      return stratifiedKFold({ k, shuffle: true })
    case 'grouped':
      return groupKFold({ k })
    case 'temporal':
      return expandingWindow({ splits: k, gap })
  }
}

/** Plots inside a Figure take the frame's height; this gives the plots inside it a fixed height instead. */
function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

// 48 rows in time order: a binary label, positive about one time in four, and one of eight entities per row.
const LABELS = toFlat(bernoulli(child(stream('split-assignments'), 'y'), 0.25, { shape: [N] }) as never)
const ENTITY = toFlat(integers(child(stream('split-assignments'), 'entity'), ENTITIES, { shape: [N] }))

/** Which rows each split trains and tests on, for four splitters. */
export function SplitAssignments() {
  const state = useFigureState({
    splitter: choice(SPLITTERS, 'random', { label: 'splitter' }),
    k: int(4, { ge: 2, le: 8, suggestions: [3, 4, 5], label: 'splits k' }),
    gap: int(0, { ge: 0, le: 6, suggestions: [0, 2, 4], label: 'gap (expanding window)' }),
  })
  const { splitter, k, gap } = state
  const view = useMemo(() => {
    const splits = makeSplitter(splitter, k, gap).split(
      { n: N, y: Array.from(LABELS), groups: Array.from(ENTITY) },
      stream('split-assignments/splits'),
    )
    const z = toRows(assignment(splits, N)) as number[][]
    // The share of positives in each test fold, and how many entities appear on both sides of a split.
    const positives = splits.map((s) => {
      const test = Array.from(toFlat(s.test))
      return test.filter((i) => LABELS[i] === 1).length / test.length
    })
    const shared = splits.map((s) => {
      const train = new Set(Array.from(toFlat(s.train)).map((i) => ENTITY[i]))
      return new Set(
        Array.from(toFlat(s.test))
          .map((i) => ENTITY[i])
          .filter((e) => train.has(e)),
      ).size
    })
    return { z, splits: splits.map((_, i) => i + 1), positives, shared }
  }, [splitter, k, gap])
  const rx = useAxis({ label: 'row, in time order', range: [-0.5, N - 0.5], nice: false })
  const sy = useAxis({ label: 'split', range: [0.5, view.splits.length + 0.5], nice: false, integer: true })
  const ex = useAxis({ label: 'row, in time order', range: [-0.5, N - 0.5], nice: false })
  const ey = useAxis({ label: 'entity', range: [-0.5, ENTITIES - 0.5], nice: false, integer: true })
  const entityGrid = useMemo(
    () =>
      Array.from({ length: ENTITIES }, (_, e) => ROWS.map((i) => (ENTITY[i] === e ? (LABELS[i] === 1 ? 3 : 2) : NaN))),
    [],
  )
  const range = (v: number[]) => `${Math.round(100 * Math.min(...v))}–${Math.round(100 * Math.max(...v))}%`
  return (
    <Figure
      title="Which rows each split trains and tests on"
      state={state}
      hoverReadout={false}
      caption="Top: each row of the grid is one split, and each column one of 48 rows of data in time order; a cell is a training row (blue), a test row (orange) or unused (empty). Bottom: the entity each row belongs to (a patient, a user, a device), coloured by the row's label, positive about one time in four. Random k-fold mixes every entity and every time into every fold. Stratified k-fold keeps the share of positives in each test fold close to the overall share. Grouped k-fold keeps each entity wholly on one side, so no entity appears on both sides of a split. The expanding window trains only on the past; a gap leaves unused rows between the training and test blocks, so labels that look ahead in time cannot reach across."
      readouts={
        <>
          <Readout label="positives per test fold" value={range(view.positives)} />
          <Readout label="entities on both sides, per split" value={view.shared.join(', ')} />
        </>
      }
    >
      <FixedHeight height={70 + 26 * view.splits.length}>
        <Plot x={rx} y={sy} title="splits">
          <Raster
            name="assignment"
            x={ROWS}
            y={view.splits}
            z={view.z}
            scale="categorical"
            categoryNames={['train', 'test']}
          />
        </Plot>
      </FixedHeight>
      <FixedHeight height={70 + 22 * ENTITIES}>
        <Plot x={ex} y={ey} title="entities and labels">
          <Raster
            name="entity"
            x={ROWS}
            y={Array.from({ length: ENTITIES }, (_, e) => e)}
            z={entityGrid}
            scale="categorical"
            categoryNames={['', '', 'negative label', 'positive label']}
          />
        </Plot>
      </FixedHeight>
    </Figure>
  )
}
