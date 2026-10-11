import { useContext, useMemo, type ReactNode } from 'react'
import {
  Bars,
  Curve,
  Figure,
  formatNumber,
  FrameContext,
  int,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { blobs } from 'aifn-methods/data/synthetic'
import { kNearestNeighbours } from 'aifn-methods/learning/neighbours'
import { crossValidate, groupKFold, kFold } from 'aifn-compute/learning/validate'
import { accuracy } from 'aifn-compute/learning/metrics'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, slice, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'

const PANEL = 300
const SPLITS = ['random 5-fold', 'grouped 5-fold']

/** Plots inside a Figure take the frame's height; this gives the plots inside it a fixed height instead. */
function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

/** Rows that come in groups: random cross-validation scores a model on groups it has seen, grouped on new groups. */
export function GroupedSplit() {
  const state = useFigureState({
    groups: int(30, { ge: 6, le: 100, suggestions: [10, 30, 60], label: 'groups (e.g. patients)' }),
    rows: int(10, { ge: 2, le: 50, suggestions: [2, 10, 30], label: 'rows per group' }),
    spread: slider(0.2, 5, 1, { step: 0.1, label: 'spread of rows within a group' }),
    k: int(5, { ge: 1, le: 25, suggestions: [1, 5, 15], label: 'neighbours k' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  })
  const { groups, rows, spread, k, seed } = state
  const view = useMemo(() => {
    const d = blobs(stream(`data-setups/${seed}`), {
      n: Array.from({ length: groups }, () => rows),
      centers: groups,
      sd: spread,
    })
    const group = Array.from(toFlat(d.y as Tensor))
    // The label is a property of the group, unrelated to where the group lies: odd groups are class 1.
    const label = group.map((g) => g % 2)
    const data = {
      kind: 'dataset' as const,
      x: d.x,
      y: fromData(Float64Array.from(label), [label.length]),
      groups: group,
    }
    const model = kNearestNeighbours({ k })
    const options = { stream: stream(`data-setups/cv/${seed}`) }
    const random = crossValidate(model, data, kFold({ k: 5, shuffle: true }), [accuracy], options).mean.accuracy
    const grouped = crossValidate(model, data, groupKFold({ k: 5 }), [accuracy], options).mean.accuracy
    const x1 = Array.from(toFlat(slice(d.x, null, 0) as Tensor))
    const x2 = Array.from(toFlat(slice(d.x, null, 1) as Tensor))
    const pick = (c: number) => ({
      x: x1.filter((_, i) => label[i] === c),
      y: x2.filter((_, i) => label[i] === c),
    })
    return { random, grouped, class0: pick(0), class1: pick(1) }
  }, [groups, rows, spread, k, seed])
  const px = useAxis({ label: 'x₁' })
  const py = useAxis({ label: 'x₂' })
  const sx = useAxis({ label: 'cross-validation', categories: SPLITS })
  const sy = useAxis({ label: 'accuracy', range: [0, 1] })
  return (
    <Figure
      title="Random against grouped cross-validation"
      state={state}
      hoverReadout={false}
      caption="Each group is a cloud of rows around its own centre, as repeated measurements of one patient would be. The class is a property of the group, assigned without regard to where the group lies, so a model can predict the class of a new group no better than chance (50%, dashed). Random 5-fold cross-validation puts rows of every group on both sides of the split, and k-nearest neighbours recognises the group from its own rows: the score is high. Grouped 5-fold cross-validation keeps each group in one fold and measures what the model does for a new group. Tight groups and few neighbours widen the gap; with two rows per group or a wide spread it narrows."
      readouts={
        <>
          <Readout label="random 5-fold accuracy" value={formatNumber(view.random)} />
          <Readout label="grouped 5-fold accuracy" value={formatNumber(view.grouped)} />
          <Readout label="rows" value={String(groups * rows)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <FixedHeight height={PANEL}>
          <Plot x={px} y={py} title="Rows, coloured by class">
            <Points name="class 0" x={view.class0.x} y={view.class0.y} slot={0} />
            <Points name="class 1" x={view.class1.x} y={view.class1.y} slot={1} />
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={sx} y={sy} title="Estimated accuracy">
            <Bars name="estimate" x={[0, 1]} y={[view.random, view.grouped]} width={0.5} slot={2} />
            <Curve name="chance" x={[-0.5, 1.5]} y={[0.5, 0.5]} muted dashed />
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
