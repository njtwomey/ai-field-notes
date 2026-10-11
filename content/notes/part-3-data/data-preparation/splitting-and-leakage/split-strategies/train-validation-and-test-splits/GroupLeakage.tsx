import { useContext, useMemo, type ReactNode } from 'react'
import {
  Bars,
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
import { child, normals, stream, type Stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat } from 'aifn-compute/foundation/tensor'
import { dataset } from 'aifn-compute/learning/estimators'
import { accuracy } from 'aifn-compute/learning/metrics'
import { crossValidate, groupKFold, kFold } from 'aifn-compute/learning/validate'
import { kNearestNeighbours } from 'aifn-methods/learning/neighbours'

const PANEL = 280
const NEW_ENTITIES = 200
const ESTIMATES = ['random 5-fold', 'grouped 5-fold', 'new entities']

/** Plots inside a Figure take the frame's height; this gives the plots inside it a fixed height instead. */
function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

/**
 * `entities` entities with `records` rows each. Each entity has a centre in the plane and one label, positive when its
 * first coordinate plus unit noise is above 0; its rows scatter around the centre with standard deviation `spread`.
 */
function entityData(s: Stream, entities: number, records: number, spread: number) {
  const centre = toFlat(normals(child(s, 'centre'), [entities, 2]))
  const noise = toFlat(normals(child(s, 'label'), [entities]))
  const jitter = toFlat(normals(child(s, 'rows'), [entities * records, 2]))
  const n = entities * records
  const x = new Float64Array(2 * n)
  const y = new Float64Array(n)
  const groups: number[] = []
  for (let i = 0; i < n; i++) {
    const e = Math.floor(i / records)
    x[2 * i] = centre[2 * e] + spread * jitter[2 * i]
    x[2 * i + 1] = centre[2 * e + 1] + spread * jitter[2 * i + 1]
    y[i] = centre[2 * e] + noise[e] > 0 ? 1 : 0
    groups.push(e)
  }
  return { x, y, groups, n }
}

/** Random and grouped cross-validation of a nearest-neighbour classifier on rows that come in entities. */
export function GroupLeakage() {
  const state = useFigureState({
    entities: int(40, { ge: 10, le: 100, suggestions: [20, 40, 80], label: 'entities' }),
    records: int(6, { ge: 1, le: 12, suggestions: [1, 3, 6, 10], label: 'rows per entity' }),
    spread: slider(0.01, 1, 0.05, { step: 0.01, label: 'spread of an entity’s rows' }),
    k: int(1, { ge: 1, le: 25, suggestions: [1, 5, 15], label: 'neighbours k' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  })
  const { entities, records, spread, k, seed } = state
  const view = useMemo(() => {
    const s = stream(`group-leakage/${seed}`)
    const d = entityData(child(s, 'train'), entities, records, spread)
    const data = dataset(fromData(d.x, [d.n, 2]), fromData(d.y), { groups: d.groups })
    const model = kNearestNeighbours({ k })
    const scores = [kFold({ k: 5, shuffle: true }), groupKFold({ k: 5 })].map((splitter) => {
      try {
        return crossValidate(model, data, splitter, [accuracy], { stream: child(s, 'cv') }).mean.accuracy
      } catch {
        return NaN
      }
    })
    // The accuracy the procedure actually has: train on all rows, test on rows of entities never seen.
    const fresh = entityData(child(s, 'new'), NEW_ENTITIES, Math.max(records, 1), spread)
    const fitted = model.fit(data)
    const predicted = toFlat(fitted.decide(fromData(fresh.x, [fresh.n, 2])))
    const truth = predicted.filter((p, i) => p === fresh.y[i]).length / fresh.n
    const xs = Array.from({ length: d.n }, (_, i) => d.x[2 * i])
    const ys = Array.from({ length: d.n }, (_, i) => d.x[2 * i + 1])
    const pos = Array.from({ length: d.n }, (_, i) => i).filter((i) => d.y[i] === 1)
    const neg = Array.from({ length: d.n }, (_, i) => i).filter((i) => d.y[i] === 0)
    return {
      estimates: [...scores, truth],
      pos: { x: pos.map((i) => xs[i]), y: pos.map((i) => ys[i]) },
      neg: { x: neg.map((i) => xs[i]), y: neg.map((i) => ys[i]) },
    }
  }, [entities, records, spread, k, seed])
  const dx = useAxis({ label: 'x₁', range: [-4, 4] })
  const dy = useAxis({ label: 'x₂', range: [-4, 4], equal: dx })
  const ex = useAxis({ categories: ESTIMATES })
  const ey = useAxis({ label: 'accuracy', range: [0.4, 1] })
  return (
    <Figure
      title="Random splits of grouped rows overstate accuracy"
      state={state}
      hoverReadout={false}
      caption={`Left: rows from entities (patients, users, devices). Each entity has a centre and one label, and its rows scatter around the centre; the label depends on the first coordinate plus noise, so the best possible accuracy on a new entity is well below 1. Right: the accuracy of a k-nearest-neighbour classifier estimated three ways: by random 5-fold cross-validation, by grouped 5-fold cross-validation (each entity wholly in one fold), and by training on all rows and testing on ${NEW_ENTITIES} entities never seen. Widen the spread, set one row per entity or raise k to see the gap between the random estimate and the other two shrink.`}
      readouts={
        <>
          {ESTIMATES.map((name, i) => (
            <Readout key={name} label={name} value={formatNumber(view.estimates[i])} />
          ))}
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <FixedHeight height={PANEL}>
          <Plot x={dx} y={dy} title="rows">
            {view.neg.x.length > 0 && <Points name="negative" x={view.neg.x} y={view.neg.y} slot={0} />}
            {view.pos.x.length > 0 && <Points name="positive" x={view.pos.x} y={view.pos.y} slot={1} />}
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={ex} y={ey} title="estimated accuracy">
            <Bars name="accuracy" x={[0, 1, 2]} y={view.estimates} base={0.4} slot={2} />
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
