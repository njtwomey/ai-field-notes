import { useMemo } from 'react'
import {
  choice,
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Points,
  Readout,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { child, stream, type Stream } from 'aifn-compute/foundation/random'
import { fromData, mean, toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { dataset } from 'aifn-compute/learning/estimators'
import { accuracy } from 'aifn-compute/learning/metrics'
import { crossValidate, groupKFold, kFold } from 'aifn-compute/learning/validate'
import { blobs, gaussians } from 'aifn-methods/data/synthetic'
import { kNearestNeighbours } from 'aifn-methods/learning/neighbours'
import { FoldGrid } from '../_shared/FoldGrid'

const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')
const avg = (v: number[]) => mean(fromData(Float64Array.from(v)))
const COLUMNS = ['record 5-fold', 'group 5-fold', 'leave group out', 'true: new records', 'true: new entities']
const SHOWN = [
  { value: 'record', label: 'record-wise 5-fold' },
  { value: 'group', label: 'group 5-fold' },
] as const
const SEPARATION = 1.5

/**
 * Entities (patients, users, devices) whose centres come from two Gaussian classes; each entity carries its class as
 * its label and contributes `records` rows scattered around its centre.
 */
function entityData(s: Stream, entities: number, records: number, spread: number) {
  const centres = gaussians(child(s, 'centres'), { n: entities, separation: SEPARATION })
  const centre = toRows(centres.x) as number[][]
  const label = Array.from(toFlat(centres.y))
  const rows = blobs(child(s, 'rows'), { centers: centre, n: Array(entities).fill(records), sd: spread })
  const group = Array.from(toFlat(rows.y))
  const y = fromData(Float64Array.from(group.map((e) => label[e])))
  return { centre, label, group, data: dataset(rows.x, y, { groups: group }) }
}

/**
 * Record-wise and grouped cross-validation of a nearest-neighbour classifier on rows that come in entities, beside two
 * true accuracies: on new records of the entities already seen, and on entities never seen.
 */
export function RecordOrSubject() {
  const state = useFigureState({
    entities: int(30, { ge: 10, le: 80, suggestions: [15, 30, 60], label: 'entities' }),
    records: int(6, { ge: 1, le: 12, suggestions: [1, 3, 6, 10], label: 'rows per entity' }),
    spread: float(0.2, { min: 0.02, max: 1.5, step: 0.02, label: 'spread of an entity’s rows' }),
    k: int(1, { ge: 1, le: 15, suggestions: [1, 5, 9], label: 'neighbours k' }),
    replicates: int(12, { ge: 2, le: 40, suggestions: [6, 12, 24], label: 'datasets drawn' }),
    shown: choice(SHOWN, 'group', { label: 'folds drawn' }),
  })
  const { entities, records, spread, k, replicates, shown } = state
  const run = useComputed(
    () => {
      const model = kNearestNeighbours({ k })
      const values = COLUMNS.map(() => [] as number[])
      for (let r = 0; r < replicates; r++) {
        const s = child(stream('grouped-cv'), r)
        const d = entityData(s, entities, records, spread)
        const splitters = [kFold({ k: 5, shuffle: true }), groupKFold({ k: 5 }), groupKFold({ k: entities })]
        splitters.forEach((splitter, j) => {
          try {
            values[j].push(crossValidate(model, d.data, splitter, [accuracy], { stream: child(s, 'cv') }).mean.accuracy)
          } catch {
            values[j].push(NaN)
          }
        })
        const fitted = model.fit(d.data)
        const again = blobs(child(s, 'new-records'), { centers: d.centre, n: Array(entities).fill(20), sd: spread })
        const againLabels = Array.from(toFlat(again.y)).map((e) => d.label[e])
        values[3].push(accuracy(againLabels, fitted.decide(again.x)))
        const fresh = entityData(child(s, 'new-entities'), 200, Math.max(records, 1), spread)
        values[4].push(accuracy(fresh.data.y, fitted.decide(fresh.data.x)))
      }
      return { values, means: values.map((v) => avg(v.filter(Number.isFinite))) }
    },
    [entities, records, spread, k, replicates],
    { mode: 'release' },
  )
  const v = run.value
  const first = useMemo(
    () => entityData(child(stream('grouped-cv'), 0), entities, records, spread),
    [entities, records, spread],
  )
  const splits = useMemo(
    () =>
      (shown === 'group' ? groupKFold({ k: 5 }) : kFold({ k: 5, shuffle: true })).split(
        first.data,
        stream('grouped-cv/drawn'),
      ),
    [first, shown],
  )
  const boundaries = useMemo(
    () => Array.from({ length: entities - 1 }, (_, e) => (e + 1) * records - 0.5),
    [entities, records],
  )
  const cx = useAxis({ categories: COLUMNS })
  const ay = useAxis({ label: 'accuracy', range: [0.4, 1] })
  const pts = useMemo(
    () => ({
      x: v.values.flatMap((col, j) => col.map(() => j)),
      y: v.values.flatMap((col) => col),
    }),
    [v],
  )
  return (
    <Figure
      title="Record-wise and subject-wise estimates"
      state={state}
      hoverReadout={false}
      caption="Each entity (a patient, a user, a device) has a centre drawn from one of two Gaussian classes and carries that class as its label; its rows scatter around the centre. Top: the folds of one dataset, rows ordered by entity, with thin lines between entities; group folds keep each entity whole, record-wise folds cut through them. Bottom: the accuracy of a k-nearest-neighbour classifier estimated three ways (record-wise 5-fold, group 5-fold, leave one group out), beside two true accuracies of the model fitted to all rows, one for new records of the entities it has seen and one for entities it has never seen, each dot one dataset and the larger dots their means. Record-wise cross-validation estimates the first; group 5-fold and leave-one-group-out estimate the second. With a small spread and k = 1, a record's nearest neighbour is a sibling from its own entity, and the record-wise estimate is far above the accuracy on new entities. Widen the spread, raise k or set one row per entity to close the gap."
      readouts={
        <>
          {COLUMNS.map((c, j) => (
            <Readout key={c} label={c} value={fmt(v.means[j])} />
          ))}
        </>
      }
    >
      <FoldGrid
        splits={splits}
        n={entities * records}
        title={shown === 'group' ? 'group 5-fold' : 'record-wise 5-fold'}
        rowLabel="fold"
        boundaries={records > 1 ? boundaries : undefined}
        rowHeight={16}
      />
      <Plot x={cx} y={ay} title="estimates and true accuracies">
        <Points name="one dataset" x={pts.x} y={pts.y} slot={0} thin stale={run.stale} />
        <Points name="mean over datasets" x={COLUMNS.map((_, j) => j)} y={v.means} emphasis size={10} />
      </Plot>
    </Figure>
  )
}
