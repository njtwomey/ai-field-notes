/**
 * Showcase: recommenders from popularity to two towers. One recommender of aifn-methods `recommenderRun` is trained in
 * the worker on `implicitFeedback` (a clustered low-rank taste model under popularity-biased exposure, the last items
 * of each user held out); the page draws the checkpoints it streams: every user's scores, held-out recall and NDCG
 * against the epoch, the embedding map, and the pinned user's top-k list.
 */
import { useMemo, useState } from 'react'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { implicitFeedback } from 'aifn-methods/data/synthetic'
import {
  RECOMMENDERS,
  topK,
  type RecommenderKind,
  type RecommenderRunOptions,
  type RecommenderSnapshot,
} from 'aifn-methods/retrieval/recommenders'
import { Player } from 'aifn-render/controls'
import { ControlRow, Figure } from 'aifn-render/layout'
import { call, choice, float, int, pinField, row, useFigureState, usePinned, type Task } from 'aifn-render/state'
import { formatValue, TrainControls, useTrainedRun } from '@lab/views'
import { Curve, Handle, Plot, Plots, Points, Raster, Readout, Segments, useAxis } from 'aifn-render/viz'

const f3 = (v: number | undefined) =>
  v !== undefined && Number.isFinite(v) ? formatValue(Number(v.toPrecision(3))) : '—'
const CATEGORIES = ['category 1', 'category 2', 'category 3', 'category 4', 'category 5']
const NEURAL: RecommenderKind[] = [
  'logistic-mf',
  'bpr',
  'factorisation-machine',
  'field-aware-factorisation-machine',
  'wide-and-deep',
  'deepfm',
  'neural-collaborative-filtering',
  'two-tower',
  'sasrec',
]

type Knobs = { users: number; items: number; exposureBias: number; stickiness: number }
type Settings = Omit<RecommenderRunOptions, 'data'> & { knobs: Knobs; dataSeed: number }

const runTask = ({ knobs, dataSeed, ...rest }: Settings): Task<RecommenderSnapshot> =>
  call<RecommenderSnapshot>('applied/retrieval/recommenders/recommenderRun', {
    ...rest,
    data: call('applied/data/synthetic/implicitFeedback', call('foundation/random/stream', dataSeed), knobs),
  })

export function RecommenderShowcase() {
  const state = useFigureState({
    data: row('1 · data', {
      users: int(80, { ge: 10, le: 300, suggestions: [40, 80, 150], label: 'users' }),
      items: int(100, { ge: 10, le: 300, suggestions: [60, 100, 200], label: 'items' }),
      exposureBias: float(1, { ge: 0, le: 3, suggestions: [0, 1, 2], label: 'exposure bias (Zipf exponent)' }),
      stickiness: float(1.5, { ge: 0, le: 5, suggestions: [0, 1.5, 3], label: 'category stickiness γ' }),
      dataSeed: int(1, { ge: 0, le: 9999, label: 'data seed' }),
    }),
    model: row('2 · model', {
      kind: choice(
        RECOMMENDERS.map((r) => ({ value: r.kind, label: r.name })),
        'bpr',
        { label: 'recommender' },
      ),
      dimension: int(16, { ge: 1, le: 64, suggestions: [4, 8, 16, 32], label: 'embedding dimension' }),
      neighbours: int(20, { ge: 1, le: 100, suggestions: [5, 20, 50], label: 'neighbours (kNN models)' }),
    }),
    run: row('3 · training', {
      epochs: int(15, { ge: 1, suggestions: [5, 15, 30], label: 'epochs' }),
      stepSize: float(0.01, { gt: 0, scale: 'log10', suggestions: [0.003, 0.01, 0.03], label: 'learning rate (Adam)' }),
      regularisation: float(0.01, { ge: 0, scale: 'log10', suggestions: [0.001, 0.01, 0.1], label: 'L2 penalty' }),
      k: int(10, { ge: 1, le: 50, suggestions: [5, 10, 20], label: 'cut-off k' }),
      seed: int(1, { ge: 0, le: 9999, label: 'seed' }),
    }),
    pin: pinField(),
  })
  const knobs: Knobs = {
    users: state.data.users,
    items: state.data.items,
    exposureBias: state.data.exposureBias,
    stickiness: state.data.stickiness,
  }
  const kind = state.model.kind as RecommenderKind
  const settings: Settings = {
    knobs,
    dataSeed: state.data.dataSeed,
    model: kind,
    dimension: state.model.dimension,
    neighbours: state.model.neighbours,
    epochs: state.run.epochs,
    stepSize: state.run.stepSize,
    regularisation: NEURAL.includes(kind) ? state.run.regularisation : undefined,
    k: state.run.k,
    seed: state.run.seed,
  }
  const trained = useTrainedRun(settings, runTask)
  const snap = trained.run.value ?? undefined
  const shown = trained.trained ?? settings

  // The data of the run shown (the same registered call the worker makes).
  const data = useMemo(
    () => implicitFeedback(stream(shown.dataSeed), shown.knobs),
    // oxlint-disable-next-line react-hooks/exhaustive-deps -- the knobs are compared by value
    [JSON.stringify(shown.knobs), shown.dataSeed],
  )
  const U = data.users
  const I = data.items
  const category = useMemo(() => Array.from(toFlat(data.itemCategory)), [data])
  const trainRows = useMemo(() => Array.from(toFlat(data.train)), [data])
  const testRows = useMemo(() => Array.from(toFlat(data.test)), [data])
  const seen = useMemo(() => {
    const s = Array.from({ length: U }, () => new Set<number>())
    for (let r = 0; r < trainRows.length; r += 2) s[trainRows[r]].add(trainRows[r + 1])
    return s
  }, [trainRows, U])
  const held = useMemo(() => {
    const s = Array.from({ length: U }, () => new Set<number>())
    for (let r = 0; r < testRows.length; r += 2) s[testRows[r]].add(testRows[r + 1])
    return s
  }, [testRows, U])

  // The checkpoint shown: a new run opens at epoch 0.
  const shots = snap?.checkpoints ?? []
  const [picked, setPicked] = useState<{ run: unknown; index: number } | null>(null)
  const index = Math.min(picked && picked.run === trained.trained ? picked.index : 0, Math.max(0, shots.length - 1))
  const pick = (i: number) => setPicked({ run: trained.trained, index: i })
  const shot = shots[index]

  // The pinned user (click a row of the score matrix or a user on the map; Escape unpins).
  const pins = usePinned(state.pin, (v) => state.set('pin', v), { valid: (v) => v < U })
  const user = pins.focus ?? 0

  // Scores as a users × items field, each row scaled to [0, 1] so every user's ranking shows.
  const field = useMemo(() => {
    if (!shot) return null
    return Array.from({ length: U }, (_, u) => {
      const row_ = shot.scores.subarray(u * I, (u + 1) * I)
      let lo = Infinity
      let hi = -Infinity
      for (const v of row_) {
        lo = Math.min(lo, v)
        hi = Math.max(hi, v)
      }
      return Array.from(row_, (v) => (hi > lo ? (v - lo) / (hi - lo) : 0.5))
    })
  }, [shot, U, I])
  const trainDots = useMemo(() => {
    const x: number[] = []
    const y: number[] = []
    for (let r = 0; r < trainRows.length; r += 2) {
      y.push(trainRows[r])
      x.push(trainRows[r + 1])
    }
    return { x, y }
  }, [trainRows])
  const testDots = useMemo(() => {
    const x: number[] = []
    const y: number[] = []
    for (let r = 0; r < testRows.length; r += 2) {
      y.push(testRows[r])
      x.push(testRows[r + 1])
    }
    return { x, y }
  }, [testRows])

  const k = shown.k ?? 10
  const list = useMemo(
    () => (shot ? topK(shot.scores.subarray(user * I, (user + 1) * I), k, seen[user]) : []),
    [shot, user, I, k, seen],
  )
  const hits = list.filter((i) => held[user].has(i))

  const ax = useAxis({ label: 'item', range: [-0.5, I - 0.5], key: I, integer: true })
  const ay = useAxis({ label: 'user', range: [-0.5, U - 0.5], key: U, integer: true })
  const ex = useAxis({ label: 'epoch', range: [0, Math.max(1, snap?.epochs ?? shown.epochs ?? 1)], key: snap?.epochs })
  const my = useAxis({ label: `held-out metric @${k}`, range: [0, 1.02] })
  const mapX = useAxis({ label: 'principal axis 1', hold: 'union', key: trained.trained })
  const mapY = useAxis({ label: 'principal axis 2', hold: 'union', key: trained.trained, equal: mapX })
  const rankX = useAxis({ label: 'rank', range: [0.5, k + 0.5], key: k, integer: true })
  const prefY = useAxis({ label: 'true affinity wᵤᵀvᵢ', hold: 'union', key: shown.dataSeed })

  const pref = useMemo(() => toFlat(data.preference), [data])
  const h = snap?.history
  const hi = h && shot ? h.epoch.indexOf(shot.epoch) : -1
  const empty = !trained.trained ? 'press Train to start' : !shot ? 'training…' : undefined
  const name = RECOMMENDERS.find((r) => r.kind === shown.model)?.name ?? shown.model
  const done = snap?.epoch ?? 0
  const total = snap?.epochs || 1
  const marker = shot ? (
    <Handle kind="x" at={shot.epoch} onDrag={(e) => pick(nearest(shots, e))} label={`epoch ${shot.epoch}`} />
  ) : null
  const pinRow = (
    <Segments
      name={`user ${user}`}
      segments={[
        { from: [-0.5, user - 0.5], to: [I - 0.5, user - 0.5] },
        { from: [-0.5, user + 0.5], to: [I - 0.5, user + 0.5] },
      ]}
      emphasis
      width={1.5}
    />
  )
  const userPoint =
    shot?.userMap && pins.focus !== null ? { x: [shot.userMap[user * 2]], y: [shot.userMap[user * 2 + 1]] } : null

  return (
    <Figure
      title="Recommenders: from popularity to two towers"
      purpose="Every recommender ranks the items a user has not seen; held-out recall and NDCG measure how many of each user's next items land in the top k. Popularity ignores taste; the trained models recover the low-rank structure that exposure bias hides."
      state={state}
      defaultSize="XL"
      controls={
        <>
          <TrainControls run={trained as never} progress={done / total} progressText={`${done} / ${total} epochs`} />
          <ControlRow label="checkpoints">
            <Player
              value={index}
              onChange={pick}
              count={Math.max(1, shots.length)}
              label="checkpoint"
              format={(i) => `epoch ${shots[i]?.epoch ?? 0}`}
            />
          </ControlRow>
        </>
      }
      readouts={{
        run: (
          <>
            <Readout label="epoch" value={shot ? shot.epoch : '—'} />
            <Readout label={`recall@${k}`} value={hi >= 0 ? f3(h!.recall[hi]) : '—'} />
            <Readout label={`NDCG@${k}`} value={hi >= 0 ? f3(h!.ndcg[hi]) : '—'} />
            <Readout label="hit rate" value={hi >= 0 ? f3(h!.hitRate[hi]) : '—'} />
            <Readout label="catalogue coverage" value={hi >= 0 ? f3(h!.coverage[hi]) : '—'} />
            <Readout label="training loss" value={hi >= 0 ? f3(h!.loss[hi]) : '—'} />
          </>
        ),
        user: (
          <>
            <Readout label="user" value={`${user}${pins.pinned !== null ? ' (pinned)' : ''}`} />
            <Readout label="seen / held out" value={`${seen[user].size} / ${held[user].size}`} />
            <Readout
              label={`top ${k}`}
              value={list.map((i) => (held[user].has(i) ? `${i}✓` : String(i))).join(' ') || '—'}
            />
            <Readout label="hits" value={shot ? `${hits.length} of ${held[user].size}` : '—'} />
          </>
        ),
      }}
      caption={
        <>
          aifn-methods <code>recommenderRun</code> trains {name} on <code>implicitFeedback</code> ({U} users, {I} items;
          a rank-3 taste model with item categories and user groups; exposure ∝ rank^−
          {shown.knobs.exposureBias}; each user&apos;s last two items held out). Top left: every user&apos;s scores at
          the checkpoint, each row scaled to its own range (dark: ranked high), with the training interactions (small
          dots) and the held-out items (rings); click a row to pin that user, Escape to unpin. Top right: recall@{k} and
          NDCG@{k} on the held-out items against the epoch (popularity and the kNN models have nothing to train: one
          point); drag the epoch marker or play the checkpoints from epoch 0. Bottom left: items (and users, pale) on
          the first two principal axes of the learned embeddings, coloured by item category: categories separate as the
          taste structure is learned. Bottom right: the pinned user&apos;s top {k} unseen items by rank, at their true
          affinity, held-out items marked.
        </>
      }
    >
      <Plots rows={2} cols={2} heights={[1, 1]}>
        <Plot
          x={ax}
          y={ay}
          title={empty ?? `${name}: every user's scores`}
          onPlotClick={([, y]) => pins.toggle(Math.max(0, Math.min(U - 1, Math.round(y))))}
        >
          {field && (
            <Raster
              x={Array.from({ length: I }, (_, i) => i)}
              y={Array.from({ length: U }, (_, u) => u)}
              z={field}
              scale="sequential"
              range={[0, 1]}
              valueLabel="score (row-scaled)"
              fillOpacity={0.8}
            />
          )}
          <Points name="training interactions" x={trainDots.x} y={trainDots.y} thin size={3} muted />
          <Points name="held-out items" x={testDots.x} y={testDots.y} emphasis size={4} />
          {pinRow}
        </Plot>
        <Plot x={ex} y={my} title={empty ?? 'held-out ranking quality'}>
          {h && <Curve name={`recall@${k}`} x={h.epoch} y={h.recall} slot={0} />}
          {h && <Curve name={`NDCG@${k}`} x={h.epoch} y={h.ndcg} slot={1} />}
          {h && h.epoch.length === 1 && <Points name="untrained model" x={h.epoch} y={h.recall} slot={0} size={10} />}
          {marker}
        </Plot>
        <Plot x={mapX} y={mapY} title={empty ?? (shot?.itemMap ? 'learned embeddings' : 'no learned embeddings')}>
          {shot?.userMap && (
            <Points
              name="users"
              x={Array.from({ length: U }, (_, u) => shot.userMap![u * 2])}
              y={Array.from({ length: U }, (_, u) => shot.userMap![u * 2 + 1])}
              muted
              thin
            />
          )}
          {shot?.itemMap && (
            <Points
              name="items"
              x={Array.from({ length: I }, (_, i) => shot.itemMap![i * 2])}
              y={Array.from({ length: I }, (_, i) => shot.itemMap![i * 2 + 1])}
              group={category}
              groupNames={CATEGORIES}
            />
          )}
          {userPoint && <Points name={`user ${user}`} x={userPoint.x} y={userPoint.y} emphasis size={12} />}
        </Plot>
        <Plot x={rankX} y={prefY} title={empty ?? `user ${user}: top ${k} unseen items`}>
          {list.length > 0 && (
            <Points
              name="recommended"
              x={list.map((_, r) => r + 1)}
              y={list.map((i) => pref[user * I + i])}
              group={list.map((i) => category[i])}
              groupNames={CATEGORIES}
              labels={list.map((i) => `item ${i}`)}
            />
          )}
          {hits.length > 0 && (
            <Points
              name="held-out (a hit)"
              x={hits.map((i) => list.indexOf(i) + 1)}
              y={hits.map((i) => pref[user * I + i])}
              emphasis
              size={14}
            />
          )}
        </Plot>
      </Plots>
    </Figure>
  )
}

/** The index of the checkpoint whose epoch is nearest `epoch`. */
function nearest(shots: readonly { epoch: number }[], epoch: number): number {
  let best = 0
  shots.forEach((c, i) => {
    if (Math.abs(c.epoch - epoch) < Math.abs(shots[best].epoch - epoch)) best = i
  })
  return best
}
