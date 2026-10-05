import { useMemo } from 'react'
import { choice, Curve, Figure, formatNumber, Plot, Raster, Readout, useAxis, useFigureState } from 'aifn-render'
import { stream, uniform } from 'aifn-compute/foundation/random'
import {
  accuracy,
  argmax,
  fitBcc,
  fitCommunitiesRestarts,
  majorityVote,
  matchCommunities,
  simulateCrowd,
} from '../_shared/crowd'

/** A uniform draw function on its own seeded stream. */
const uniformDraws = (seed: number) => {
  const g = stream(seed)
  return () => uniform(g)
}

const CLASSES = ['positive', 'negative', 'neutral']
const COMMUNITIES = ['reliable', 'neutral-leaning', 'random']
/** Rows: true class; columns: label given. */
const CONFUSION = [
  [
    [0.85, 0.05, 0.1],
    [0.05, 0.85, 0.1],
    [0.05, 0.05, 0.9],
  ],
  [
    [0.45, 0.05, 0.5],
    [0.05, 0.45, 0.5],
    [0.1, 0.1, 0.8],
  ],
  [
    [1 / 3, 1 / 3, 1 / 3],
    [1 / 3, 1 / 3, 1 / 3],
    [1 / 3, 1 / 3, 1 / 3],
  ],
]
const ITEMS = 240
const LABELS_PER_ITEM = 4
const SIZES = [12, 24, 48, 96]
const SHARES = [0.5, 0.3, 0.2]

type Run = {
  workers: number
  perWorker: number
  mv: number
  bcc: number
  community: number
  oracle: number
  confusion: number[][][]
  sizes: number[]
}

function run(workers: number): Run {
  const sizes = [Math.round(workers * SHARES[0]), Math.round(workers * SHARES[1])]
  sizes.push(workers - sizes[0] - sizes[1])
  const d = simulateCrowd(uniformDraws(2), ITEMS, CONFUSION, sizes, LABELS_PER_ITEM)
  const bcc = fitBcc(d.labels, ITEMS, workers, 3)
  const com = fitCommunitiesRestarts(d.labels, ITEMS, workers, 3, 3, uniformDraws(11), 4)
  const perm = matchCommunities(com.membership, d.community, 3)
  // The Bayes-optimal answer if every worker's true confusion matrix were known.
  const logp = Array.from({ length: ITEMS }, () => [0, 0, 0])
  for (const l of d.labels)
    for (let c = 0; c < 3; c++) logp[l.item][c] += Math.log(CONFUSION[d.community[l.worker]][c][l.label])
  return {
    workers,
    perWorker: (ITEMS * LABELS_PER_ITEM) / workers,
    mv: accuracy(majorityVote(d.labels, ITEMS, 3), d.truth),
    bcc: accuracy(bcc.items.map(argmax), d.truth),
    community: accuracy(com.items.map(argmax), d.truth),
    oracle: accuracy(logp.map(argmax), d.truth),
    confusion: perm.map((m) => com.confusion[m]),
    sizes: perm.map((m) => com.membership.reduce((s, r) => s + r[m], 0)),
  }
}

const AXIS = [0, 1, 2]

/** Majority vote, per-worker confusion matrices and community confusion matrices as the crowd grows sparser. */
export function CrowdCommunities() {
  const runs = useMemo(() => SIZES.map(run), [])
  const state = useFigureState({
    size: choice(
      SIZES.map((s) => ({ value: String(s), label: `${s} workers` })),
      '48',
      { label: 'crowd size' },
    ),
    community: choice(
      COMMUNITIES.map((c, i) => ({ value: String(i), label: c })),
      '0',
      { label: 'community' },
    ),
  })
  const r = runs[SIZES.indexOf(Number(state.size))]
  const m = Number(state.community)

  const curves = useMemo(() => {
    const x = runs.map((q) => q.perWorker)
    return [
      { name: 'majority vote', x, y: runs.map((q) => q.mv), slot: 0 },
      { name: 'BCC (one matrix per worker)', x, y: runs.map((q) => q.bcc), slot: 1 },
      { name: 'community model', x, y: runs.map((q) => q.community), slot: 2 },
      { name: 'known matrices', x, y: runs.map((q) => q.oracle), muted: true },
    ] as const
  }, [runs])

  const xAxis = useAxis({ label: 'labels per worker', hold: 'union' })
  const yAxis = useAxis({ label: 'accuracy', range: [0.6, 1] })
  const xAxis2 = useAxis({ label: 'label given' })
  const yAxis2 = useAxis({ label: 'true class' })
  const xAxis3 = useAxis({ label: 'label given' })
  const yAxis3 = useAxis({ label: 'true class (recovered)' })
  return (
    <Figure
      title="Recovering worker communities from crowd labels"
      state={state}
      caption="240 tweets with three true classes, each labelled by four workers drawn from a crowd with three kinds of worker: reliable, neutral-leaning (says neutral about half the time) and random. The crowd grows from 12 to 96 workers, so each worker gives fewer labels. Left: accuracy against labels per worker. Majority vote ignores who labelled what. BCC learns a confusion matrix per worker and degrades as each worker's data thins. The community model shares one matrix per community and stays close to the accuracy achievable with the true matrices. Right: a true community confusion matrix and the one the community model recovered, rows the true class."

      readouts={
        <>
          <Readout label="labels per worker" value={formatNumber(r.perWorker)} />
          <Readout label="majority vote" value={formatNumber(r.mv)} />
          <Readout label="BCC" value={formatNumber(r.bcc)} />
          <Readout label="community model" value={formatNumber(r.community)} />
          <Readout label={`workers assigned to ${COMMUNITIES[m]}`} value={formatNumber(r.sizes[m])} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <Plot x={xAxis} y={yAxis} height={300}>
          <Curve {...curves[0]} />
          <Curve {...curves[1]} />
          <Curve {...curves[2]} />
          <Curve {...curves[3]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={300}>
          <Raster x={AXIS} y={AXIS} z={CONFUSION[m]} range={[0, 1]} valueLabel={'probability'} />
        </Plot>
        <Plot x={xAxis3} y={yAxis3} height={300}>
          <Raster x={AXIS} y={AXIS} z={r.confusion[m]} range={[0, 1]} valueLabel={'probability'} />
        </Plot>
      </div>
      <p className="text-xs text-muted-foreground">
        Classes on both axes: {CLASSES.map((c, i) => `${i} ${c}`).join(', ')}.
      </p>
    </Figure>
  )
}
