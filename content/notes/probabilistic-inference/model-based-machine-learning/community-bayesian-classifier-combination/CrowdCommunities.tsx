import { useMemo, useState } from 'react'
import { Heatmap, Interactive, ParamChoice, Readout, XYChart, formatNumber, type XYSeries } from '@/components/viz'
import { rng } from '@/lib/math'
import {
  accuracy,
  argmax,
  fitBcc,
  fitCommunitiesRestarts,
  majorityVote,
  matchCommunities,
  simulateCrowd,
} from '../_shared/crowd'

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
  const d = simulateCrowd(rng(2).uniform, ITEMS, CONFUSION, sizes, LABELS_PER_ITEM)
  const bcc = fitBcc(d.labels, ITEMS, workers, 3)
  const com = fitCommunitiesRestarts(d.labels, ITEMS, workers, 3, 3, rng(11).uniform, 4)
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
  const [size, setSize] = useState('48')
  const [community, setCommunity] = useState('0')
  const r = runs[SIZES.indexOf(Number(size))]
  const m = Number(community)

  const curves = useMemo((): XYSeries[] => {
    const x = runs.map((q) => q.perWorker)
    return [
      { name: 'majority vote', type: 'line', x, y: runs.map((q) => q.mv), slot: 0 },
      { name: 'BCC (one matrix per worker)', type: 'line', x, y: runs.map((q) => q.bcc), slot: 1 },
      { name: 'community model', type: 'line', x, y: runs.map((q) => q.community), slot: 2 },
      { name: 'known matrices', type: 'line', x, y: runs.map((q) => q.oracle), muted: true },
    ]
  }, [runs])

  return (
    <Interactive
      title="Recovering worker communities from crowd labels"
      caption="240 tweets with three true classes, each labelled by four workers drawn from a crowd with three kinds of worker: reliable, neutral-leaning (says neutral about half the time) and random. The crowd grows from 12 to 96 workers, so each worker gives fewer labels. Left: accuracy against labels per worker. Majority vote ignores who labelled what. BCC learns a confusion matrix per worker and degrades as each worker's data thins. The community model shares one matrix per community and stays close to the accuracy achievable with the true matrices. Right: a true community confusion matrix and the one the community model recovered, rows the true class."
      controls={
        <>
          <ParamChoice
            label="crowd size"
            value={size}
            onChange={setSize}
            options={SIZES.map((s) => ({ value: String(s), label: `${s} workers` }))}
          />
          <ParamChoice
            label="community"
            value={community}
            onChange={setCommunity}
            options={COMMUNITIES.map((c, i) => ({ value: String(i), label: c }))}
          />
        </>
      }
      readout={
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
        <XYChart series={curves} xLabel="labels per worker" yLabel="accuracy" yRange={[0.6, 1]} height={300} />
        <Heatmap
          x={AXIS}
          y={AXIS}
          z={CONFUSION[m]}
          range={[0, 1]}
          xLabel="label given"
          yLabel="true class"
          valueLabel="probability"
          height={300}
        />
        <Heatmap
          x={AXIS}
          y={AXIS}
          z={r.confusion[m]}
          range={[0, 1]}
          xLabel="label given"
          yLabel="true class (recovered)"
          valueLabel="probability"
          height={300}
        />
      </div>
      <p className="text-xs text-muted-foreground">
        Classes on both axes: {CLASSES.map((c, i) => `${i} ${c}`).join(', ')}.
      </p>
    </Interactive>
  )
}
