import { useMemo, useState } from 'react'
import {
  call,
  Figure,
  formatNumber,
  Handle,
  Histogram,
  int,
  Plot,
  Readout,
  slider,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { child, permutation, stream, units } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { histogram } from 'aifn-compute/probability/stats'
import { pairedShapes, type PairedViews } from 'aifn-methods/data/synthetic'
import { embed, similarities, TwoTower, type ContrastiveSnapshot } from 'aifn-methods/neural/contrastive'

/** Pairs the scorer is trained on (seen combinations only), and pairs in the crawled pool it scores. */
const TRAIN_N = 1500
const POOL_N = 400
const DATA = { heldOut: 2 }
const EDGES = Array.from({ length: 41 }, (_, i) => -1 + i / 20)

const pct = (v: number) => (Number.isFinite(v) ? `${Math.round(100 * v)}%` : '—')

/** The pool, rebuilt on the page from the same stream the worker drew it from. */
const poolOf = (seed: number) =>
  pairedShapes(stream(`paired-data-curation/pool/${seed}`), { ...DATA, n: POOL_N, include: 'all' })

/**
 * A tiny CLIP scores a pool of image–caption pairs, some of whose captions belong to another image. Filtering keeps
 * the pairs whose cosine similarity clears a threshold. The scorer was trained on seen shape–colour combinations only,
 * so correct pairs of unseen combinations score lower and are the first correct pairs a strict threshold drops.
 */
export function ScoreFiltering() {
  const state = useFigureState({
    wrong: slider(0, 0.6, 0.3, { step: 0.05, label: 'share of pairs with a wrong caption' }),
    steps: int(600, { ge: 50, le: 3000, suggestions: [200, 600, 1500], label: 'scorer training steps' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  })
  const { wrong, steps, seed } = state
  // The threshold has a natural place on the score axis, so it is dragged there.
  const [threshold, setThreshold] = useState(0.3)

  const key = `${steps}|${seed}`
  const run = useComputed(
    () =>
      call<ContrastiveSnapshot>(
        'neural/contrastive/contrastiveTrainingRun',
        call<PairedViews>(
          'data/synthetic/pairedShapes',
          call('foundation/random/stream', `paired-data-curation/train/${seed}`),
          { ...DATA, n: TRAIN_N },
        ),
        call<PairedViews>(
          'data/synthetic/pairedShapes',
          call('foundation/random/stream', `paired-data-curation/pool/${seed}`),
          { ...DATA, n: POOL_N, include: 'all' },
        ),
        { steps, every: Math.max(1, Math.round(steps / 10)), dim: 8, batchSize: 64, seed, evaluate: 100 },
      ),
    [steps, seed],
    {
      mode: 'worker',
      initial: null as { snap: ContrastiveSnapshot; key: string } | null,
      then: (snap) => ({ snap, key }),
      cancelAfter: 400,
    },
  )
  const snap = run.value?.key === key ? run.value.snap : null
  const params = snap?.checkpoints.at(-1)?.params ?? null
  const trainedTo = snap?.checkpoints.at(-1)?.step ?? 0

  const pool = useMemo(() => poolOf(seed), [seed])

  // Every pool pair's score under the matching caption and under a caption drawn from another pair.
  const scores = useMemo(() => {
    if (!snap || !params) return null
    const model = TwoTower({
      inA: pool.a.shape[1],
      inB: pool.b.shape[1],
      hidden: snap.config.hidden,
      dim: snap.config.dim,
    })
    const s = similarities(embed(model, params, pool.a, 'a'), embed(model, params, pool.b, 'b'))
    const mix = stream(`paired-data-curation/mix/${seed}`)
    const other = toFlat(permutation(child(mix, 'caption'), POOL_N))
    return { s, other, draw: units(child(mix, 'wrong'), POOL_N), unseen: toFlat(pool.truth.heldOut) }
  }, [snap, params, pool, seed])

  const groups = useMemo(() => {
    const seen: number[] = []
    const unseen: number[] = []
    const mismatched: number[] = []
    if (!scores) return { seen, unseen, mismatched }
    for (let i = 0; i < POOL_N; i++) {
      // A pair is given a wrong caption when its draw falls below the chosen share (and the other caption is not its own).
      if (scores.draw[i] < wrong && scores.other[i] !== i) mismatched.push(scores.s[i * POOL_N + scores.other[i]])
      else (scores.unseen[i] ? unseen : seen).push(scores.s[i * POOL_N + i])
    }
    return { seen, unseen, mismatched }
  }, [scores, wrong])

  const kept = (v: number[]) => v.filter((x) => x >= threshold).length
  const total = groups.seen.length + groups.unseen.length + groups.mismatched.length
  const keptAll = kept(groups.seen) + kept(groups.unseen) + kept(groups.mismatched)
  const wrongAfter = keptAll > 0 ? kept(groups.mismatched) / keptAll : NaN

  // The tallest bar of the three histograms sets the vertical range.
  const top = Math.max(
    10,
    ...[groups.seen, groups.unseen, groups.mismatched].map((v) =>
      v.length ? Math.max(...toFlat(histogram(v, { bins: EDGES }).counts)) : 0,
    ),
  )
  const x = useAxis({ label: 'cosine similarity of image and caption', range: [-1, 1] })
  const y = useAxis({ label: 'pairs', range: [0, Math.ceil(top * 1.1)] })

  return (
    <Figure
      title="Filtering pairs by an image–caption similarity score"
      state={state}
      hoverReadout={false}
      caption={`A pool of ${POOL_N} image–caption pairs of coloured shapes, in which a chosen share of captions has been swapped for another pair's caption. A small two-tower model trained contrastively on ${TRAIN_N} correct pairs scores every pair by the cosine similarity of its image and caption embeddings. The scorer never saw two shape–colour combinations (red triangles and blue squares), which make up about a twelfth of the pool. Drag the dashed line to set the threshold: pairs to its right are kept. Most wrong captions score low, but a wrong caption that happens to describe a similar shape scores high. Correct pairs of the unseen combinations score lower than other correct pairs, so a strict threshold removes them first. The scorer trains in the background; the readouts update when it finishes.`}
      readouts={
        <>
          <Readout
            label="scorer"
            value={run.error ? `failed: ${run.error}` : snap ? `trained ${trainedTo} of ${steps} steps` : 'training…'}
          />
          <Readout label="pairs kept" value={total ? `${keptAll} of ${total}` : '—'} />
          <Readout
            label="wrong captions, before → after"
            value={`${pct(groups.mismatched.length / Math.max(1, total))} → ${pct(wrongAfter)}`}
          />
          <Readout
            label="correct pairs kept: seen, unseen combinations"
            value={`${pct(kept(groups.seen) / Math.max(1, groups.seen.length))}, ${pct(kept(groups.unseen) / Math.max(1, groups.unseen.length))}`}
          />
        </>
      }
    >
      <Plot x={x} y={y}>
        {groups.seen.length > 0 && (
          <Histogram
            name="correct pair, seen combination"
            values={groups.seen}
            bins={EDGES}
            normalize="count"
            slot={0}
          />
        )}
        {groups.unseen.length > 0 && (
          <Histogram
            name="correct pair, unseen combination"
            values={groups.unseen}
            bins={EDGES}
            normalize="count"
            slot={2}
          />
        )}
        {groups.mismatched.length > 0 && (
          <Histogram name="wrong caption" values={groups.mismatched} bins={EDGES} normalize="count" slot={1} />
        )}
        <Handle
          kind="x"
          at={threshold}
          label={`threshold ${formatNumber(threshold)}`}
          onDrag={(t) => setThreshold(Math.max(-0.95, Math.min(0.95, Math.round(t * 100) / 100)))}
        />
      </Plot>
    </Figure>
  )
}
