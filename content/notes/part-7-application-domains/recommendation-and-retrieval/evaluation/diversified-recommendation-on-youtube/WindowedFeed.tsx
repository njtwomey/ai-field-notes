import { useMemo } from 'react'
import {
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
import { eigh } from 'aifn-compute/numerics/linalg'
import { tensor, toArray, toFlat } from 'aifn-compute/foundation/tensor'
import { stream, uniform } from 'aifn-compute/foundation/random'

const TOPICS = ['basketball', 'cooking', 'music', 'comedy']
const N = 32
const SHOWN = 20
/** The pointwise scorer favours basketball: the redundancy the paper starts from. */
const TOPIC_QUALITY = [0.85, 0.72, 0.7, 0.66]

type Video = { topic: number; tokens: Set<string>; q: number }

/** Candidate videos: each has its topic's three core tokens, two of its six other tokens, and a quality score. */
function candidates(): Video[] {
  const r = stream('youtube-dpp-feed')
  return Array.from({ length: N }, (_, i) => {
    const topic = i % TOPICS.length
    const tokens = new Set([`${topic}:a`, `${topic}:b`, `${topic}:c`])
    while (tokens.size < 5) tokens.add(`${topic}:${Math.floor(uniform(r) * 6)}`)
    return { topic, tokens, q: Math.min(1, TOPIC_QUALITY[topic] + 0.1 * (uniform(r) - 0.5)) }
  })
}
const VIDEOS = candidates()

function jaccardDistance(a: Set<string>, b: Set<string>): number {
  let both = 0
  for (const t of a) if (b.has(t)) both++
  return 1 - both / (a.size + b.size - both)
}
const D = VIDEOS.map((a) => VIDEOS.map((b) => jaccardDistance(a.tokens, b.tokens)))

/** L_ii = q_i², L_ij = α q_i q_j exp(−D_ij / 2σ²); negative eigenvalues (possible when α > 1) are set to zero. */
function kernel(alpha: number, sigma: number): { L: number[][]; clipped: number } {
  const raw = VIDEOS.map((a, i) =>
    VIDEOS.map((b, j) => (i === j ? a.q * a.q : alpha * a.q * b.q * Math.exp(-D[i][j] / (2 * sigma * sigma)))),
  )
  const { values, vectors } = eigh(tensor(raw))
  const lam = toFlat(values)
  const clipped = lam.filter((l) => l < 0).length
  if (!clipped) return { L: raw, clipped }
  const V = toArray(vectors) as number[][]
  const L = raw.map((_, i) => raw.map((__, j) => lam.reduce((s, l, n) => s + Math.max(0, l) * V[i][n] * V[j][n], 0)))
  return { L, clipped }
}

/** Greedy size-k maximisation of det(L_Y) over the items in `pool`, by incremental Cholesky; returns them in order. */
function greedy(L: number[][], pool: number[], k: number): number[] {
  const d2 = new Map(pool.map((i) => [i, L[i][i]]))
  const c = new Map(pool.map((i) => [i, [] as number[]]))
  const chosen: number[] = []
  while (chosen.length < k) {
    let j = -1
    for (const i of pool) if (!chosen.includes(i) && (j < 0 || d2.get(i)! > d2.get(j)!)) j = i
    const dj = Math.sqrt(Math.max(d2.get(j)!, 1e-12))
    chosen.push(j)
    for (const i of pool) {
      if (chosen.includes(i)) continue
      const ci = c.get(i)!
      const cj = c.get(j)!
      let dot = 0
      for (let t = 0; t < cj.length; t++) dot += cj[t] * ci[t]
      const e = (L[j][i] - dot) / dj
      ci.push(e)
      d2.set(i, d2.get(i)! - e * e)
    }
    c.get(j)!.push(dj)
  }
  return chosen
}

/** Algorithm 1 of Wilhelm et al.: fill the feed k videos at a time, each window a greedy DPP choice among the rest. */
function windowedFeed(L: number[][], k: number): number[] {
  let rest = VIDEOS.map((_, i) => i)
  const feed: number[] = []
  while (rest.length) {
    const window = greedy(L, rest, Math.min(k, rest.length))
    feed.push(...window)
    rest = rest.filter((i) => !window.includes(i))
  }
  return feed
}

const BY_QUALITY = VIDEOS.map((_, i) => i).sort((a, b) => VIDEOS[b].q - VIDEOS[a].q)

/** Distinct topics among the first n videos of a feed, and the longest run of one topic. */
function summary(feed: number[], n: number) {
  const top = feed.slice(0, n).map((i) => VIDEOS[i].topic)
  let run = 1
  let best = 1
  for (let t = 1; t < SHOWN; t++) {
    run = VIDEOS[feed[t]].topic === VIDEOS[feed[t - 1]].topic ? run + 1 : 1
    best = Math.max(best, run)
  }
  return { topics: new Set(top).size, run: best }
}

export function WindowedFeed() {
  const state = useFigureState({
    alpha: float(0.9, { min: 0.05, max: 3, step: 0.05, label: 'α (off-diagonal scale)', suggestions: [0.5, 0.9, 1.5] }),
    sigma: float(0.8, { min: 0.1, max: 2, step: 0.05, label: 'σ (similarity width)', suggestions: [0.4, 0.8, 1.2] }),
    k: int(6, { min: 1, max: N, label: 'window size k', suggestions: [1, 6, 12] }),
  })
  const { alpha, sigma, k } = state
  const kern = useComputed(() => kernel(alpha, sigma), [alpha, sigma])
  const feed = useMemo(() => windowedFeed(kern.value.L, k), [kern.value, k])

  const points = useMemo(() => {
    const rows = [BY_QUALITY, feed]
    const x: number[] = []
    const y: number[] = []
    const g: number[] = []
    rows.forEach((row, r) =>
      row.slice(0, SHOWN).forEach((i, pos) => {
        x.push(pos + 1)
        y.push(r)
        g.push(VIDEOS[i].topic)
      }),
    )
    return { x, y, g }
  }, [feed])

  const sorted = summary(BY_QUALITY, 8)
  const dpp = summary(feed, 8)
  const xAxis = useAxis({ label: 'position in the feed', range: [0.5, SHOWN + 0.5], integer: true })
  const yAxis = useAxis({ label: 'ranking', categories: ['sorted by quality q', 'windowed DPP'] })
  return (
    <Figure
      title="A homepage feed, sorted by quality or chosen by a DPP"
      state={state}
      caption="Thirty-two candidate videos in four topics; the pointwise scorer rates basketball highest, so sorting by quality opens the feed with a run of basketball. The DPP feed is built as YouTube built it: a greedy most-likely set of k videos for the top window, then the next k from the rest, and so on. Distances are Jaccard distances between the videos' tokens. With k = 1 each window is the single best video, which is the sorted feed again. Raising α or σ strengthens the repulsion; α above 1 can make the kernel non-PSD, and negative eigenvalues are then set to zero."
      readouts={
        <>
          <Readout label="topics in top 8 (sorted, DPP)" value={`${sorted.topics}, ${dpp.topics}`} />
          <Readout label="longest same-topic run (sorted, DPP)" value={`${sorted.run}, ${dpp.run}`} />
          <Readout label="eigenvalues clipped" value={String(kern.value.clipped)} />
          <Readout
            label="mean q of top 8 (sorted, DPP)"
            value={`${formatNumber(BY_QUALITY.slice(0, 8).reduce((s, i) => s + VIDEOS[i].q, 0) / 8)}, ${formatNumber(feed.slice(0, 8).reduce((s, i) => s + VIDEOS[i].q, 0) / 8)}`}
          />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={200}>
        <Points name="video" x={points.x} y={points.y} group={points.g} groupNames={TOPICS} size={13} />
      </Plot>
    </Figure>
  )
}
