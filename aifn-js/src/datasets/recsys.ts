/**
 * Seeded recommendation data: Zipf popularity catalogues, low-rank ratings matrices, and click logs under the
 * position-based and cascade click models.
 */

import { aliasSample, aliasTable, normal, type Stream } from 'aifn/random'
import { fromData, type Tensor } from 'aifn/tensor'
import { checkCount, labels, matrix, vector, type DatasetMeta } from './types'

/**
 * Zipf popularity weights p_i ∝ i^{−s} for items i = 1, …, n (normalised to sum to one): the long-tailed popularity of
 * catalogues, words and queries (Zipf, 1949, "Human Behavior and the Principle of Least Effort").
 */
export function zipfWeights(items: number, exponent = 1): Tensor {
  checkCount(items, 'zipfWeights')
  const w = Float64Array.from({ length: items }, (_, i) => (i + 1) ** -exponent)
  const total = w.reduce((a, b) => a + b, 0)
  return vector(w.map((v) => v / total))
}

/** A draw from a Zipf catalogue. */
export interface ZipfCatalogue {
  /** Popularity p_i of each item (item 0 is the most popular). */
  weights: Tensor
  /** Item index of each interaction, int32, length n. */
  draws: Tensor
  /** Interactions per item, int32, length `items`. */
  counts: Tensor
  meta: DatasetMeta
}

/** n interactions with a catalogue of `items` whose popularity follows Zipf's law with the given exponent. */
export function zipfCatalogue(
  s: Stream,
  options: { items?: number; exponent?: number; n?: number } = {},
): ZipfCatalogue {
  const { items = 1000, exponent = 1, n = 10000 } = options
  checkCount(n, 'zipfCatalogue')
  const weights = zipfWeights(items, exponent)
  const draws = aliasSample(s, aliasTable(weights), { shape: [n] })
  const counts = new Int32Array(items)
  for (let i = 0; i < n; i++) counts[draws.data[i]]++
  return {
    weights,
    draws,
    counts: labels(counts),
    meta: {
      name: 'Zipf catalogue',
      description: `${n} interactions with ${items} items whose popularity falls as rank^−${exponent}.`,
      task: 'recommendation',
      featureNames: ['item'],
      source: 'Zipf (1949), Human Behavior and the Principle of Least Effort',
      stream: s.key,
    },
  }
}

/** A synthetic ratings problem. */
export interface Ratings {
  users: number
  items: number
  /** Every user's rating of every item (users × items), including entries no one observed. */
  truth: Tensor
  /** The noise-free low-rank score before rounding (users × items). */
  score: Tensor
  /** Observed training entries as rows (user, item, rating), m × 3. */
  train: Tensor
  /** Observed held-out entries, as rows (user, item, rating). */
  test: Tensor
  /** Mean training rating. */
  mean: number
  /** The true user and item factors (users × rank, items × rank). */
  userFactors: Tensor
  itemFactors: Tensor
  meta: DatasetMeta
}

/** Options for `ratings`. */
export interface RatingsOptions {
  users?: number
  items?: number
  /** Rank of the taste model. Default 2. */
  rank?: number
  /** Fraction of entries observed. Default 0.45. */
  observed?: number
  /** Share of observed entries held out for testing. Default 0.25. */
  testShare?: number
  /** Noise standard deviation before rounding. Default 0.3. */
  noise?: number
  /** Ratings are rounded and clipped to this integer scale. Default [1, 5]. */
  scale?: readonly [number, number]
}

/**
 * A ratings matrix from a low-rank taste model: user and item factors w_u, v_i ~ N(0, I), score 3 + 0.9 w_uᵀv_i + ε,
 * rounded and clipped to the rating scale. A random fraction of entries is observed; of those, `testShare` are held
 * out. Rounding and clipping are part of the recipe (they are the data-generating process), not numerical guards.
 */
export function ratings(s: Stream, options: RatingsOptions = {}): Ratings {
  const { users = 12, items = 16, rank = 2, observed = 0.45, testShare = 0.25, noise = 0.3, scale = [1, 5] } = options
  const mid = (scale[0] + scale[1]) / 2
  const fs = s.child('factors')
  const W = Float64Array.from({ length: users * rank }, () => normal(fs))
  const V = Float64Array.from({ length: items * rank }, () => normal(fs))
  const score = new Float64Array(users * items)
  const truth = new Float64Array(users * items)
  const eps = s.child('noise')
  for (let u = 0; u < users; u++)
    for (let i = 0; i < items; i++) {
      let dot = 0
      for (let r = 0; r < rank; r++) dot += W[u * rank + r] * V[i * rank + r]
      score[u * items + i] = mid + 0.9 * dot
      truth[u * items + i] = Math.round(
        Math.min(scale[1], Math.max(scale[0], score[u * items + i] + noise * normal(eps))),
      )
    }
  const train: number[] = []
  const test: number[] = []
  const mask = s.child('mask')
  for (let u = 0; u < users; u++)
    for (let i = 0; i < items; i++) {
      const p = mask.uniform()
      if (p < observed * (1 - testShare)) train.push(u, i, truth[u * items + i])
      else if (p < observed) test.push(u, i, truth[u * items + i])
    }
  let mean = 0
  for (let k = 2; k < train.length; k += 3) mean += train[k]
  mean /= Math.max(1, train.length / 3)
  return {
    users,
    items,
    truth: matrix(truth, users, items),
    score: matrix(score, users, items),
    train: matrix(Float64Array.from(train), train.length / 3, 3),
    test: matrix(Float64Array.from(test), test.length / 3, 3),
    mean,
    userFactors: matrix(W, users, rank),
    itemFactors: matrix(V, items, rank),
    meta: {
      name: 'ratings',
      description: `${users} users rating ${items} items on a ${scale[0]}–${scale[1]} scale from a rank-${rank} taste model; ${Math.round(observed * 100)}% of ratings observed.`,
      task: 'recommendation',
      featureNames: ['user', 'item', 'rating'],
      stream: s.key,
    },
  }
}

/** A click log: one row per (session, rank). */
export interface ClickLog {
  session: Tensor
  /** Rank in the shown list, 1-based. */
  rank: Tensor
  item: Tensor
  /** 1 if the user examined the result, else 0. */
  examined: Tensor
  clicked: Tensor
  /** P(examine | rank) under the model (for the cascade model, given the clicks above). */
  propensity: Tensor
  /** The true click probability given examination, per item. */
  relevance: Tensor
  meta: DatasetMeta
}

/** Options for `clickLog`. */
export interface ClickLogOptions {
  sessions?: number
  /** P(click | examined) per item; item 0 is the most relevant. Default ten items from 0.62 down to 0.15. */
  relevance?: readonly number[]
  /** `position`: examination π(k) = k^{−η} independent of relevance; `cascade`: users scan down and stop at a click. */
  model?: 'position' | 'cascade'
  /** Position-bias exponent η. Default 1. */
  eta?: number
  /** The logging ranker sorts items by relevance plus N(0, rankerNoise²), so rank is confounded with relevance. */
  rankerNoise?: number
}

/**
 * Simulated search sessions under a click model (Craswell, Zoeter, Taylor and Ramsey, 2008, "An experimental comparison
 * of click position-bias models", WSDM): each session ranks every item by noisy relevance; under the position-based
 * model the result at rank k is examined with probability k^{−η}; under the cascade model the user reads from the top
 * and stops after the first click. A click needs examination and then happens with the item's relevance.
 */
export function clickLog(s: Stream, options: ClickLogOptions = {}): ClickLog {
  const {
    sessions = 200,
    relevance = [0.62, 0.55, 0.5, 0.42, 0.4, 0.33, 0.3, 0.24, 0.2, 0.15],
    model = 'position',
    eta = 1,
    rankerNoise = 0.1,
  } = options
  checkCount(sessions, 'clickLog')
  const k = relevance.length
  const rows = sessions * k
  const cols = {
    session: new Int32Array(rows),
    rank: new Int32Array(rows),
    item: new Int32Array(rows),
    examined: new Float64Array(rows),
    clicked: new Float64Array(rows),
    propensity: new Float64Array(rows),
  }
  for (let q = 0; q < sessions; q++) {
    const r = s.child('session', q)
    const order = relevance
      .map((rel, i) => ({ i, score: rel + rankerNoise * normal(r) }))
      .sort((a, b) => b.score - a.score)
    let reading = true
    order.forEach(({ i }, pos) => {
      const row = q * k + pos
      const exam = model === 'position' ? (pos + 1) ** -eta : reading ? 1 : 0
      const examined = r.uniform() < exam
      const clicked = examined && r.uniform() < relevance[i]
      if (model === 'cascade' && clicked) reading = false
      cols.session[row] = q
      cols.rank[row] = pos + 1
      cols.item[row] = i
      cols.examined[row] = examined ? 1 : 0
      cols.clicked[row] = clicked ? 1 : 0
      cols.propensity[row] = exam
    })
  }
  return {
    session: labels(cols.session),
    rank: labels(cols.rank),
    item: labels(cols.item),
    examined: fromData(cols.examined),
    clicked: fromData(cols.clicked),
    propensity: fromData(cols.propensity),
    relevance: vector(relevance),
    meta: {
      name: 'click log',
      description: `${sessions} sessions of ${k} ranked results under the ${model === 'position' ? `position-based model (η = ${eta})` : 'cascade model'}.`,
      task: 'recommendation',
      featureNames: ['session', 'rank', 'item', 'examined', 'clicked', 'propensity'],
      source: 'Craswell et al. (2008), WSDM',
      stream: s.key,
    },
  }
}
