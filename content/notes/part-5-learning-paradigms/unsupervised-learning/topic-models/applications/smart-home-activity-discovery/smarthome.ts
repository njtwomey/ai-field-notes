/**
 * A simulated smart home, the document segmentation of Chen, Diethe and Flach, and a collapsed Gibbs sampler for their
 * one-topic-per-document model with unigrams and (optionally) bigrams.
 */
import { categorical, normal, stream, uniform, type Stream } from 'aifn/foundation/random'

export const LOCATIONS = ['bedroom', 'bathroom', 'kitchen', 'lounge', 'hall'] as const

/** Binary sensors: name and location index. Each firing emits an ON and an OFF event. */
export const SENSORS: { name: string; loc: number }[] = [
  { name: 'bed pressure', loc: 0 },
  { name: 'bedroom motion', loc: 0 },
  { name: 'wardrobe door', loc: 0 },
  { name: 'bathroom motion', loc: 1 },
  { name: 'toilet flush', loc: 1 },
  { name: 'shower', loc: 1 },
  { name: 'kitchen motion', loc: 2 },
  { name: 'fridge door', loc: 2 },
  { name: 'hob', loc: 2 },
  { name: 'cupboard', loc: 2 },
  { name: 'lounge motion', loc: 3 },
  { name: 'TV', loc: 3 },
  { name: 'hall motion', loc: 4 },
  { name: 'front door', loc: 4 },
]
/** Words are (sensor, state): ON is 2s, OFF is 2s + 1. */
export const V = SENSORS.length * 2

export const ACTIVITIES = ['sleep', 'toilet', 'shower', 'dress', 'cook', 'relax', 'leave/return'] as const

/** Firing rates (per minute) of each sensor during each activity. */
const RATES: Record<number, [number, number][]> = {
  0: [
    [0, 0.03],
    [1, 0.01],
  ],
  1: [
    [3, 0.8],
    [4, 0.25],
  ],
  2: [
    [3, 0.4],
    [5, 0.35],
  ],
  3: [
    [1, 0.6],
    [2, 0.5],
  ],
  4: [
    [6, 0.6],
    [7, 0.25],
    [8, 0.2],
    [9, 0.3],
  ],
  5: [
    [10, 0.12],
    [11, 0.04],
  ],
  6: [
    [12, 1],
    [13, 0.6],
  ],
}

export type Event = { t: number; word: number; sensor: number; activity: number }
type Span = { a: number; s: number; e: number }

/** One day's schedule, in minutes after midnight, with seeded jitter. */
function schedule(r: Stream): Span[] {
  const j = (sd: number) => sd * normal(r)
  const spans: Span[] = []
  const wake = 420 + j(25)
  const nightToilet = 120 + uniform(r) * 180
  spans.push({ a: 0, s: 0, e: nightToilet }, { a: 1, s: nightToilet, e: nightToilet + 5 })
  spans.push({ a: 0, s: nightToilet + 5, e: wake })
  let t = wake
  const add = (a: number, dur: number) => {
    spans.push({ a, s: t, e: t + dur })
    t += dur
  }
  add(1, 6)
  add(2, 14 + j(3))
  add(3, 10 + j(2))
  add(4, 20 + j(4))
  const out = uniform(r) < 0.6
  add(5, 540 - t + j(15))
  if (out) {
    add(6, 3)
    t += 150 + j(30) // nobody home: no events
    add(6, 3)
  }
  add(5, 750 - t + j(15))
  add(4, 30 + j(5))
  add(5, 900 - t + j(30))
  add(1, 5)
  add(5, 1110 - t + j(20))
  add(4, 45 + j(8))
  add(5, 1350 - t + j(20))
  add(1, 5)
  spans.push({ a: 0, s: t, e: 1440 })
  return spans
}

/** Events of one day: Poisson firings of each activity's sensors; each firing is ON then OFF 0.2–2 minutes later. */
function simulateDay(r: Stream, offset: number): Event[] {
  const events: Event[] = []
  for (const { a, s, e } of schedule(r)) {
    for (const [sensor, rate] of RATES[a]) {
      let t = s - Math.log(1 - uniform(r)) / rate
      while (t < e) {
        const off = Math.min(t + 0.2 + 1.8 * uniform(r), e)
        events.push({ t: offset + t, word: 2 * sensor, sensor, activity: a })
        events.push({ t: offset + off, word: 2 * sensor + 1, sensor, activity: a })
        t -= Math.log(1 - uniform(r)) / rate
      }
    }
  }
  return events.sort((x, y) => x.t - y.t)
}

export const DAYS = 14

export function simulate(seed: number): Event[] {
  const r = stream(seed)
  return Array.from({ length: DAYS }, (_, d) => simulateDay(r, d * 1440)).flat()
}

/**
 * Algorithm 1 of Chen, Diethe and Flach: walk the stream location run by location run, and close a document at the end
 * of a run once the document has lasted longer than `threshold` minutes. Returns [start, stop] event indices.
 */
export function segment(events: Event[], threshold: number): [number, number][] {
  const loc = events.map((e) => SENSORS[e.sensor].loc)
  const docs: [number, number][] = []
  let start = 0
  let stop = 0
  while (stop < events.length) {
    let next = stop
    while (next < events.length && loc[next] === loc[stop]) next++
    const end = next - 1
    if (events[end].t - events[start].t > threshold || next === events.length) {
      docs.push([start, end])
      start = next
    }
    stop = next
  }
  return docs
}

export type Fit = { z: number[]; unigram: number[][]; docCount: number[] }

/**
 * Collapsed Gibbs sampling for a mixture in which each document has one topic and generates its unigrams from ν_k and,
 * optionally, its bigrams from a topic-specific transition matrix Φ_k. The document's likelihood is computed exactly,
 * adding its own tokens to the counts one at a time.
 */
export function fit(
  events: Event[],
  docs: [number, number][],
  K: number,
  bigrams: boolean,
  seed: number,
  sweeps = 30,
): Fit {
  const alpha = 50 / K
  const gamma = 5 / V
  const beta = 5 / V
  const r = stream(seed)
  const words = docs.map(([s, e]) => events.slice(s, e + 1).map((ev) => ev.word))
  const nk = new Array<number>(K).fill(0)
  const uni = Array.from({ length: K }, () => new Array<number>(V).fill(0))
  const uniTot = new Array<number>(K).fill(0)
  const bi = Array.from({ length: K }, () => new Float64Array(V * V))
  const biTot = Array.from({ length: K }, () => new Float64Array(V))
  const z = words.map(() => Math.floor(uniform(r) * K))

  const apply = (d: number, k: number, sign: number) => {
    nk[k] += sign
    const ws = words[d]
    for (let i = 0; i < ws.length; i++) {
      uni[k][ws[i]] += sign
      uniTot[k] += sign
      if (bigrams && i > 0) {
        bi[k][ws[i - 1] * V + ws[i]] += sign
        biTot[k][ws[i - 1]] += sign
      }
    }
  }
  words.forEach((_, d) => apply(d, z[d], 1))

  const logLik = (d: number, k: number) => {
    const ws = words[d]
    const seenU = new Map<number, number>()
    const seenB = new Map<number, number>()
    const seenPrev = new Map<number, number>()
    let ll = 0
    for (let i = 0; i < ws.length; i++) {
      const w = ws[i]
      const cu = seenU.get(w) ?? 0
      ll += Math.log((uni[k][w] + cu + gamma) / (uniTot[k] + i + V * gamma))
      seenU.set(w, cu + 1)
      if (bigrams && i > 0) {
        const p = ws[i - 1]
        const key = p * V + w
        const cb = seenB.get(key) ?? 0
        const cp = seenPrev.get(p) ?? 0
        ll += Math.log((bi[k][key] + cb + beta) / (biTot[k][p] + cp + V * beta))
        seenB.set(key, cb + 1)
        seenPrev.set(p, cp + 1)
      }
    }
    return ll
  }

  for (let s = 0; s < sweeps; s++) {
    for (let d = 0; d < words.length; d++) {
      apply(d, z[d], -1)
      const lp = Array.from({ length: K }, (_, k) => Math.log(nk[k] + alpha) + logLik(d, k))
      const m = Math.max(...lp)
      z[d] = categorical(
        r,
        lp.map((v) => Math.exp(v - m)),
      )
      apply(d, z[d], 1)
    }
  }
  return {
    z,
    unigram: uni.map((row, k) => row.map((c) => (c + gamma) / (uniTot[k] + V * gamma))),
    docCount: nk,
  }
}

/** Map each topic to the activity that produced most of its events (several topics may map to one activity). */
export function topicToActivity(events: Event[], docs: [number, number][], z: number[], K: number): number[] {
  const counts = Array.from({ length: K }, () => new Array<number>(ACTIVITIES.length).fill(0))
  docs.forEach(([s, e], d) => {
    for (let i = s; i <= e; i++) counts[z[d]][events[i].activity]++
  })
  return counts.map((row) => row.indexOf(Math.max(...row)))
}
