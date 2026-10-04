import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { normal, stream, uniform } from 'aifn/foundation/random'
import { normalCdf } from 'aifn/numerics/special'

const DOCS = 10
/** Click probability given examination for each document; document 0 is the best. */
const REL = [0.6, 0.5, 0.42, 0.35, 0.3, 0.25, 0.2, 0.16, 0.12, 0.1]
const RANKER_A = Array.from({ length: DOCS }, (_, i) => i)
const USERS_SAMPLE = 400
const LOGN = Array.from({ length: 61 }, (_, i) => 1.5 + (3.5 * i) / 60)
const Z = 1.959964

/** Ranker B reverses A's top `m` documents: the same documents, worse ordered at the top. */
const rankerB = (m: number) => [...RANKER_A.slice(0, m).reverse(), ...RANKER_A.slice(m)]

/**
 * Every team-draft interleaving of rankings a and b, equally likely. Each round a fair coin decides which team picks
 * first, and each team adds its highest-ranked document not yet in the list. Returns, for each outcome, the team
 * (0 = A, 1 = B) credited with the document at each position.
 */
function teamDraftOutcomes(a: number[], b: number[]): { list: number[]; team: number[] }[] {
  const rounds = Math.ceil(DOCS / 2)
  const out: { list: number[]; team: number[] }[] = []
  for (let mask = 0; mask < 2 ** rounds; mask++) {
    const list: number[] = []
    const team: number[] = []
    const used = new Set<number>()
    for (let round = 0; round < rounds && list.length < DOCS; round++) {
      const order = (mask >> round) & 1 ? [1, 0] : [0, 1]
      for (const t of order) {
        const d = (t === 0 ? a : b).find((x) => !used.has(x))
        if (d === undefined || list.length >= DOCS) continue
        used.add(d)
        list.push(d)
        team.push(t)
      }
    }
    out.push({ list, team })
  }
  return out
}

type Stats = { muA: number; muB: number; varA: number; varB: number; pA: number; pB: number }

/** Draws of a Gamma activity factor with mean 1 and coefficient of variation cv (Marsaglia–Tsang), fixed seed. */
function activities(cv: number, n: number): number[] {
  if (cv < 1e-3) return new Array(n).fill(1)
  const r = stream(17)
  const shape = 1 / (cv * cv)
  const boost = shape < 1
  const k = boost ? shape + 1 : shape
  const d = k - 1 / 3
  const c = 1 / Math.sqrt(9 * d)
  return Array.from({ length: n }, () => {
    let x = 0
    for (;;) {
      const z = normal(r)
      const v = (1 + c * z) ** 3
      if (v <= 0) continue
      if (Math.log(uniform(r)) < 0.5 * z * z + d - d * v + d * Math.log(v)) {
        x = d * v
        break
      }
    }
    if (boost) x *= uniform(r) ** (1 / shape)
    return x / shape
  })
}

/**
 * Per-session quantities, computed exactly given each user's activity and averaged over a fixed sample of users. A user
 * with activity α clicks the document at rank k with probability min(1, α·π(k)·r), π(k) = k^(−η), independently.
 * A/B: mean and variance of clicks per session. Interleaving: probability that A's documents get more clicks than B's,
 * and the reverse, by exact convolution over positions and enumeration of team-draft coin flips.
 */
function compute(m: number, cv: number, eta: number): Stats {
  const b = rankerB(m)
  const prop = Array.from({ length: DOCS }, (_, k) => (k + 1) ** -eta)
  const outcomes = teamDraftOutcomes(RANKER_A, b)
  const acts = activities(cv, USERS_SAMPLE)
  let muA = 0
  let muB = 0
  let m2A = 0
  let m2B = 0
  let pA = 0
  let pB = 0
  const moments = (ranking: number[], act: number) => {
    let mean = 0
    let v = 0
    ranking.forEach((d, k) => {
      const p = Math.min(1, act * prop[k] * REL[d])
      mean += p
      v += p * (1 - p)
    })
    return { mean, second: v + mean * mean }
  }
  for (const act of acts) {
    const A = moments(RANKER_A, act)
    const B = moments(b, act)
    muA += A.mean
    muB += B.mean
    m2A += A.second
    m2B += B.second
    for (const { list, team } of outcomes) {
      // Distribution of credit = (A clicks) − (B clicks), offset by DOCS.
      let dist = new Array(2 * DOCS + 1).fill(0)
      dist[DOCS] = 1
      list.forEach((d, k) => {
        const p = Math.min(1, act * prop[k] * REL[d])
        const step = team[k] === 0 ? 1 : -1
        const next = new Array(2 * DOCS + 1).fill(0)
        dist.forEach((q, i) => {
          if (!q) return
          next[i] += q * (1 - p)
          next[i + step] += q * p
        })
        dist = next
      })
      const w = 1 / outcomes.length
      dist.forEach((q, i) => {
        if (i > DOCS) pA += w * q
        else if (i < DOCS) pB += w * q
      })
    }
  }
  const n = acts.length
  muA /= n
  muB /= n
  return { muA, muB, varA: m2A / n - muA * muA, varB: m2B / n - muB * muB, pA: pA / n, pB: pB / n }
}

/** Two-sided power of a z-test with standardised effect `effect` per unit and n units. */
const power = (effect: number, n: number) => normalCdf(Math.abs(effect) * Math.sqrt(n) - Z)

/** Power of an A/B test on clicks per session against team-draft interleaving, as the number of users grows. */
export function InterleavingPower() {
  const state = useFigureState({
    swap: int(3, { min: 2, max: 10, step: 1, label: 'top documents reversed by B', format: (v) => String(v) }),
    cv: float(1, { min: 0, max: 2, step: 0.1, label: 'user heterogeneity (CV)' }),
    eta: float(1, { min: 0.2, max: 2, step: 0.1, label: 'position bias η' }),
  })
  const s = useMemo(() => compute(state.swap, state.cv, state.eta), [state.swap, state.cv, state.eta])

  // A/B: N users split evenly; the difference of means has variance (varA + varB)/(N/2).
  const abEffect = (s.muA - s.muB) / Math.sqrt(2 * (s.varA + s.varB))
  // Interleaving: every user is one session; the per-session preference is +1, −1 or 0.
  const m = s.pA - s.pB
  const ilEffect = m / Math.sqrt(Math.max(s.pA + s.pB - m * m, 1e-9))
  const series = [
    {
      name: 'A/B test on clicks per session',
      x: LOGN,
      y: LOGN.map((l) => power(abEffect, 10 ** l)),
      slot: 0,
    },
    { name: 'team-draft interleaving', x: LOGN, y: LOGN.map((l) => power(ilEffect, 10 ** l)), slot: 1 },
  ] as const
  const needed = (effect: number) => (effect === 0 ? Infinity : ((Z + 0.841621) / effect) ** 2)
  const nAb = needed(abEffect)
  const nIl = needed(ilEffect)

  const xAxis = useAxis({ label: 'log₁₀ total users', range: [LOGN[0], LOGN[LOGN.length - 1]] })
  const yAxis = useAxis({ label: 'power', range: [0, 1] })
  return (
    <Figure
      title="How much traffic each design needs"
      state={state}
      caption="Ranker A orders ten documents by quality; ranker B shows the same documents but reverses the order of the top few. Users differ in how much they click (a Gamma activity factor with the given coefficient of variation) and examine rank k with probability k^(−η). The A/B test compares clicks per session between two groups of users; interleaving shows every user one team-draft list and counts whose documents got more clicks. Quantities are computed exactly for a fixed sample of 400 users, and the chart shows each design's power at the 5% level against the total number of users. Heterogeneous users inflate the A/B test's noise, and weak position bias makes clicks per session almost blind to order; interleaving is largely immune to both because each user compares the two rankers directly."

      readouts={
        <>
          <Readout label="clicks per session A / B" value={`${formatNumber(s.muA)} / ${formatNumber(s.muB)}`} />
          <Readout label="interleaving wins A / B" value={`${formatNumber(s.pA)} / ${formatNumber(s.pB)}`} />
          <Readout
            label="users for 80% power, A/B"
            value={Number.isFinite(nAb) ? Math.round(nAb).toLocaleString() : '∞'}
          />
          <Readout label="interleaving" value={Number.isFinite(nIl) ? Math.round(nIl).toLocaleString() : '∞'} />
          <Readout label="ratio" value={Number.isFinite(nAb / nIl) ? `${formatNumber(nAb / nIl)}×` : '–'} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
      </Plot>
    </Figure>
  )
}
