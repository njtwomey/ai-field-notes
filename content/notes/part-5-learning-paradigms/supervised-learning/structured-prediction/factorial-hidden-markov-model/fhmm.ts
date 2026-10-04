import { normal, stream, uniform } from 'aifn/foundation/random'
/**
 * An additive factorial HMM with binary chains: appliance m is off (0) or on (1), draws power P_m when on, and the
 * meter reads x_n = Σ_m P_m y_n^(m) + ε with ε ~ N(0, σ²). Exact inference runs forward–backward on the 2^M product
 * states; structured mean field runs one small forward–backward per chain against the others' expected power.
 */

export type Appliance = { name: string; power: number; turnOn: number; turnOff: number }

export const APPLIANCES: Appliance[] = [
  { name: 'fridge', power: 150, turnOn: 0.08, turnOff: 0.1 },
  { name: 'heater', power: 1000, turnOn: 0.02, turnOff: 0.05 },
  { name: 'kettle', power: 2000, turnOn: 0.03, turnOff: 0.25 },
  { name: 'microwave', power: 800, turnOn: 0.02, turnOff: 0.15 },
]

const chainTransition = (a: Appliance) => [
  [1 - a.turnOn, a.turnOn],
  [a.turnOff, 1 - a.turnOff],
]
const stationaryOn = (a: Appliance) => a.turnOn / (a.turnOn + a.turnOff)

export function simulate(apps: Appliance[], n: number, sigma: number, seed: number) {
  const g = stream(seed)
  const states = apps.map((a) => {
    const s: number[] = []
    let on = uniform(g) < stationaryOn(a) ? 1 : 0
    for (let t = 0; t < n; t++) {
      if (t > 0) on = uniform(g) < chainTransition(a)[on][1] ? 1 : 0
      s.push(on)
    }
    return s
  })
  const x = Array.from(
    { length: n },
    (_, t) => apps.reduce((s, a, m) => s + a.power * states[m][t], 0) + sigma * normal(g),
  )
  return { states, x }
}

/** Two-state forward–backward with node potentials already including the initial distribution. Returns p(on). */
function chainPosterior(A: number[][], psi: number[][]): number[] {
  const N = psi.length
  const f: number[][] = []
  const scale: number[] = []
  for (let n = 0; n < N; n++) {
    const a = n === 0 ? [1, 1] : [0, 1].map((v) => f[n - 1][0] * A[0][v] + f[n - 1][1] * A[1][v])
    const g = [a[0] * psi[n][0], a[1] * psi[n][1]]
    const c = g[0] + g[1]
    scale.push(c)
    f.push([g[0] / c, g[1] / c])
  }
  const beta: number[][] = Array.from({ length: N }, () => [1, 1])
  for (let n = N - 2; n >= 0; n--) {
    const d = [beta[n + 1][0] * psi[n + 1][0], beta[n + 1][1] * psi[n + 1][1]]
    beta[n] = [0, 1].map((u) => (A[u][0] * d[0] + A[u][1] * d[1]) / scale[n + 1])
  }
  return f.map((fi, n) => {
    const on = fi[1] * beta[n][1]
    return on / (fi[0] * beta[n][0] + on)
  })
}

/** Exact posterior p(y_n^(m) = on | x) by forward–backward over all 2^M joint states. */
export function exactPosterior(apps: Appliance[], x: number[], sigma: number): number[][] {
  const M = apps.length
  const S = 1 << M
  const bit = (s: number, m: number) => (s >> m) & 1
  const trans = apps.map(chainTransition)
  // The joint transition factorises over chains: A(s, s') = Π_m A_m(s_m, s'_m).
  const A = Array.from({ length: S }, (_, s) =>
    Array.from({ length: S }, (_, t) => apps.reduce((p, _a, m) => p * trans[m][bit(s, m)][bit(t, m)], 1)),
  )
  const prior = Array.from({ length: S }, (_, s) =>
    apps.reduce((p, a, m) => p * (bit(s, m) ? stationaryOn(a) : 1 - stationaryOn(a)), 1),
  )
  const mean = Array.from({ length: S }, (_, s) => apps.reduce((p, a, m) => p + a.power * bit(s, m), 0))
  const N = x.length
  // Emission likelihoods relative to their maximum at each step, which keeps the numbers in range.
  const psi = x.map((xn, n) => {
    const logs = mean.map((mu) => (-0.5 * (xn - mu) ** 2) / (sigma * sigma))
    const top = Math.max(...logs)
    return logs.map((l, s) => Math.exp(l - top) * (n === 0 ? prior[s] : 1))
  })
  const f: number[][] = []
  const scale: number[] = []
  for (let n = 0; n < N; n++) {
    const a =
      n === 0
        ? Array(S).fill(1)
        : Array.from({ length: S }, (_, v) => f[n - 1].reduce((acc, g, u) => acc + g * A[u][v], 0))
    const g = a.map((ai, s) => ai * psi[n][s])
    const c = g.reduce((acc, gi) => acc + gi, 0)
    scale.push(c)
    f.push(g.map((gi) => gi / c))
  }
  const beta: number[][] = Array.from({ length: N }, () => Array(S).fill(1))
  for (let n = N - 2; n >= 0; n--) {
    const d = beta[n + 1].map((b, v) => b * psi[n + 1][v])
    beta[n] = Array.from({ length: S }, (_, u) => A[u].reduce((acc, auv, v) => acc + auv * d[v], 0) / scale[n + 1])
  }
  return apps.map((_a, m) =>
    f.map((fi, n) => {
      let on = 0
      let total = 0
      for (let s = 0; s < S; s++) {
        const p = fi[s] * beta[n][s]
        total += p
        if (bit(s, m)) on += p
      }
      return on / total
    }),
  )
}

/**
 * Structured mean field: q(Y) = Π_m q_m(y^(m)), each q_m an HMM. Chain m sees the residual
 * r_n = x_n − Σ_{l≠m} P_l E[y_n^(l)] and the variational emission exp{(P_m r_n − P_m²/2)/σ²} for "on" (1 for "off").
 * Sweeping over the chains until nothing changes is coordinate ascent on the evidence lower bound.
 */
export function meanFieldPosterior(apps: Appliance[], x: number[], sigma: number, sweeps = 30): number[][] {
  const N = x.length
  const trans = apps.map(chainTransition)
  const q = apps.map((a) => Array(N).fill(stationaryOn(a)))
  for (let sweep = 0; sweep < sweeps; sweep++) {
    apps.forEach((a, m) => {
      const psi = x.map((xn, n) => {
        const others = apps.reduce((s, b, l) => (l === m ? s : s + b.power * q[l][n]), 0)
        const r = xn - others
        const logOn = (a.power * r - (a.power * a.power) / 2) / (sigma * sigma)
        // Normalise the pair [off, on] to avoid overflow; only the ratio matters.
        const on = logOn > 0 ? 1 : Math.exp(logOn)
        const off = logOn > 0 ? Math.exp(-logOn) : 1
        const p0 = n === 0 ? 1 - stationaryOn(a) : 1
        const p1 = n === 0 ? stationaryOn(a) : 1
        return [off * p0, on * p1]
      })
      q[m] = chainPosterior(trans[m], psi)
    })
  }
  return q
}
