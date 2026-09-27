/**
 * One child, one allergen, sensitisation at four ages as a two-state Markov chain, observed through noisy skin tests.
 * A latent class selects the chain's parameters (a gate). Exact inference by enumerating the 16 paths.
 */

export type ChainParams = { initial: number; gain: number; retain: number }
export type TestParams = { sensitivity: number; falsePositive: number }
/** A test result per age: true (positive), false (negative) or null (not done). */
export type Tests = (boolean | null)[]

const testLik = (t: boolean | null, sens: boolean, p: TestParams) => {
  if (t === null) return 1
  const pos = sens ? p.sensitivity : p.falsePositive
  return t ? pos : 1 - pos
}

/** Joint probability of every path, for one class. */
function paths(c: ChainParams, tests: Tests, p: TestParams) {
  const T = tests.length
  const out: { states: boolean[]; prob: number }[] = []
  for (let mask = 0; mask < 1 << T; mask++) {
    const states = Array.from({ length: T }, (_, t) => ((mask >> t) & 1) === 1)
    let prob = states[0] ? c.initial : 1 - c.initial
    for (let t = 1; t < T; t++) {
      const on = states[t - 1] ? c.retain : c.gain
      prob *= states[t] ? on : 1 - on
    }
    states.forEach((s, t) => (prob *= testLik(tests[t], s, p)))
    out.push({ states, prob })
  }
  return out
}

/**
 * Posterior over classes and posterior probability of sensitisation at each age, marginalising the class with prior
 * `classPrior`. With a single class of prior 1 this is plain forward–backward.
 */
export function posterior(classes: ChainParams[], classPrior: number[], tests: Tests, p: TestParams) {
  const T = tests.length
  const evidence = classes.map((c) => paths(c, tests, p))
  const z = evidence.map((e, k) => classPrior[k] * e.reduce((s, x) => s + x.prob, 0))
  const total = z.reduce((a, b) => a + b, 0)
  const sens = Array(T).fill(0)
  evidence.forEach((e, k) =>
    e.forEach(({ states, prob }) =>
      states.forEach((s, t) => {
        if (s) sens[t] += (classPrior[k] * prob) / total
      }),
    ),
  )
  return { classPosterior: z.map((v) => v / total), sensitised: sens }
}
