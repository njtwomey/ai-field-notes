/**
 * The skills model of Winn and Bishop's "Assessing people's skills": binary skills with a Bernoulli prior, and each
 * question answered correctly with probability `pKnow` if the candidate has every skill it needs, else `pGuess`.
 * Exact posteriors by enumeration, and loopy belief propagation with a sequential schedule for comparison.
 */

export type SkillsModel = {
  nSkills: number
  /** Skills needed by each question. */
  questions: number[][]
  prior: number
  pKnow: number
  pGuess: number
}

const likelihood = (m: SkillsModel, hasAll: boolean, correct: boolean) => {
  const p = hasAll ? m.pKnow : m.pGuess
  return correct ? p : 1 - p
}

/** Posterior probability of each skill, by summing over all 2^nSkills skill combinations. */
export function exactPosterior(m: SkillsModel, answers: boolean[]): number[] {
  const post = Array(m.nSkills).fill(0)
  let z = 0
  for (let mask = 0; mask < 1 << m.nSkills; mask++) {
    const has = (k: number) => ((mask >> k) & 1) === 1
    let p = 1
    for (let k = 0; k < m.nSkills; k++) p *= has(k) ? m.prior : 1 - m.prior
    m.questions.forEach((q, f) => {
      p *= likelihood(
        m,
        q.every((k) => has(k)),
        answers[f],
      )
    })
    z += p
    for (let k = 0; k < m.nSkills; k++) if (has(k)) post[k] += p
  }
  return post.map((v) => v / z)
}

/**
 * Loopy belief propagation. Messages are stored as P(skill = true) after normalisation and start uniform (0.5). One
 * sweep visits every question factor in order and recomputes its messages to its skills. Returns the skill marginals
 * before any sweep (the prior) and after each of `sweeps` sweeps. On a tree the result after one sweep is exact.
 */
export function loopyHistory(m: SkillsModel, answers: boolean[], sweeps: number): number[][] {
  const msg = m.questions.map((q) => q.map(() => 0.5))
  const toFactor = (f: number, k: number) => {
    let p1 = m.prior
    let p0 = 1 - m.prior
    m.questions.forEach((q, g) => {
      const j = q.indexOf(k)
      if (g !== f && j >= 0) {
        p1 *= msg[g][j]
        p0 *= 1 - msg[g][j]
      }
    })
    return p1 / (p1 + p0)
  }
  const marginals = () =>
    Array.from({ length: m.nSkills }, (_, k) => {
      let p1 = m.prior
      let p0 = 1 - m.prior
      m.questions.forEach((q, f) => {
        const j = q.indexOf(k)
        if (j >= 0) {
          p1 *= msg[f][j]
          p0 *= 1 - msg[f][j]
        }
      })
      return p1 / (p1 + p0)
    })
  const history = [marginals()]
  for (let s = 0; s < sweeps; s++) {
    m.questions.forEach((q, f) => {
      q.forEach((k, j) => {
        // Probability that every other skill the question needs is present, from the incoming messages.
        const others = q.filter((o) => o !== k).reduce((acc, o) => acc * toFactor(f, o), 1)
        const l1 = likelihood(m, true, answers[f]) * others + likelihood(m, false, answers[f]) * (1 - others)
        const l0 = likelihood(m, false, answers[f])
        msg[f][j] = l1 / (l1 + l0)
      })
    })
    history.push(marginals())
  }
  return history
}
