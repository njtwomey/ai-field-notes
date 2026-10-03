/**
 * Non-parametric survival estimators and the log-rank test, for the figures in this category. Inputs are follow-up
 * times t and event indicators d (1 = event observed, 0 = right-censored).
 */

export type Step = {
  /** Distinct event time. */
  time: number
  /** Number at risk just before the time. */
  atRisk: number
  /** Number of events at the time. */
  events: number
  /** Kaplan–Meier survival just after the time. */
  surv: number
  /** Greenwood variance of the Kaplan–Meier estimate. */
  variance: number
  /** Nelson–Aalen cumulative hazard just after the time. */
  cumHazard: number
}

export function eventTable(t: number[], d: number[]): Step[] {
  const times = [...new Set(t.filter((_, i) => d[i] === 1))].sort((a, b) => a - b)
  let surv = 1
  let greenwood = 0
  let cumHazard = 0
  return times.map((time) => {
    const atRisk = t.filter((ti) => ti >= time).length
    const events = t.filter((ti, i) => ti === time && d[i] === 1).length
    surv *= 1 - events / atRisk
    greenwood += atRisk > events ? events / (atRisk * (atRisk - events)) : 0
    cumHazard += events / atRisk
    return { time, atRisk, events, surv, variance: surv * surv * greenwood, cumHazard }
  })
}

/** Points tracing a right-continuous step function from (0, 1), ending at tMax. */
export function stepPath(steps: { time: number; value: number }[], tMax: number, start = 1) {
  const x: number[] = [0]
  const y: number[] = [start]
  let last = start
  for (const s of steps) {
    if (s.time > tMax) break
    x.push(s.time, s.time)
    y.push(last, s.value)
    last = s.value
  }
  x.push(tMax)
  y.push(last)
  return { x, y }
}

/** Smallest time at which a survival curve reaches 0.5 or below; null when it never does. */
export function median(steps: { time: number; value: number }[]): number | null {
  return steps.find((s) => s.value <= 0.5)?.time ?? null
}

/**
 * Log-rank test of equal hazards in two groups (g = 1 or 0). Returns observed and expected events in group 1, the
 * hypergeometric variance summed over event times, and the chi-squared statistic on one degree of freedom.
 */
export function logRank(t: number[], d: number[], g: number[]) {
  const times = [...new Set(t.filter((_, i) => d[i] === 1))].sort((a, b) => a - b)
  let observed = 0
  let expected = 0
  let variance = 0
  for (const time of times) {
    let n = 0
    let n1 = 0
    let dd = 0
    let d1 = 0
    t.forEach((ti, i) => {
      if (ti >= time) {
        n++
        if (g[i] === 1) n1++
      }
      if (ti === time && d[i] === 1) {
        dd++
        if (g[i] === 1) d1++
      }
    })
    observed += d1
    expected += (dd * n1) / n
    if (n > 1) variance += (dd * (n1 / n) * (1 - n1 / n) * (n - dd)) / (n - 1)
  }
  const chi2 = variance > 0 ? (observed - expected) ** 2 / variance : 0
  return { observed, expected, variance, chi2 }
}
