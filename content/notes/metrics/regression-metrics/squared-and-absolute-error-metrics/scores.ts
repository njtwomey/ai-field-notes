/** Regression and correlation scores for the outlier figure. Each takes true values y and predictions p. */

const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length

export const mse = (y: number[], p: number[]) => mean(y.map((v, i) => (v - p[i]) ** 2))
export const mae = (y: number[], p: number[]) => mean(y.map((v, i) => Math.abs(v - p[i])))

/** Coefficient of determination 1 − SSE/SST, with the mean of y as the baseline. */
export function r2(y: number[], p: number[]): number {
  const m = mean(y)
  const sse = y.reduce((s, v, i) => s + (v - p[i]) ** 2, 0)
  const sst = y.reduce((s, v) => s + (v - m) ** 2, 0)
  return 1 - sse / sst
}

export function pearson(x: number[], y: number[]): number {
  const mx = mean(x)
  const my = mean(y)
  let sxy = 0
  let sxx = 0
  let syy = 0
  for (let i = 0; i < x.length; i++) {
    sxy += (x[i] - mx) * (y[i] - my)
    sxx += (x[i] - mx) ** 2
    syy += (y[i] - my) ** 2
  }
  return sxy / Math.sqrt(sxx * syy)
}

/** Ranks from 1, with tied values given the average of their ranks. */
export function ranks(v: number[]): number[] {
  const order = v.map((x, i) => [x, i] as const).sort((a, b) => a[0] - b[0])
  const out = new Array<number>(v.length)
  for (let i = 0; i < order.length;) {
    let j = i
    while (j + 1 < order.length && order[j + 1][0] === order[i][0]) j++
    for (let k = i; k <= j; k++) out[order[k][1]] = (i + j) / 2 + 1
    i = j + 1
  }
  return out
}

export const spearman = (x: number[], y: number[]) => pearson(ranks(x), ranks(y))

/** Lin's concordance correlation coefficient, with population (1/n) moments as in Lin (1989). */
export function concordance(x: number[], y: number[]): number {
  const mx = mean(x)
  const my = mean(y)
  const sxy = mean(x.map((v, i) => (v - mx) * (y[i] - my)))
  const sxx = mean(x.map((v) => (v - mx) ** 2))
  const syy = mean(y.map((v) => (v - my) ** 2))
  return (2 * sxy) / (sxx + syy + (mx - my) ** 2)
}
