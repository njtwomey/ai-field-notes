/** Small dense linear algebra for the unsupervised-learning widgets: matrices are arrays of rows. */

export type Matrix = number[][]

/** Column means of the rows of x. */
export function colMeans(x: Matrix): number[] {
  const d = x[0]?.length ?? 0
  const m = new Array<number>(d).fill(0)
  for (const row of x) for (let j = 0; j < d; j++) m[j] += row[j] / x.length
  return m
}

/** Covariance with divisor n of the rows of x. */
export function covariance(x: Matrix): Matrix {
  const d = x[0]?.length ?? 0
  const m = colMeans(x)
  const s = Array.from({ length: d }, () => new Array<number>(d).fill(0))
  for (const row of x)
    for (let a = 0; a < d; a++) for (let b = a; b < d; b++) s[a][b] += ((row[a] - m[a]) * (row[b] - m[b])) / x.length
  for (let a = 0; a < d; a++) for (let b = 0; b < a; b++) s[a][b] = s[b][a]
  return s
}

/**
 * Eigenvalues and eigenvectors of a symmetric matrix by cyclic Jacobi rotations. Values are sorted in decreasing
 * order; `vectors[j]` is the unit eigenvector of `values[j]`. Fine for the d ≤ 50 matrices the widgets use.
 */
export function eigSymmetric(a: Matrix, sweeps = 60): { values: number[]; vectors: number[][] } {
  const n = a.length
  const m = a.map((r) => [...r])
  const v: Matrix = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)))
  for (let sweep = 0; sweep < sweeps; sweep++) {
    let off = 0
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += m[p][q] ** 2
    if (off < 1e-20) break
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(m[p][q]) < 1e-300) continue
        const theta = (m[q][q] - m[p][p]) / (2 * m[p][q])
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1))
        const c = 1 / Math.sqrt(t * t + 1)
        const s = t * c
        for (let k = 0; k < n; k++) {
          const mkp = m[k][p]
          const mkq = m[k][q]
          m[k][p] = c * mkp - s * mkq
          m[k][q] = s * mkp + c * mkq
        }
        for (let k = 0; k < n; k++) {
          const mpk = m[p][k]
          const mqk = m[q][k]
          m[p][k] = c * mpk - s * mqk
          m[q][k] = s * mpk + c * mqk
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k][p]
          const vkq = v[k][q]
          v[k][p] = c * vkp - s * vkq
          v[k][q] = s * vkp + c * vkq
        }
      }
    }
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => m[j][j] - m[i][i])
  return { values: order.map((i) => m[i][i]), vectors: order.map((i) => v.map((row) => row[i])) }
}
