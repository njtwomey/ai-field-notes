/**
 * Full eigendecomposition of a dense symmetric matrix by Householder tridiagonalisation and the implicit QL algorithm
 * (tred2 and tql2, as in EISPACK and JAMA). O(n³) with a small constant: a 300 × 300 matrix takes a few tens of
 * milliseconds, where cyclic Jacobi takes seconds.
 */

/** Eigenvalues in decreasing order; `vectors[k]` is the unit eigenvector of `values[k]`. */
export function symmetricEigen(a: number[][]): { values: number[]; vectors: number[][] } {
  const n = a.length
  const V = a.map((row) => Float64Array.from(row))
  const d = new Float64Array(n)
  const e = new Float64Array(n)
  tred2(V, d, e)
  tql2(V, d, e)
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => d[j] - d[i])
  return {
    values: order.map((i) => d[i]),
    vectors: order.map((k) => Array.from({ length: n }, (_, i) => V[i][k])),
  }
}

function tred2(V: Float64Array[], d: Float64Array, e: Float64Array) {
  const n = V.length
  for (let j = 0; j < n; j++) d[j] = V[n - 1][j]
  for (let i = n - 1; i > 0; i--) {
    let scale = 0
    let h = 0
    for (let k = 0; k < i; k++) scale += Math.abs(d[k])
    if (scale === 0) {
      e[i] = d[i - 1]
      for (let j = 0; j < i; j++) {
        d[j] = V[i - 1][j]
        V[i][j] = 0
        V[j][i] = 0
      }
    } else {
      for (let k = 0; k < i; k++) {
        d[k] /= scale
        h += d[k] * d[k]
      }
      let f = d[i - 1]
      let g = Math.sqrt(h)
      if (f > 0) g = -g
      e[i] = scale * g
      h -= f * g
      d[i - 1] = f - g
      for (let j = 0; j < i; j++) e[j] = 0
      for (let j = 0; j < i; j++) {
        f = d[j]
        V[j][i] = f
        g = e[j] + V[j][j] * f
        for (let k = j + 1; k <= i - 1; k++) {
          g += V[k][j] * d[k]
          e[k] += V[k][j] * f
        }
        e[j] = g
      }
      f = 0
      for (let j = 0; j < i; j++) {
        e[j] /= h
        f += e[j] * d[j]
      }
      const hh = f / (h + h)
      for (let j = 0; j < i; j++) e[j] -= hh * d[j]
      for (let j = 0; j < i; j++) {
        f = d[j]
        g = e[j]
        for (let k = j; k <= i - 1; k++) V[k][j] -= f * e[k] + g * d[k]
        d[j] = V[i - 1][j]
        V[i][j] = 0
      }
    }
    d[i] = h
  }
  for (let i = 0; i < n - 1; i++) {
    V[n - 1][i] = V[i][i]
    V[i][i] = 1
    const h = d[i + 1]
    if (h !== 0) {
      for (let k = 0; k <= i; k++) d[k] = V[k][i + 1] / h
      for (let j = 0; j <= i; j++) {
        let g = 0
        for (let k = 0; k <= i; k++) g += V[k][i + 1] * V[k][j]
        for (let k = 0; k <= i; k++) V[k][j] -= g * d[k]
      }
    }
    for (let k = 0; k <= i; k++) V[k][i + 1] = 0
  }
  for (let j = 0; j < n; j++) {
    d[j] = V[n - 1][j]
    V[n - 1][j] = 0
  }
  V[n - 1][n - 1] = 1
  e[0] = 0
}

function tql2(V: Float64Array[], d: Float64Array, e: Float64Array) {
  const n = V.length
  for (let i = 1; i < n; i++) e[i - 1] = e[i]
  e[n - 1] = 0
  let f = 0
  let tst1 = 0
  const eps = 2 ** -52
  for (let l = 0; l < n; l++) {
    tst1 = Math.max(tst1, Math.abs(d[l]) + Math.abs(e[l]))
    let m = l
    while (m < n) {
      if (Math.abs(e[m]) <= eps * tst1) break
      m++
    }
    if (m > l) {
      for (let iter = 0; iter < 60; iter++) {
        let g = d[l]
        let p = (d[l + 1] - g) / (2 * e[l])
        let r = Math.hypot(p, 1)
        if (p < 0) r = -r
        d[l] = e[l] / (p + r)
        d[l + 1] = e[l] * (p + r)
        const dl1 = d[l + 1]
        let h = g - d[l]
        for (let i = l + 2; i < n; i++) d[i] -= h
        f += h
        p = d[m]
        let c = 1
        let c2 = c
        let c3 = c
        const el1 = e[l + 1]
        let s = 0
        let s2 = 0
        for (let i = m - 1; i >= l; i--) {
          c3 = c2
          c2 = c
          s2 = s
          g = c * e[i]
          h = c * p
          r = Math.hypot(p, e[i])
          e[i + 1] = s * r
          s = e[i] / r
          c = p / r
          p = c * d[i] - s * g
          d[i + 1] = h + s * (c * g + s * d[i])
          for (let k = 0; k < n; k++) {
            h = V[k][i + 1]
            V[k][i + 1] = s * V[k][i] + c * h
            V[k][i] = c * V[k][i] - s * h
          }
        }
        p = (-s * s2 * c3 * el1 * e[l]) / dl1
        e[l] = s * p
        d[l] = c * p
        if (Math.abs(e[l]) <= eps * tst1) break
      }
    }
    d[l] += f
    e[l] = 0
  }
}
