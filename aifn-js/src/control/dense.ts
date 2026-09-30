/**
 * Private helpers: a small row-major matrix record `{ d, r, c }` over Float64Array with the handful of operations the
 * control algorithms need, and the conversions to and from the public tensor surface. Linear solves go through
 * `aifn/linalg` (LU with partial pivoting), which reports a singular system instead of dividing by zero.
 */

import { lu, luSolve, svd } from 'aifn/linalg'
import { fromData, isTensor, toFlat, type Matrix, type Tensor } from 'aifn/tensor'
import type { F64, MatrixLike, VectorLike } from './vector'

/** A row-major r×c matrix. */
export type M = { d: F64; r: number; c: number }

export const zerosM = (r: number, c: number): M => ({ d: new Float64Array(r * c), r, c })

export function eyeM(n: number): M {
  const m = zerosM(n, n)
  for (let i = 0; i < n; i++) m.d[i * n + i] = 1
  return m
}

/** A matrix argument as M; a vector (or a plain number array) becomes a column. */
export function asM(a: MatrixLike | VectorLike | number, where: string): M {
  if (typeof a === 'number') return { d: Float64Array.of(a), r: 1, c: 1 }
  if (isTensor(a)) {
    if (a.shape.length === 0) return { d: Float64Array.from(toFlat(a)), r: 1, c: 1 }
    if (a.shape.length === 1) return { d: Float64Array.from(toFlat(a)), r: a.shape[0], c: 1 }
    if (a.shape.length !== 2) throw new Error(`${where}: expected a matrix, got shape [${a.shape.join(', ')}]`)
    return { d: Float64Array.from(toFlat(a)), r: a.shape[0], c: a.shape[1] }
  }
  const list = a as readonly (ArrayLike<number> | number)[]
  if (list.length === 0) return zerosM(0, 0)
  if (typeof list[0] === 'number') {
    const v = Float64Array.from(list as ArrayLike<number>)
    return { d: v, r: v.length, c: 1 }
  }
  const rows = list as readonly ArrayLike<number>[]
  const c = rows[0].length
  const d = new Float64Array(rows.length * c)
  rows.forEach((row, i) => {
    if (row.length !== c) throw new Error(`${where}: ragged matrix rows`)
    for (let j = 0; j < c; j++) d[i * c + j] = row[j]
  })
  return { d, r: rows.length, c }
}

export const toT = (m: M): Matrix => fromData(Float64Array.from(m.d), [m.r, m.c])
export const toV = (v: ArrayLike<number>): Tensor => fromData(Float64Array.from(v), [v.length])

export function mul(a: M, b: M): M {
  if (a.c !== b.r) throw new Error(`control: cannot multiply ${a.r}×${a.c} by ${b.r}×${b.c}`)
  const out = zerosM(a.r, b.c)
  for (let i = 0; i < a.r; i++)
    for (let k = 0; k < a.c; k++) {
      const v = a.d[i * a.c + k]
      if (v === 0) continue
      for (let j = 0; j < b.c; j++) out.d[i * b.c + j] += v * b.d[k * b.c + j]
    }
  return out
}

export function add(a: M, b: M, beta = 1): M {
  const out = zerosM(a.r, a.c)
  for (let i = 0; i < a.d.length; i++) out.d[i] = a.d[i] + beta * b.d[i]
  return out
}

export const sub = (a: M, b: M): M => add(a, b, -1)

export function scaleM(a: M, s: number): M {
  return { d: a.d.map((v) => v * s), r: a.r, c: a.c }
}

export function tr(a: M): M {
  const out = zerosM(a.c, a.r)
  for (let i = 0; i < a.r; i++) for (let j = 0; j < a.c; j++) out.d[j * a.r + i] = a.d[i * a.c + j]
  return out
}

/** (A + Aᵀ)/2. */
export function symmetrise(a: M): M {
  const out = zerosM(a.r, a.c)
  for (let i = 0; i < a.r; i++)
    for (let j = 0; j < a.c; j++) out.d[i * a.c + j] = 0.5 * (a.d[i * a.c + j] + a.d[j * a.c + i])
  return out
}

/** Solve A X = B by LU with partial pivoting; null when A is singular to working precision. */
export function solveM(a: M, b: M): M | null {
  if (!a.d.every(Number.isFinite) || !b.d.every(Number.isFinite)) return null
  const f = lu(fromData(Float64Array.from(a.d), [a.r, a.c]))
  if (f.singular) return null
  const x = luSolve(f, fromData(Float64Array.from(b.d), [b.r, b.c]))
  return { d: Float64Array.from(toFlat(x)), r: b.r, c: b.c }
}

export function invM(a: M): M | null {
  return solveM(a, eyeM(a.r))
}

/** max |a_ij| */
export function maxAbsM(a: M): number {
  let m = 0
  for (const v of a.d) m = Math.max(m, Math.abs(v))
  return m
}

/** Numerical rank: singular values above max(r, c)·ε·σ_max (numpy's `matrix_rank` default), plus the values. */
export function rankM(a: M, tol?: number): { rank: number; singularValues: number[] } {
  if (a.r === 0 || a.c === 0) return { rank: 0, singularValues: [] }
  const s = toFlat(svd(fromData(Float64Array.from(a.d), [a.r, a.c])).S)
  const cut = tol ?? Math.max(a.r, a.c) * 2 ** -52 * (s[0] ?? 0)
  return { rank: s.filter((v) => v > cut).length, singularValues: s }
}

/** Horizontal concatenation [A B …]. */
export function hcat(parts: M[]): M {
  const r = parts[0].r
  const c = parts.reduce((s, p) => s + p.c, 0)
  const out = zerosM(r, c)
  let off = 0
  for (const p of parts) {
    for (let i = 0; i < r; i++) for (let j = 0; j < p.c; j++) out.d[i * c + off + j] = p.d[i * p.c + j]
    off += p.c
  }
  return out
}

/** Vertical concatenation [A; B; …]. */
export function vcat(parts: M[]): M {
  const c = parts[0].c
  const r = parts.reduce((s, p) => s + p.r, 0)
  const out = zerosM(r, c)
  let off = 0
  for (const p of parts) {
    out.d.set(p.d, off * c)
    off += p.r
  }
  return out
}

/** The block of rows [r0, r1) and columns [c0, c1). */
export function block(a: M, r0: number, r1: number, c0: number, c1: number): M {
  const out = zerosM(r1 - r0, c1 - c0)
  for (let i = r0; i < r1; i++) for (let j = c0; j < c1; j++) out.d[(i - r0) * (c1 - c0) + (j - c0)] = a.d[i * a.c + j]
  return out
}

/** The Kronecker product A ⊗ B. */
export function kronM(a: M, b: M): M {
  const out = zerosM(a.r * b.r, a.c * b.c)
  const C = a.c * b.c
  for (let i = 0; i < a.r; i++)
    for (let j = 0; j < a.c; j++) {
      const v = a.d[i * a.c + j]
      if (v === 0) continue
      for (let k = 0; k < b.r; k++)
        for (let l = 0; l < b.c; l++) out.d[(i * b.r + k) * C + j * b.c + l] = v * b.d[k * b.c + l]
    }
  return out
}

export const allFiniteM = (a: M): boolean => a.d.every(Number.isFinite)
