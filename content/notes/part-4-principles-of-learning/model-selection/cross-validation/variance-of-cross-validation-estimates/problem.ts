/** The regression problem of both figures of this note, so that they draw the same data sets. */
import { child, normals, type Stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat } from 'aifn-compute/foundation/tensor'
import { dataset } from 'aifn-compute/learning/estimators'

/** Rows of the fresh sample on which the conditional error of a fitted model is measured. */
export const TEST = 5000
/** The stream both figures draw their data sets and fresh sample from. */
export const ROOT = 'naive-cv-intervals'
export const LEARNERS = ['least squares', '10-nearest neighbours'] as const

/** n rows of p standard-normal inputs with y = x₁ + … + x₅ + unit noise. */
export function draw(s: Stream, rows: number, p: number) {
  const x = Float64Array.from(toFlat(normals(child(s, 'x'), [rows, p])))
  const e = toFlat(normals(child(s, 'e'), [rows]))
  const y = Float64Array.from({ length: rows }, (_, i) => {
    let v = e[i]
    for (let j = 0; j < Math.min(5, p); j++) v += x[i * p + j]
    return v
  })
  return dataset(fromData(x, [rows, p]), fromData(y))
}
