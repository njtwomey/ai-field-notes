import { useMemo } from 'react'
import { split } from 'aifn-methods/data/synthetic'
import { type Dataset } from 'aifn-methods/data'
import { stream } from 'aifn/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn/foundation/tensor'

/**
 * The input of the multi-feature data views: a dataset (`aifn-methods/data`), or a plain matrix with optional labels.
 * Integer labels colour the rows by class; without them every row is one class.
 */
export type ClassTableInput = {
  /** A dataset: `x` [n, d], integer labels `y`, feature and label names from `meta`. */
  data?: Dataset
  /** In place of `data`: a matrix [n, d] (a tensor or rows). */
  x?: Tensor | readonly (readonly number[])[]
  /** Class labels 0…K−1, one per row. */
  y?: Tensor | ArrayLike<number>
  featureNames?: readonly string[]
  labelNames?: readonly string[]
}

/** Rows as columns, with labels and names, after any subsampling for drawing. */
export type ClassTable = {
  /** Rows drawn. */
  n: number
  /** Rows in the input. */
  total: number
  d: number
  /** One array per feature, length n. */
  columns: Float64Array[]
  /** Class per drawn row (all 0 without labels). */
  labels: Int32Array
  /** Classes (1 without labels). */
  k: number
  featureNames: string[]
  labelNames: string[]
  /** Whether rows were labelled by class. */
  labelled: boolean
}

function toDataset({ data, x, y, featureNames, labelNames }: ClassTableInput): Dataset {
  if (data) return data
  if (!x) throw new Error('a data view needs `data` or `x`')
  const t = 'shape' in x ? x : fromData(Float64Array.from(x.flat()), [x.length, x[0]?.length ?? 0])
  const d = t.shape[1]
  return {
    kind: 'dataset',
    x: t,
    y: y === undefined ? undefined : fromData(Int32Array.from('shape' in y ? toFlat(y) : Array.from(y))),
    meta: {
      name: 'data',
      description: '',
      task: 'classification',
      featureNames: featureNames ? [...featureNames] : Array.from({ length: d }, (_, j) => `x${j + 1}`),
      labelNames: labelNames ? [...labelNames] : undefined,
    },
  }
}

/**
 * The input as columns and labels. Above `maxRows` rows, a class-stratified sample of `maxRows` (aifn/datasets `split`,
 * seeded) is drawn instead, so every class keeps its share and a rare class is not lost in the sample.
 */
export function useClassTable(input: ClassTableInput, maxRows: number): ClassTable {
  const { data, x, y, featureNames, labelNames } = input
  return useMemo(() => {
    const full = toDataset({ data, x, y, featureNames, labelNames })
    const total = full.x.shape[0]
    const drawn = total > maxRows ? split(stream('lab/data-view'), full, { test: maxRows / total }).test : full
    const [n, d] = drawn.x.shape
    const flat = toFlat(drawn.x)
    const columns = Array.from({ length: d }, (_, j) => Float64Array.from({ length: n }, (_, i) => flat[i * d + j]))
    const labelled = drawn.y !== undefined && drawn.y.dtype === 'int32'
    const labels = labelled ? Int32Array.from(toFlat(drawn.y!)) : new Int32Array(n)
    let top = -1
    for (const c of labels) top = Math.max(top, c)
    const K = labelled ? Math.max(drawn.meta.labelNames?.length ?? 0, top + 1, 1) : 1
    const names = labelled
      ? Array.from({ length: K }, (_, c) => drawn.meta.labelNames?.[c] ?? `class ${c}`)
      : [drawn.meta.name]
    return {
      n,
      total,
      d,
      columns,
      labels,
      k: names.length,
      featureNames: drawn.meta.featureNames.slice(0, d),
      labelNames: names,
      labelled,
    }
  }, [data, x, y, featureNames, labelNames, maxRows])
}

/** Toggle membership of `c` in a set, as a new set. */
export const toggled = (set: ReadonlySet<number>, c: number): Set<number> => {
  const next = new Set(set)
  if (!next.delete(c)) next.add(c)
  return next
}
