/**
 * Dynamic programming over tables (Bellman, 1957, "Dynamic Programming"): a generic table-filling algorithm that a
 * figure can step row by row, and the classical instances built on it: 0/1 and unbounded knapsack, longest common
 * subsequence, edit distance (Wagner and Fischer, 1974, J. ACM 21(1)), and global and local sequence alignment
 * (Needleman and Wunsch, 1970, J. Mol. Biol. 48(3); Smith and Waterman, 1981, J. Mol. Biol. 147(1)), each with its
 * traceback.
 */

import type { Tensor } from 'aifn/tensor'
import type { Algorithm } from 'aifn/trace'
import { intTensor, intVector, matrix, readVector, vector, type VectorInput } from './dense'

/**
 * A dynamic program over a table of rank 1 or 2. `cell(i, j, get)` returns the value of cell (i, j) (j is 0 for a rank-1
 * table), optionally with an integer `choice` recording which option won (for the traceback). `get(i, j)` reads a cell
 * computed earlier in row-major order; reading one not yet computed throws.
 */
export interface DynamicProgram {
  /** `[rows]` or `[rows, columns]`. */
  shape: readonly [number] | readonly [number, number]
  cell(i: number, j: number, get: (i: number, j: number) => number): number | { value: number; choice: number }
}

/** One state of `dp`: the table filled up to (not including) row `row`. */
export interface DynamicProgramState {
  /** The values, with NaN in cells not yet computed; the shape of the program. */
  table: Tensor
  /** The choice recorded for each cell (−1 when none), int32, same shape. */
  choice: Tensor
  /** Rows filled so far. */
  row: number
  done: boolean
}

/** Fill one row of a dynamic program's table in place. */
function fillRow(p: DynamicProgram, values: Float64Array, choices: Int32Array, i: number): void {
  const cols = p.shape.length === 2 ? p.shape[1] : 1
  let limit = i * cols
  const get = (a: number, b: number) => {
    const k = a * cols + b
    if (a < 0 || b < 0 || b >= cols || k >= limit)
      throw new Error(`dp: cell (${a}, ${b}) read before it was computed (computing row ${i})`)
    return values[k]
  }
  for (let j = 0; j < cols; j++) {
    const out = p.cell(i, j, get)
    const k = i * cols + j
    if (typeof out === 'number') values[k] = out
    else {
      values[k] = out.value
      choices[k] = out.choice
    }
    limit = k + 1
  }
}

/**
 * A dynamic program filled row by row, as a traceable algorithm: options are a `DynamicProgram` and each step fills one
 * row (one cell for a rank-1 table), so a figure can show the table growing. `dp` runs it to the end.
 */
export const dynamicProgram: Algorithm<DynamicProgram, DynamicProgramState & { program: DynamicProgram }> = {
  name: 'dynamic-program',
  init: (p) => {
    const size = p.shape.length === 2 ? p.shape[0] * p.shape[1] : p.shape[0]
    return {
      program: p,
      table: tableOf(new Float64Array(size).fill(NaN), p.shape),
      choice: intTensor(new Int32Array(size).fill(-1), p.shape),
      row: 0,
      done: p.shape[0] === 0,
    }
  },
  step: (s) => {
    if (s.done) return s
    const values = Float64Array.from(s.table.data)
    const choices = Int32Array.from(s.choice.data)
    fillRow(s.program, values, choices, s.row)
    const row = s.row + 1
    return {
      ...s,
      table: tableOf(values, s.program.shape),
      choice: intTensor(choices, s.program.shape),
      row,
      done: row >= s.program.shape[0],
    }
  },
  done: (s) => s.done,
}

const tableOf = (values: Float64Array, shape: readonly number[]) =>
  shape.length === 2 ? matrix(values, shape[0], shape[1]) : vector(values)

/**
 * Fill a dynamic program's whole table (a `run` of `dynamicProgram`). Returns the table of values and of choices.
 * @example dp({ shape: [11], cell: (i, _j, get) => (i < 2 ? i : get(i - 1, 0) + get(i - 2, 0)) }) // Fibonacci
 */
export function dp(program: DynamicProgram): DynamicProgramState {
  const values = new Float64Array(program.shape.length === 2 ? program.shape[0] * program.shape[1] : program.shape[0])
  const choices = new Int32Array(values.length).fill(-1)
  for (let i = 0; i < program.shape[0]; i++) fillRow(program, values, choices, i)
  return {
    table: tableOf(values, program.shape),
    choice: intTensor(choices, program.shape),
    row: program.shape[0],
    done: true,
  }
}

const at = (t: Tensor, i: number, j: number) => t.data[t.offset + i * t.strides[0] + j * (t.strides[1] ?? 0)]

// ---------------------------------------------------------------------------------------------------------------------
// Knapsack.

function readItems(values: VectorInput, weights: VectorInput, capacity: number, where: string) {
  const v = readVector(values, `${where}: values`)
  const w = readVector(weights, `${where}: weights`, v.length)
  if (!Number.isInteger(capacity) || capacity < 0) throw new Error(`${where}: capacity must be a non-negative integer`)
  for (const x of w)
    if (!Number.isInteger(x) || x < 0) throw new Error(`${where}: weights must be non-negative integers`)
  return { v, w }
}

/**
 * The 0/1 knapsack table as a dynamic program: cell (i, c) is the best value from the first i items within capacity c,
 * T[i][c] = max(T[i−1][c], T[i−1][c − wᵢ] + vᵢ); the choice is 1 when item i is taken. Shape (n + 1) × (C + 1).
 */
export function knapsackProgram(values: VectorInput, weights: VectorInput, capacity: number): DynamicProgram {
  const { v, w } = readItems(values, weights, capacity, 'knapsack')
  return {
    shape: [v.length + 1, capacity + 1],
    cell: (i, c, get) => {
      if (i === 0) return { value: 0, choice: 0 }
      const skip = get(i - 1, c)
      const wi = w[i - 1]
      if (wi <= c) {
        const take = get(i - 1, c - wi) + v[i - 1]
        if (take > skip) return { value: take, choice: 1 }
      }
      return { value: skip, choice: 0 }
    },
  }
}

/** The result of `knapsack`. */
export interface KnapsackResult {
  /** The best total value. */
  value: number
  /** Total weight of the chosen items. */
  weight: number
  /** 1 for each chosen item, else 0 (0/1 knapsack); how many of each item (unbounded knapsack); int32, length n. */
  take: Tensor
  /** The DP table: (n + 1) × (C + 1) for 0/1, C + 1 for unbounded. */
  table: Tensor
  /** The traceback through the table: the cells visited, shape [k, 2] (rows (i, c)) or [k] (capacities). */
  path: Tensor
}

/**
 * The 0/1 knapsack problem: choose items (each at most once) of integer weights wᵢ and values vᵢ to maximise total
 * value within integer capacity C, by dynamic programming in O(nC), with the table and the traceback.
 */
export function knapsack(values: VectorInput, weights: VectorInput, capacity: number): KnapsackResult {
  const program = knapsackProgram(values, weights, capacity)
  const { w } = readItems(values, weights, capacity, 'knapsack')
  const { table, choice } = dp(program)
  const n = w.length
  const take = new Int32Array(n)
  const path: number[] = []
  let c = capacity
  for (let i = n; i >= 1; i--) {
    path.push(i, c)
    if (at(choice, i, c) === 1) {
      take[i - 1] = 1
      c -= w[i - 1]
    }
  }
  path.push(0, c)
  return {
    value: at(table, n, capacity),
    weight: capacity - c,
    take: intVector(take),
    table,
    path: intTensor(path, [path.length / 2, 2]),
  }
}

/**
 * The unbounded knapsack table as a dynamic program: cell c is the best value within capacity c when items may repeat,
 * T[c] = max(T[c − 1], maxᵢ T[c − wᵢ] + vᵢ); the choice is the item added, or −1 when T[c] = T[c − 1].
 */
export function unboundedKnapsackProgram(values: VectorInput, weights: VectorInput, capacity: number): DynamicProgram {
  const { v, w } = readItems(values, weights, capacity, 'unboundedKnapsack')
  for (let i = 0; i < w.length; i++)
    if (w[i] === 0 && v[i] > 0)
      throw new Error('unboundedKnapsack: an item of weight 0 and positive value is unbounded')
  return {
    shape: [capacity + 1],
    cell: (c, _j, get) => {
      if (c === 0) return { value: 0, choice: -1 }
      let best = get(c - 1, 0)
      let choice = -1
      for (let i = 0; i < v.length; i++) {
        if (w[i] === 0 || w[i] > c) continue
        const option = get(c - w[i], 0) + v[i]
        if (option > best) {
          best = option
          choice = i
        }
      }
      return { value: best, choice }
    },
  }
}

/** The unbounded knapsack problem (each item may be taken any number of times), with the table and the traceback. */
export function unboundedKnapsack(values: VectorInput, weights: VectorInput, capacity: number): KnapsackResult {
  const program = unboundedKnapsackProgram(values, weights, capacity)
  const { w } = readItems(values, weights, capacity, 'unboundedKnapsack')
  const { table, choice } = dp(program)
  const take = new Int32Array(w.length)
  const path: number[] = []
  let c = capacity
  while (c > 0) {
    path.push(c)
    const i = choice.data[c]
    if (i < 0) c -= 1
    else {
      take[i]++
      c -= w[i]
    }
  }
  path.push(c)
  let weight = 0
  take.forEach((k, i) => (weight += k * w[i]))
  return { value: table.data[capacity], weight, take: intVector(take), table, path: intVector(path) }
}

// ---------------------------------------------------------------------------------------------------------------------
// Sequences.

/** A sequence: a string (compared by character) or an array of comparable items (compared with ===). */
export type Sequence<T> = string | readonly T[]

const itemsOf = <T>(s: Sequence<T>): readonly (T | string)[] => (typeof s === 'string' ? Array.from(s) : s)

/** Traceback moves, used as choices: diagonal (match or substitution), up (item of a only), left (item of b only). */
export const DIAGONAL = 0
export const UP = 1
export const LEFT = 2
/** Local alignment only: the alignment starts here (the cell is 0). */
export const STOP = 3

/** Longest common subsequence as a dynamic program; table (|a| + 1) × (|b| + 1). */
export function lcsProgram<T>(a: Sequence<T>, b: Sequence<T>): DynamicProgram {
  const x = itemsOf(a)
  const y = itemsOf(b)
  return {
    shape: [x.length + 1, y.length + 1],
    cell: (i, j, get) => {
      if (i === 0 || j === 0) return { value: 0, choice: i === 0 ? (j === 0 ? STOP : LEFT) : UP }
      if (x[i - 1] === y[j - 1]) return { value: get(i - 1, j - 1) + 1, choice: DIAGONAL }
      const up = get(i - 1, j)
      const left = get(i, j - 1)
      return up >= left ? { value: up, choice: UP } : { value: left, choice: LEFT }
    },
  }
}

/** The result of `lcs`. */
export interface LCSResult<T> {
  length: number
  /** The subsequence: a string when both inputs are strings, else an array. */
  subsequence: T[] | string
  /** Matched positions, shape [length, 2]: rows (index in a, index in b); int32. */
  pairs: Tensor
  table: Tensor
  /** The traceback path through the table from (|a|, |b|) to (0, 0), shape [k, 2]; int32. */
  path: Tensor
}

/** Walk the choice table back from (i, j) until `stop(i, j, move)` or the origin; returns the cells visited. */
function traceback(choice: Tensor, i0: number, j0: number, local: boolean): { cells: number[]; moves: number[] } {
  const cells: number[] = []
  const moves: number[] = []
  let i = i0
  let j = j0
  for (;;) {
    cells.push(i, j)
    const move = at(choice, i, j)
    if ((i === 0 && j === 0) || move === STOP || (local && move < 0)) break
    moves.push(move)
    if (move === DIAGONAL) {
      i--
      j--
    } else if (move === UP) i--
    else j--
  }
  return { cells, moves }
}

const pairsTensor = (cells: number[]) => intTensor(cells, [cells.length / 2, 2])

/** The longest common subsequence of two sequences, by dynamic programming in O(|a||b|), with the traceback. */
export function lcs<T>(a: Sequence<T>, b: Sequence<T>): LCSResult<T> {
  const x = itemsOf(a)
  const { table, choice } = dp(lcsProgram(a, b))
  const { cells, moves } = traceback(choice, x.length, itemsOf(b).length, false)
  const pairs: number[] = []
  let i = x.length
  let j = itemsOf(b).length
  for (const move of moves) {
    if (move === DIAGONAL) {
      pairs.unshift(i - 1, j - 1)
      i--
      j--
    } else if (move === UP) i--
    else j--
  }
  const items = pairs.filter((_, k) => k % 2 === 0).map((p) => x[p])
  return {
    length: at(table, x.length, itemsOf(b).length),
    subsequence: typeof a === 'string' && typeof b === 'string' ? items.join('') : (items as T[]),
    pairs: pairsTensor(pairs),
    table,
    path: pairsTensor(cells),
  }
}

/** Costs of the edit operations. */
export interface EditCosts {
  insert?: number
  delete?: number
  substitute?: number
}

/**
 * Edit distance as a dynamic program (Wagner and Fischer, 1974): cell (i, j) is the cost of turning a[0:i] into
 * b[0:j], D[i][j] = min(D[i−1][j−1] + [aᵢ ≠ bⱼ]·substitute, D[i−1][j] + delete, D[i][j−1] + insert).
 */
export function editDistanceProgram<T>(a: Sequence<T>, b: Sequence<T>, costs: EditCosts = {}): DynamicProgram {
  const x = itemsOf(a)
  const y = itemsOf(b)
  const ins = costs.insert ?? 1
  const del = costs.delete ?? 1
  const sub = costs.substitute ?? 1
  return {
    shape: [x.length + 1, y.length + 1],
    cell: (i, j, get) => {
      if (i === 0 && j === 0) return { value: 0, choice: STOP }
      if (i === 0) return { value: get(0, j - 1) + ins, choice: LEFT }
      if (j === 0) return { value: get(i - 1, 0) + del, choice: UP }
      const diag = get(i - 1, j - 1) + (x[i - 1] === y[j - 1] ? 0 : sub)
      const up = get(i - 1, j) + del
      const left = get(i, j - 1) + ins
      if (diag <= up && diag <= left) return { value: diag, choice: DIAGONAL }
      return up <= left ? { value: up, choice: UP } : { value: left, choice: LEFT }
    },
  }
}

/** One edit operation, with positions in a (i) and b (j). */
export type EditOperation =
  { op: 'match' | 'substitute'; i: number; j: number } | { op: 'delete'; i: number } | { op: 'insert'; j: number }

/** The result of `editDistance`. */
export interface EditDistanceResult {
  distance: number
  /** The operations turning a into b, in order. */
  operations: EditOperation[]
  table: Tensor
  /** The traceback path from (|a|, |b|) to (0, 0), shape [k, 2]; int32. */
  path: Tensor
}

/** The edit (Levenshtein) distance between two sequences with its operations; costs default to 1. */
export function editDistance<T>(a: Sequence<T>, b: Sequence<T>, costs: EditCosts = {}): EditDistanceResult {
  const x = itemsOf(a)
  const y = itemsOf(b)
  const { table, choice } = dp(editDistanceProgram(a, b, costs))
  const { cells, moves } = traceback(choice, x.length, y.length, false)
  const operations: EditOperation[] = []
  let i = x.length
  let j = y.length
  for (const move of moves) {
    if (move === DIAGONAL) {
      operations.unshift({ op: x[i - 1] === y[j - 1] ? 'match' : 'substitute', i: i - 1, j: j - 1 })
      i--
      j--
    } else if (move === UP) operations.unshift({ op: 'delete', i: --i })
    else operations.unshift({ op: 'insert', j: --j })
  }
  return { distance: at(table, x.length, y.length), operations, table, path: pairsTensor(cells) }
}

/** Scores for sequence alignment with a linear gap penalty (a gap of length k scores k · gap). */
export interface AlignmentScoring<T> {
  /** Score of aligning two equal items (default 1). */
  match?: number
  /** Score of aligning two different items (default −1). */
  mismatch?: number
  /** Score of each gap position (default −1). */
  gap?: number
  /** A substitution score s(x, y) used instead of match and mismatch (e.g. from BLOSUM62). */
  score?: (x: T | string, y: T | string) => number
}

/**
 * Sequence alignment as a dynamic program. `global` is Needleman–Wunsch: F[i][j] = max(F[i−1][j−1] + s(aᵢ, bⱼ),
 * F[i−1][j] + gap, F[i][j−1] + gap) with gap-filled borders. `local` is Smith–Waterman: the same with 0 as a fourth
 * option (the alignment may start anywhere) and zero borders.
 */
export function alignmentProgram<T>(
  a: Sequence<T>,
  b: Sequence<T>,
  scoring: AlignmentScoring<T> = {},
  mode: 'global' | 'local' = 'global',
): DynamicProgram {
  const x = itemsOf(a)
  const y = itemsOf(b)
  const gap = scoring.gap ?? -1
  const s = scoring.score ?? ((p, q) => (p === q ? (scoring.match ?? 1) : (scoring.mismatch ?? -1)))
  const local = mode === 'local'
  return {
    shape: [x.length + 1, y.length + 1],
    cell: (i, j, get) => {
      if (i === 0 && j === 0) return { value: 0, choice: STOP }
      if (i === 0) return local ? { value: 0, choice: STOP } : { value: get(0, j - 1) + gap, choice: LEFT }
      if (j === 0) return local ? { value: 0, choice: STOP } : { value: get(i - 1, 0) + gap, choice: UP }
      const diag = get(i - 1, j - 1) + s(x[i - 1], y[j - 1])
      const up = get(i - 1, j) + gap
      const left = get(i, j - 1) + gap
      let value = diag
      let choice = DIAGONAL
      if (up > value) [value, choice] = [up, UP]
      if (left > value) [value, choice] = [left, LEFT]
      if (local && value <= 0) return { value: 0, choice: STOP }
      return { value, choice }
    },
  }
}

/** The result of `needlemanWunsch` and `smithWaterman`. */
export interface AlignmentResult<T> {
  score: number
  /** The aligned sequences with gaps: strings with `-` when the inputs are strings, else arrays with null. */
  alignedA: string | (T | null)[]
  alignedB: string | (T | null)[]
  /** The aligned region: a[start[0]:end[0]] against b[start[1]:end[1]] (the whole sequences for global alignment). */
  start: readonly [number, number]
  end: readonly [number, number]
  /** Score table, (|a| + 1) × (|b| + 1). */
  table: Tensor
  /** Traceback choices (DIAGONAL, UP, LEFT, STOP), int32, same shape. */
  choice: Tensor
  /** Traceback path from the end cell back to the start cell, shape [k, 2]; int32. */
  path: Tensor
}

function align<T>(a: Sequence<T>, b: Sequence<T>, scoring: AlignmentScoring<T>, mode: 'global' | 'local') {
  const x = itemsOf(a)
  const y = itemsOf(b)
  const { table, choice } = dp(alignmentProgram(a, b, scoring, mode))
  let ei = x.length
  let ej = y.length
  if (mode === 'local') {
    // The local alignment ends at the highest-scoring cell (the first one in row-major order on ties).
    let best = -Infinity
    for (let i = 0; i <= x.length; i++)
      for (let j = 0; j <= y.length; j++)
        if (at(table, i, j) > best) {
          best = at(table, i, j)
          ei = i
          ej = j
        }
  }
  const { cells, moves } = traceback(choice, ei, ej, mode === 'local')
  const ga: (T | string | null)[] = []
  const gb: (T | string | null)[] = []
  let i = ei
  let j = ej
  for (const move of moves) {
    if (move === DIAGONAL) {
      ga.unshift(x[--i])
      gb.unshift(y[--j])
    } else if (move === UP) {
      ga.unshift(x[--i])
      gb.unshift(null)
    } else {
      ga.unshift(null)
      gb.unshift(y[--j])
    }
  }
  const strings = typeof a === 'string' && typeof b === 'string'
  const show = (g: (T | string | null)[]) => (strings ? g.map((v) => v ?? '-').join('') : (g as (T | null)[]))
  return {
    score: at(table, ei, ej),
    alignedA: show(ga),
    alignedB: show(gb),
    start: [i, j] as const,
    end: [ei, ej] as const,
    table,
    choice,
    path: pairsTensor(cells),
  }
}

/** Global alignment of two sequences (Needleman and Wunsch, 1970) with a linear gap penalty, with the traceback. */
export function needlemanWunsch<T>(
  a: Sequence<T>,
  b: Sequence<T>,
  scoring: AlignmentScoring<T> = {},
): AlignmentResult<T> {
  return align(a, b, scoring, 'global')
}

/** Local alignment of two sequences (Smith and Waterman, 1981) with a linear gap penalty, with the traceback. */
export function smithWaterman<T>(
  a: Sequence<T>,
  b: Sequence<T>,
  scoring: AlignmentScoring<T> = {},
): AlignmentResult<T> {
  return align(a, b, scoring, 'local')
}
