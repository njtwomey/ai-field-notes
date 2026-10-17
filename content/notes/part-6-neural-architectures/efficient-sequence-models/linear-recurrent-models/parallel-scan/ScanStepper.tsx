import { useMemo, useState } from 'react'
import {
  choice,
  Diagram,
  type DiagramEdge,
  type DiagramNode,
  type DiagramSpec,
  Figure,
  formatNumber,
  Player,
  Readout,
  useFigureState,
} from 'aifn-render'
import {
  type BlellochPhase,
  blellochScanSteps,
  hillisSteeleScanSteps,
  type ScanMove,
  shapeOfValue,
  type Tensor,
  tensor,
  toFlat,
  toRows,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { linearRecurrence } from 'aifn-compute/nn/sequence'

type Example = 'worked' | 'decay' | 'sums'
type Algo = 'blelloch' | 'hillis-steele'

/** The coefficients (a_t, b_t) of h_t = a_t h_{t−1} + b_t for each example. */
const EXAMPLES: Record<Example, { a: number[]; b: number[] }> = {
  worked: { a: [0.5, 2, 0.5, 1], b: [1, 1, 2, -1] },
  decay: { a: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], b: [4, 0, 0, 2, 0, 0, 0, 8] },
  sums: { a: [1, 1, 1, 1, 1, 1, 1, 1], b: [3, 1, 4, 1, 5, 9, 2, 6] },
}

/**
 * "e_s then e_t" for the affine maps h ↦ a h + b, row by row: (a_s, b_s) • (a_t, b_t) = (a_t a_s, a_t b_s + b_t).
 * The scan steppers pass whole blocks of rows; each row is one pair.
 */
function combine(p: Value, q: Value): Value {
  const x = toFlat(p as Tensor)
  const y = toFlat(q as Tensor)
  const out: number[] = []
  for (let i = 0; i < x.length; i += 2) out.push(y[i] * x[i], y[i] * x[i + 1] + y[i + 1])
  return tensor(out, shapeOfValue(p))
}

type Row = { values: number[][]; label: string; active: Set<number>; edges: [number, number][] }

/** The fields both scan steppers' states share (`offset` is Hillis–Steele's, `phase` and `inclusive` Blelloch's). */
type RoundState = {
  values: Tensor
  moves: readonly ScanMove[]
  work: number
  phase?: BlellochPhase
  offset?: number
  inclusive?: Tensor | null
}

const fmt = (v: number) => formatNumber(Math.round(v * 1000) / 1000)
const pair = ([a, b]: number[]) => `(${fmt(a)}, ${fmt(b)})`

/** Every round of the chosen scan, as rows of pairs with the combinations that produced them. */
function rounds(algo: Algo, a: number[], b: number[]): { rows: Row[]; work: number[]; inclusive: number[][] } {
  const x = tensor(
    a.flatMap((ai, i) => [ai, b[i]]),
    [a.length, 2],
  )
  const steps: readonly RoundState[] =
    algo === 'blelloch'
      ? trace(blellochScanSteps(combine, x, tensor([1, 0])), undefined, 64).steps
      : trace(hillisSteeleScanSteps(combine, x), undefined, 64).steps
  const rows: Row[] = steps.map((s, r): Row => {
    const values = toRows(s.values)
    if (r === 0) return { values, label: 'input $e_t$', active: new Set<number>(), edges: [] }
    const prev = steps[r - 1]
    const active = new Set<number>()
    const edges: [number, number][] = []
    let label: string
    if (algo === 'hillis-steele') {
      label = `round ${r}: offset ${prev.offset}`
      for (const { from, to } of s.moves) {
        active.add(to)
        edges.push([from, to], [to, to])
      }
    } else if (prev.phase === 'up-sweep') {
      label = `up-sweep ${r}`
      for (const { from, to } of s.moves) {
        active.add(to)
        edges.push([from, to], [to, to])
      }
    } else if (prev.phase === 'clear') {
      label = 'root ← identity'
      active.add(values.length - 1)
    } else {
      label = 'down-sweep'
      // Swap-and-combine: the left child takes the parent's prefix; the parent's slot takes prefix • left total.
      for (const { from, to } of s.moves) {
        active.add(to)
        active.add(from)
        edges.push([from, to], [from, from], [to, from])
      }
    }
    return { values, label, active, edges }
  })
  const last = steps[steps.length - 1]
  const inclusive = algo === 'blelloch' && last.inclusive ? toRows(last.inclusive) : rows[rows.length - 1].values
  return { rows, work: steps.map((s) => s.work), inclusive }
}

const DX = 2.7
const DY = 1.25

function diagram(rows: Row[], shown: number, n: number): DiagramSpec {
  const nodes: DiagramNode[] = []
  const edges: DiagramEdge[] = []
  const P = rows[0].values.length
  rows.slice(0, shown + 1).forEach((row, r) => {
    nodes.push({ id: `l${r}`, x: -1.9, y: r * DY, shape: 'text', w: 2.6, small: true, label: row.label })
    row.values.forEach((v, i) => {
      const padded = i >= n
      nodes.push({
        id: `n${r}_${i}`,
        x: i * DX,
        y: r * DY,
        w: 2.3,
        h: 0.7,
        small: true,
        label: pair(v),
        tone: row.active.has(i) ? 0 : 'neutral',
        dashed: padded,
        state: r === shown ? (row.active.has(i) ? 'active' : 'done') : 'done',
      })
    })
    for (const [from, to] of row.edges)
      edges.push({
        from: `n${r - 1}_${from}:s`,
        to: `n${r}_${to}:n`,
        route: 'straight',
        arrow: 'end',
        state: r === shown ? 'active' : 'done',
      })
  })
  return {
    nodes,
    edges,
    unit: 40,
    frame: { x0: -3.4, y0: -0.6, x1: (P - 1) * DX + 1.3, y1: (rows.length - 1) * DY + 0.6 },
  }
}

/** Blelloch's and Hillis–Steele's scans of the pairs (a_t, b_t) of a linear recurrence, one round at a time. */
export function ScanStepper() {
  const state = useFigureState({
    algorithm: choice<Algo>(
      [
        { value: 'blelloch', label: 'Blelloch (work-efficient)' },
        { value: 'hillis-steele', label: 'Hillis–Steele' },
      ],
      'blelloch',
      { label: 'algorithm' },
    ),
    example: choice<Example>(
      [
        { value: 'worked', label: 'worked example, n = 4' },
        { value: 'decay', label: 'decay 0.5, n = 8' },
        { value: 'sums', label: 'running sum (a = 1), n = 8' },
      ],
      'worked',
      { label: 'sequence' },
    ),
  })
  const { a, b } = EXAMPLES[state.example]
  const { rows, work, inclusive } = useMemo(() => rounds(state.algorithm, a, b), [state.algorithm, a, b])
  const [step, setStep] = useState(0)
  const shown = Math.min(step, rows.length - 1)
  const spec = useMemo(() => diagram(rows, shown, a.length), [rows, shown, a.length])

  const reference = useMemo(() => toFlat(linearRecurrence(tensor(a), tensor(b)) as Tensor), [a, b])
  const done = shown === rows.length - 1
  const h = inclusive.slice(0, a.length).map((p) => p[1])
  const gap = Math.max(...h.map((v, i) => Math.abs(v - reference[i])))

  return (
    <Figure
      title="A parallel scan, one round at a time"
      state={state}
      caption={
        state.algorithm === 'blelloch'
          ? 'Each box holds a pair (A, B), the affine map h ↦ A h + B. Rows are rounds, top to bottom; arrows show which pairs each highlighted box combined, earlier pair first. The up-sweep builds partial compositions up a binary tree until the last box holds the whole sequence. The root is reset to the identity (1, 0), and the down-sweep passes prefixes back down: each left child receives its parent’s prefix and each right child that prefix combined with the left child’s total. The last row is the exclusive scan; combining each box with its own input gives the states h_t, compared with the sequential loop below.'
          : 'Each box holds a pair (A, B), the affine map h ↦ A h + B. Rows are rounds, top to bottom; in round r every position t combines with the position 2^(r−1) to its left, all at once (arrows). After ⌈log₂ n⌉ rounds every box holds the composition of all steps up to it, and its second entry is h_t. Count the combinations: more than the n − 1 of the sequential loop.'
      }
      controls={
        <Player
          value={shown}
          onChange={setStep}
          count={rows.length}
          label="round"
          format={(k) => `${k} of ${rows.length - 1}`}
        />
      }
      readouts={
        <>
          <Readout label="round" value={`${shown} of ${rows.length - 1}`} />
          <Readout label="combinations so far" value={String(work[shown])} />
          <Readout label="sequential loop" value={`${a.length - 1} combinations, ${a.length - 1} rounds`} />
          <Readout label="hₜ by the scan" value={done ? h.map(fmt).join(', ') : '·'} />
          <Readout label="hₜ by linearRecurrence" value={reference.map(fmt).join(', ')} />
          <Readout label="largest difference" value={done ? formatNumber(gap) : '·'} />
        </>
      }
    >
      <Diagram
        spec={spec}
        ariaLabel="Rows of boxes, one row per round of the scan, each box a pair (A, B); arrows show which pairs were combined in the current round"
      />
    </Figure>
  )
}
