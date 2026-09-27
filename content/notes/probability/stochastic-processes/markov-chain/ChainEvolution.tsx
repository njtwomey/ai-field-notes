import { useMemo, useState } from 'react'
import {
  GraphDiagram,
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  type GraphEdge,
  type XYSeries,
} from '@/components/viz'

type Matrix = number[][]
type State = '0' | '1' | '2'

const NAMES = ['sunny', 'cloudy', 'rainy']
const STEPS = 30
const TIMES = Array.from({ length: STEPS + 1 }, (_, t) => t)

/** Each row i is set by the probability of staying and the share of the rest that goes to state i + 1 (mod 3). */
type RowSpec = { stay: number; next: number }

const PRESETS: Record<string, RowSpec[]> = {
  weather: [
    { stay: 0.7, next: 2 / 3 },
    { stay: 0.4, next: 0.5 },
    { stay: 0.5, next: 0.4 },
  ],
  cycle: [
    { stay: 0, next: 1 },
    { stay: 0, next: 1 },
    { stay: 0, next: 1 },
  ],
  sticky: [
    { stay: 0.95, next: 0.5 },
    { stay: 0.95, next: 0.5 },
    { stay: 0.95, next: 0.5 },
  ],
}

function toMatrix(rows: RowSpec[]): Matrix {
  return rows.map(({ stay, next }, i) => {
    const row = [0, 0, 0]
    row[i] = stay
    row[(i + 1) % 3] = (1 - stay) * next
    row[(i + 2) % 3] = (1 - stay) * (1 - next)
    return row
  })
}

/** Solve π(P − I) = 0 with Σπ = 1 by Gaussian elimination; null when the solution is not unique. */
function stationary(P: Matrix): number[] | null {
  // Rows: the first two balance equations (columns of Pᵀ − I) and the normalisation.
  const A = [
    [P[0][0] - 1, P[1][0], P[2][0], 0],
    [P[0][1], P[1][1] - 1, P[2][1], 0],
    [1, 1, 1, 1],
  ]
  for (let c = 0; c < 3; c++) {
    let pivot = c
    for (let r = c + 1; r < 3; r++) if (Math.abs(A[r][c]) > Math.abs(A[pivot][c])) pivot = r
    if (Math.abs(A[pivot][c]) < 1e-10) return null
    ;[A[c], A[pivot]] = [A[pivot], A[c]]
    for (let r = 0; r < 3; r++) {
      if (r === c) continue
      const f = A[r][c] / A[c][c]
      for (let k = c; k < 4; k++) A[r][k] -= f * A[c][k]
    }
  }
  return A.map((row, i) => row[3] / row[i])
}

/** Moduli of the two eigenvalues other than 1: roots of x² − (tr P − 1)x + det P = 0. */
function secondEigenvalue(P: Matrix): number {
  const tr = P[0][0] + P[1][1] + P[2][2]
  const det =
    P[0][0] * (P[1][1] * P[2][2] - P[1][2] * P[2][1]) -
    P[0][1] * (P[1][0] * P[2][2] - P[1][2] * P[2][0]) +
    P[0][2] * (P[1][0] * P[2][1] - P[1][1] * P[2][0])
  const b = tr - 1
  const disc = b * b - 4 * det
  if (disc < 0) return Math.sqrt(Math.max(det, 0))
  const r = Math.sqrt(disc)
  return Math.max(Math.abs((b + r) / 2), Math.abs((b - r) / 2))
}

const pct = (v: number) => v.toFixed(2)

/** A three-state chain with editable rows: the distribution of X_t from a chosen start, against the stationary π. */
export function ChainEvolution() {
  const [rows, setRows] = useState<RowSpec[]>(PRESETS.weather)
  const [start, setStart] = useState<State>('2')
  const P = useMemo(() => toMatrix(rows), [rows])
  const pi = useMemo(() => stationary(P), [P])
  const lambda2 = secondEigenvalue(P)

  const path = useMemo(() => {
    let d = [0, 0, 0]
    d[Number(start)] = 1
    const out = [d]
    for (let t = 0; t < STEPS; t++) {
      d = [0, 1, 2].map((j) => d[0] * P[0][j] + d[1] * P[1][j] + d[2] * P[2][j])
      out.push(d)
    }
    return out
  }, [P, start])

  const series = useMemo((): XYSeries[] => {
    const lines: XYSeries[] = [0, 1, 2].map((j) => ({
      name: `P(X_t = ${NAMES[j]})`,
      type: 'line',
      x: TIMES,
      y: path.map((d) => d[j]),
      slot: j,
    }))
    if (pi) {
      pi.forEach((v, j) =>
        lines.push({ name: `π(${NAMES[j]})`, type: 'line', x: [0, STEPS], y: [v, v], slot: j, dashed: true }),
      )
    }
    return lines
  }, [path, pi])

  const edges: GraphEdge[] = []
  P.forEach((row, i) =>
    row.forEach((p, j) => {
      if (i !== j && p > 0.005) edges.push({ source: String(i), target: String(j), label: pct(p), curveness: 0.25 })
    }),
  )
  const nodes = [
    { id: '0', label: `S (${pct(P[0][0])})`, x: 0, y: 0 },
    { id: '1', label: `C (${pct(P[1][1])})`, x: 2, y: 0 },
    { id: '2', label: `R (${pct(P[2][2])})`, x: 1, y: 1.6 },
  ]

  const tv = pi ? 0.5 * path[STEPS].reduce((s, v, j) => s + Math.abs(v - pi[j]), 0) : NaN
  const setRow = (i: number, patch: Partial<RowSpec>) =>
    setRows((r) => r.map((row, k) => (k === i ? { ...row, ...patch } : row)))

  return (
    <Interactive
      title="A three-state chain approaching its stationary distribution"
      caption="States sunny (S), cloudy (C) and rainy (R). Each row of P is set by the probability of staying and the share of the remainder that moves on to the next state (S → C → R → S). Solid lines are P(Xₜ = state) from the chosen start; dashed lines are the stationary π. The distance shrinks like |λ₂|ᵗ. The cycle preset is periodic and never settles; the sticky preset settles slowly."
      controls={
        <>
          <ParamChoice
            label="preset"
            value=""
            onChange={(v: string) => v && setRows(PRESETS[v])}
            options={[
              { value: 'weather', label: 'weather' },
              { value: 'cycle', label: 'cycle' },
              { value: 'sticky', label: 'sticky' },
            ]}
          />
          <ParamChoice
            label="start state"
            value={start}
            onChange={setStart}
            options={NAMES.map((n, i) => ({ value: String(i) as State, label: n }))}
          />
          {rows.map((row, i) => (
            <div key={i} className="flex flex-col gap-3">
              <ParamSlider
                label={`${NAMES[i]}: stay`}
                value={row.stay}
                onChange={(v) => setRow(i, { stay: v })}
                min={0}
                max={1}
                step={0.05}
              />
              <ParamSlider
                label={`${NAMES[i]}: share to ${NAMES[(i + 1) % 3]}`}
                value={row.next}
                onChange={(v) => setRow(i, { next: v })}
                min={0}
                max={1}
                step={0.05}
              />
            </div>
          ))}
        </>
      }
      readout={
        <>
          <Readout label="π" value={pi ? `(${pi.map(pct).join(', ')})` : 'not unique'} />
          <Readout label="|λ₂|" value={formatNumber(lambda2)} />
          <Readout label={`total variation at t = ${STEPS}`} value={Number.isNaN(tv) ? '—' : formatNumber(tv)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-[1fr_2fr]">
        <GraphDiagram nodes={nodes} edges={edges} height={260} ariaLabel="Transition graph of the chain" />
        <XYChart height={260} xLabel="t" yLabel="probability" series={series} yRange={[0, 1]} xRange={[0, STEPS]} />
      </div>
    </Interactive>
  )
}
