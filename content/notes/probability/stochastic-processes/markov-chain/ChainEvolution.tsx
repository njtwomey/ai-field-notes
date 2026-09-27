import { useMemo, useState } from 'react'
import { MathText } from '@/components/content/MathText'
import { Diagram } from '@/components/diagram/Diagram'
import type { DiagramEdge, DiagramSpec, Side } from '@/components/diagram/types'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from '@/components/viz'

type Matrix = number[][]
type State = '0' | '1' | '2'

const NAMES = ['sunny', 'cloudy', 'rainy']
const SHORT = ['S', 'C', 'R']
/** Where each state sits, and the side its self-loop is drawn on. */
const PLACES: [number, number, Side][] = [
  [0, 0, 'w'],
  [4.4, 0, 'e'],
  [2.2, 3.4, 's'],
]
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

  const time = useParam(3, { min: 0, max: STEPS, step: 1 })
  const t = time.value
  // Each state is shaded by P(X_t = state), in the colour of its line; arrows carry the transition probabilities.
  const diagram = useMemo((): DiagramSpec => {
    const edges: DiagramEdge[] = []
    P.forEach((row, i) =>
      row.forEach((p, j) => {
        if (p <= 0.005) return
        const from = i === j ? `${i}:${PLACES[i][2]}` : String(i)
        edges.push({
          from,
          to: String(j),
          route: 'curve',
          bend: 0.55,
          // Each arrow of a pair bows to its own side; its label goes on the outside of the bow.
          labelSide: 'right',
          label: `$${pct(p)}$`,
          tone: 'neutral',
        })
      }),
    )
    return {
      unit: 44,
      nodes: PLACES.map(([x, y], i) => ({
        id: String(i),
        x,
        y,
        shape: 'circle',
        w: 1.2,
        h: 1.2,
        tone: i,
        shade: path[t][i],
        label: `${SHORT[i]}\n$${pct(path[t][i])}$`,
      })),
      edges,
    }
  }, [P, path, t])

  const tv = pi ? 0.5 * path[STEPS].reduce((s, v, j) => s + Math.abs(v - pi[j]), 0) : NaN
  const setRow = (i: number, patch: Partial<RowSpec>) =>
    setRows((r) => r.map((row, k) => (k === i ? { ...row, ...patch } : row)))

  return (
    <Interactive
      title="A three-state chain approaching its stationary distribution"
      caption={
        <MathText text="States sunny (S), cloudy (C) and rainy (R). Each row of $P$ is set by the probability of staying and the share of the remainder that moves on to the next state (S → C → R → S). The graph shows the chain at time $t$: arrows carry the transition probabilities, self-loops included, and each state is shaded by $P(X_t = \text{state})$. Solid lines are $P(X_t = \text{state})$ from the chosen start; dashed lines are the stationary $\pi$. The distance shrinks like $\abs{\lambda_2}^t$. Step $t$ with the arrows or drag it on the chart. The cycle preset is periodic and never settles; the sticky preset settles slowly." />
      }
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
          <ParamSlider label="time t" param={time} format={(v) => String(v)} withArrows />
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
      <div className="grid items-center gap-4 md:grid-cols-2">
        <Diagram
          spec={diagram}
          ariaLabel="Transition graph of the chain, each state shaded by its probability at time t"
        />
        <XYChart
          height={260}
          xLabel="t"
          yLabel="probability"
          series={series}
          yRange={[0, 1]}
          xRange={[0, STEPS]}
          handles={[{ kind: 'x', at: t, label: 't', onDrag: (x) => time.set(x) }]}
        />
      </div>
    </Interactive>
  )
}
