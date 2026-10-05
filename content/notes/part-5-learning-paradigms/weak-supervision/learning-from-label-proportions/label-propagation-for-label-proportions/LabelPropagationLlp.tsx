import { useMemo } from 'react'
import { choice, Figure, float, formatNumber, int, Plot, Points, Readout, useAxis, useFigureState } from 'aifn-render'
import { normal, stream, uniform } from 'aifn-compute/foundation/random'

const RANGE: [number, number] = [-4, 14]
const Y_RANGE: [number | undefined, number | undefined] = [-4, 14]
const ROUNDS = 10
const SWEEPS = 100

/** The paper's two bag-proportion configurations for three bags. */
const CONFIGS = { A: [0.6, 0.4, 0.5], B: [0.85, 0.25, 0.4] } as const
type Config = keyof typeof CONFIGS

/** XOR data: class 1 at corners (0, 0) and (10, 10), class 0 at (0, 10) and (10, 0), unit covariance. */
function xorBags(n: number, config: Config, seed: number) {
  const g = stream(seed)
  const per = Math.floor(n / 3)
  const x1: number[] = []
  const x2: number[] = []
  const y: number[] = []
  const bag: number[] = []
  const counts: number[] = []
  CONFIGS[config].forEach((p, k) => {
    const positives = Math.round(p * per)
    counts.push(positives)
    for (let i = 0; i < per; i++) {
      const label = i < positives ? 1 : 0
      const corner = uniform(g) < 0.5
      const [cx, cy] = label === 1 ? (corner ? [0, 0] : [10, 10]) : corner ? [0, 10] : [10, 0]
      x1.push(cx + normal(g))
      x2.push(cy + normal(g))
      y.push(label)
      bag.push(k)
    }
  })
  return { x1, x2, y, bag, counts }
}

/** In-place LU factorisation with partial pivoting; returns a solver for A z = b. */
function luSolver(A: number[][]) {
  const n = A.length
  const perm = Array.from({ length: n }, (_, i) => i)
  for (let k = 0; k < n; k++) {
    let p = k
    for (let i = k + 1; i < n; i++) if (Math.abs(A[i][k]) > Math.abs(A[p][k])) p = i
    ;[A[k], A[p]] = [A[p], A[k]]
    ;[perm[k], perm[p]] = [perm[p], perm[k]]
    for (let i = k + 1; i < n; i++) {
      const f = A[i][k] / A[k][k]
      A[i][k] = f
      for (let j = k + 1; j < n; j++) A[i][j] -= f * A[k][j]
    }
  }
  return (b: number[]) => {
    const z = perm.map((i) => b[i])
    for (let i = 0; i < n; i++) for (let j = 0; j < i; j++) z[i] -= A[i][j] * z[j]
    for (let i = n - 1; i >= 0; i--) {
      for (let j = i + 1; j < n; j++) z[i] -= A[i][j] * z[j]
      z[i] /= A[i][i]
    }
    return z
  }
}

const accuracy = (f: number[], y: number[]) => f.filter((v, i) => (v > 0.5 ? 1 : 0) === y[i]).length / y.length

/**
 * LP-LLP of Poyiadzi, Santos-Rodriguez and Twomey (2018): propagate soft labels (each point starts at its bag's
 * proportion) through (1 − α)(I − αS)⁻¹, then project onto the bag-mass constraints and the box [0, 1] by alternating
 * projections, and repeat.
 */
export function LabelPropagationLlp() {
  const state = useFigureState({
    config: choice<Config>(
      [
        { value: 'A', label: 'A' },
        { value: 'B', label: 'B' },
      ],
      'B',
      { label: 'bag proportions' },
    ),
    size: int(120, { min: 60, max: 300, step: 60, label: 'points n', format: (v) => String(v) }),
    alpha: int(0.5, { min: 0.05, max: 0.95, step: 0.05, label: 'α (neighbours vs. own bag)' }),
    gamma: float(0.2, { min: 0.02, max: 2, step: 0.02, label: 'kernel γ' }),
    seed: int(1, { min: 1, max: 20, step: 1, label: 'seed', format: (v) => String(v) }),
  })

  const data = useMemo(() => xorBags(state.size, state.config, state.seed), [state.size, state.config, state.seed])
  const a = state.alpha
  const gm = state.gamma

  const r = useMemo(() => {
    const n = data.y.length
    const W = data.x1.map((_, i) =>
      data.x1.map((_, j) =>
        i === j ? 0 : Math.exp(-gm * ((data.x1[i] - data.x1[j]) ** 2 + (data.x2[i] - data.x2[j]) ** 2)),
      ),
    )
    // S = D⁻¹W; M = I − αS. Rows with no weight (isolated points) keep their own value.
    const M = W.map((row, i) => {
      const d = row.reduce((s, v) => s + v, 0)
      return row.map((v, j) => (i === j ? 1 : 0) - (d > 0 ? (a * v) / d : 0))
    })
    const solve = luSolver(M)
    const start: number[] = data.bag.map((k) => CONFIGS[state.config][k])
    const propagate = (f: number[]) => solve(f.map((v) => (1 - a) * v))
    const members = [0, 1, 2].map((k) => data.bag.flatMap((b, i) => (b === k ? [i] : [])))
    const project = (f: number[]) => {
      const out = [...f]
      for (let s = 0; s < SWEEPS; s++) {
        members.forEach((idx, k) => {
          const shift = (data.counts[k] - idx.reduce((t, i) => t + out[i], 0)) / idx.length
          for (const i of idx) out[i] += shift
        })
        for (let i = 0; i < n; i++) out[i] = Math.min(1, Math.max(0, out[i]))
      }
      return out
    }
    const unconstrained = propagate(start)
    let f = start
    for (let t = 0; t < ROUNDS; t++) f = project(propagate(f))
    // Bag-majority baseline: every point takes its bag's majority label.
    const majority = start.map((p) => (p > 0.5 ? 1 : 0))
    return {
      f,
      acc: accuracy(f, data.y),
      unc: accuracy(unconstrained, data.y),
      maj: accuracy(majority, data.y),
    }
  }, [data, a, gm, state.config])

  const bags = [
    {
      name: 'bags',
      x: data.x1,
      y: data.x2,
      group: data.bag,
      groupNames: CONFIGS[state.config].map((p, k) => `bag ${k + 1} (π = ${p})`),
    },
  ] as const
  const labels = [
    {
      name: 'LP-LLP labels',
      x: data.x1,
      y: data.x2,
      group: r.f.map((v) => (v > 0.5 ? 1 : 0)),
      groupNames: ['predicted 0', 'predicted 1'],
    },
  ] as const

  const xAxis = useAxis({ label: 'x₁', range: RANGE })
  const yAxis = useAxis({ label: 'x₂', range: Y_RANGE, equal: xAxis })
  const xAxis2 = useAxis({ label: 'x₁', range: RANGE })
  const yAxis2 = useAxis({ label: 'x₂', range: Y_RANGE, equal: xAxis2 })
  return (
    <Figure
      title="Label propagation with bag-mass constraints on XOR data"
      state={state}
      caption="Three bags of XOR data, following the synthetic set-up of the LP-LLP paper: class 1 sits at corners (0, 0) and (10, 10), class 0 at the other two, and each bag reveals only its proportion of class 1 (configuration A: 0.6, 0.4, 0.5; B: 0.85, 0.25, 0.4). Left: bag membership, which is unrelated to position. Right: the labels LP-LLP assigns after ten rounds of propagation and projection. Configuration B, whose proportions differ more, is recovered almost perfectly; A, whose proportions all lie near one half, is harder. Propagation alone, without the projection step, and the bag-majority rule are shown for comparison. The figure is a sketch of the method, not a reproduction of the paper's tables."

      readouts={
        <>
          <Readout label="LP-LLP accuracy" value={formatNumber(r.acc)} />
          <Readout label="propagation only" value={formatNumber(r.unc)} />
          <Readout label="bag majority" value={formatNumber(r.maj)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis}>
          <Points {...bags[0]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2}>
          <Points {...labels[0]} />
        </Plot>
      </div>
    </Figure>
  )
}
