import { describe, expect, it } from 'vitest'
import {
  bellmanFord,
  boxQuadprog,
  branchAndBound,
  dijkstra,
  dp,
  floydWarshall,
  gomory,
  hungarian,
  interiorPoint,
  linprog,
  lpCentralPath,
  milp,
  minCostFlow,
  quadprog,
  shortestPath,
  simplex,
  type LinearProgram,
} from 'aifn/optim/programming'
import {
  editDistance,
  knapsack,
  lcs,
  needlemanWunsch,
  smithWaterman,
  unboundedKnapsack,
} from 'aifn-applied/algorithms/dynamic-programming'
import { toFlat } from 'aifn/foundation/tensor'
import { extend, run, seek, trace } from 'aifn/foundation/trace'

const close = (a: ArrayLike<number>, b: ArrayLike<number>, tol = 1e-8) => {
  expect(a.length).toBe(b.length)
  for (let i = 0; i < a.length; i++) expect(Math.abs(a[i] - b[i])).toBeLessThan(tol)
}

// Hillier and Lieberman's Wyndor Glass problem, written as a minimisation. Reference values from scipy's linprog.
const wyndor: LinearProgram = {
  c: [-3, -5],
  A_ub: [
    [1, 0],
    [0, 2],
    [3, 2],
  ],
  b_ub: [4, 12, 18],
}

describe('linear programming', () => {
  it('simplex and interior point agree with scipy on the Wyndor problem, with duals', () => {
    for (const method of ['simplex', 'interior-point'] as const) {
      const r = linprog(wyndor, { method })
      expect(r.status).toBe('optimal')
      const tol = method === 'simplex' ? 1e-10 : 1e-6
      close(toFlat(r.x), [2, 6], tol)
      expect(r.objective).toBeCloseTo(-36, 5)
      close(toFlat(r.report!.duals.ineq), [0, -1.5, -1], tol)
      expect(Math.abs(r.report!.dualityGap)).toBeLessThan(1e-6)
      expect(r.report!.complementarity).toBeLessThan(1e-6)
    }
  })

  it('handles free and doubly bounded variables and equality rows (scipy marginals)', () => {
    const p: LinearProgram = {
      c: [1, 2, -1],
      A_eq: [[1, 1, 1]],
      b_eq: [3],
      A_ub: [[1, -1, 0]],
      b_ub: [-1],
      bounds: [
        [null, null],
        [0, 4],
        [-2, 2],
      ],
    }
    const r = linprog(p)
    close(toFlat(r.x), [0, 1, 2])
    close(toFlat(r.report!.duals.ineq), [-0.5])
    close(toFlat(r.report!.duals.eq), [1.5])
    close(toFlat(r.report!.duals.upper), [0, 0, -2.5])
    expect(linprog(p, { method: 'interior-point' }).objective).toBeCloseTo(0, 6)
  })

  it('reports infeasible and unbounded problems', () => {
    expect(linprog({ c: [1], A_ub: [[1], [-1]], b_ub: [1, -2] }).status).toBe('infeasible')
    const u = linprog({ c: [-1, 0], A_ub: [[1, -1]], b_ub: [1] })
    expect(u.status).toBe('unbounded')
    expect(u.objective).toBe(-Infinity)
    expect(linprog({ c: [-1, 0], A_ub: [[1, -1]], b_ub: [1] }, { method: 'interior-point' }).status).toBe('diverged')
  })

  it("Dantzig's rule visits every vertex of the Klee–Minty cube; Bland's rule does not cycle on Beale's example", () => {
    // Chvátal (1983), p. 47: max Σ 10^(n−j) x_j s.t. 2 Σ_{j<i} 10^(i−j) x_j + x_i ≤ 100^(i−1).
    const n = 3
    const c = Array.from({ length: n }, (_, j) => -(10 ** (n - j - 1)))
    const A = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => (j < i ? 2 * 10 ** (i - j) : j === i ? 1 : 0)),
    )
    const b = Array.from({ length: n }, (_, i) => 100 ** i)
    const s = run(simplex, { problem: { c, A_ub: A, b_ub: b }, rule: 'dantzig' }, 100)
    expect(s.status).toBe('optimal')
    expect(s.iteration).toBe(2 ** n - 1)
    expect(s.objective).toBeCloseTo(-(100 ** (n - 1)), 8)
    // Beale (1955).
    const beale: LinearProgram = {
      c: [-0.75, 20, -0.5, 6],
      A_ub: [
        [0.25, -8, -1, 9],
        [0.5, -12, -0.5, 3],
        [0, 0, 1, 0],
      ],
      b_ub: [0, 0, 1],
    }
    expect(run(simplex, { problem: beale, rule: 'dantzig' }, 100).status).toBe('cycling')
    const bland = run(simplex, { problem: beale, rule: 'bland' }, 100)
    expect(bland.status).toBe('optimal')
    expect(bland.objective).toBeCloseTo(-1.25, 10)
  })

  it('the simplex and interior-point algorithms follow the trace protocol', () => {
    const t = trace(simplex, { problem: wyndor }, 50)
    expect(seek(simplex, { problem: wyndor }, 2).x).toEqual(run(simplex, { problem: wyndor }, 2).x)
    const short = trace(interiorPoint, { problem: wyndor }, 3, { record: { mu: (s) => s.mu } })
    const long = trace(interiorPoint, { problem: wyndor }, 6, { record: { mu: (s) => s.mu } })
    expect(toFlat(extend(short, interiorPoint, { problem: wyndor }, 3).series.mu)).toEqual(toFlat(long.series.mu))
    expect(t.meta.stopped).toBe('done')
  })

  it('the central path tends to the optimum as μ → 0', () => {
    const path = lpCentralPath(wyndor, [10, 1, 0.1, 1e-4])
    expect(toFlat(path.converged)).toEqual([1, 1, 1, 1])
    close(toFlat(path.x).slice(6), [2, 6], 1e-3)
  })
})

describe('quadratic programming', () => {
  // min ½‖x − (1, 2.5)‖² s.t. the polygon of Nocedal and Wright, Example 16.3; optimum (1.4, 1.7).
  const qp = {
    Q: [
      [2, 0],
      [0, 2],
    ],
    c: [-2, -5],
    A: [
      [-1, 2],
      [1, 2],
      [1, -2],
      [-1, 0],
      [0, -1],
    ],
    b: [2, 6, 2, 0, 0],
  }
  it('active set and interior point reach the same KKT point', () => {
    for (const method of ['active-set', 'interior-point'] as const) {
      const r = quadprog(qp, { method, x0: method === 'active-set' ? [2, 0] : undefined })
      expect(r.status).toBe('optimal')
      close(toFlat(r.x), [1.4, 1.7], 1e-7)
      expect(r.report.stationarity).toBeLessThan(1e-7)
      close(toFlat(r.report.lambda), [0.8, 0, 0, 0, 0], 1e-7)
    }
    expect(quadprog(qp).status).toBe('optimal')
  })

  it('box QP projects and solves the free subspace', () => {
    const s = boxQuadprog({
      Q: [
        [2, 0.5],
        [0.5, 1],
      ],
      c: [-4, 1],
      lower: [0, 0],
      upper: [1, 1],
    })
    expect(s.converged).toBe(true)
    close(toFlat(s.x), [1, 0], 1e-9)
  })
})

describe('integer programming', () => {
  // max x + y s.t. −2x + 2y ≥ 1, −8x + 10y ≤ 13, x, y ≥ 0 integer: optimum (1, 2), value 3.
  const ip = {
    c: [-1, -1],
    A_ub: [
      [2, -2],
      [-8, 10],
    ],
    b_ub: [-1, 13],
  }
  it('branch and bound finds the integer optimum and keeps the tree', () => {
    for (const strategy of ['depth-first', 'best-bound', 'breadth-first'] as const) {
      const r = milp(ip, { strategy })
      expect(r.status).toBe('optimal')
      close(toFlat(r.x), [1, 2])
      expect(r.objective).toBe(-3)
      expect(r.tree[0].status).toBe('branched')
    }
    const t = trace(branchAndBound, { problem: ip }, 100)
    expect(t.meta.stopped).toBe('done')
  })

  it('mixed-integer: continuous variables stay continuous', () => {
    const r = milp({ ...ip, integrality: [1, 0] })
    expect(r.objective).toBeCloseTo(-8.5, 9)
  })

  it('Gomory cuts reach the integer optimum and never cut off integer points', () => {
    const s = run(gomory, { problem: ip }, 50)
    expect(s.status).toBe('optimal')
    close(toFlat(s.x), [1, 2], 1e-7)
    for (const cut of s.cuts) {
      const a = toFlat(cut.a)
      expect(a[0] * 1 + a[1] * 2).toBeLessThanOrEqual(cut.b + 1e-9)
    }
  })
})

describe('dynamic programming', () => {
  it('generic dp computes Fibonacci', () => {
    const { table } = dp({ shape: [11], cell: (i, _j, get) => (i < 2 ? i : get(i - 1, 0) + get(i - 2, 0)) })
    expect(toFlat(table)[10]).toBe(55)
  })

  it('knapsacks', () => {
    const r = knapsack([60, 100, 120], [10, 20, 30], 50)
    expect(r.value).toBe(220)
    expect(toFlat(r.take)).toEqual([0, 1, 1])
    const u = unboundedKnapsack([10, 30, 20], [5, 10, 15], 100)
    expect(u.value).toBe(300)
  })

  it('LCS, edit distance and alignments', () => {
    const l = lcs('ABCBDAB', 'BDCABA')
    expect(l.length).toBe(4)
    expect(l.subsequence).toHaveLength(4)
    expect(editDistance('kitten', 'sitting').distance).toBe(3)
    const g = needlemanWunsch('GATTACA', 'GCATGCU')
    expect(g.score).toBe(0)
    expect((g.alignedA as string).replace(/-/g, '')).toBe('GATTACA')
    const s = smithWaterman('TGTTACGG', 'GGTTGACTA', { match: 3, mismatch: -3, gap: -2 })
    expect(s.score).toBe(13)
    expect(s.alignedA).toBe('GTT-AC')
    expect(s.alignedB).toBe('GTTGAC')
  })
})

describe('graphs and assignment', () => {
  const g = {
    nodes: 5,
    edges: [
      { from: 0, to: 1, weight: 4 },
      { from: 0, to: 2, weight: 1 },
      { from: 2, to: 1, weight: 2 },
      { from: 1, to: 3, weight: 1 },
      { from: 2, to: 3, weight: 5 },
      { from: 3, to: 4, weight: 3 },
    ],
  }
  it('shortest paths agree across the three algorithms', () => {
    const d = dijkstra(g, 0)
    expect(toFlat(d.distance)).toEqual([0, 3, 1, 4, 7])
    expect(toFlat(shortestPath(d.predecessor, 0, 4))).toEqual([0, 2, 1, 3, 4])
    expect(toFlat(bellmanFord(g, 0).distance)).toEqual([0, 3, 1, 4, 7])
    expect(toFlat(floydWarshall(g).distance).slice(0, 5)).toEqual([0, 3, 1, 4, 7])
    const negative = bellmanFord(
      {
        nodes: 3,
        edges: [
          { from: 0, to: 1, weight: 1 },
          { from: 1, to: 2, weight: -3 },
          { from: 2, to: 1, weight: 1 },
        ],
      },
      0,
    )
    expect(negative.negativeCycle).not.toBeNull()
  })

  it('Hungarian matches the brute-force optimum, square and rectangular', () => {
    const C = [
      [4, 1, 3],
      [2, 0, 5],
      [3, 2, 2],
    ]
    const r = hungarian(C)
    expect(r.cost).toBe(5)
    expect(toFlat(r.assignment)).toEqual([1, 0, 2])
    expect(hungarian(C, { maximize: true }).cost).toBe(11)
    const rect = hungarian([
      [4, 1],
      [2, 0],
      [3, 2],
    ])
    expect(rect.cost).toBe(3)
  })

  it('min-cost flow sends the supply at least cost', () => {
    const s = minCostFlow({
      nodes: 4,
      arcs: [
        { from: 0, to: 1, capacity: 4, cost: 2 },
        { from: 0, to: 2, capacity: 2, cost: 2 },
        { from: 1, to: 2, capacity: 2, cost: 1 },
        { from: 1, to: 3, capacity: 3, cost: 3 },
        { from: 2, to: 3, capacity: 5, cost: 1 },
      ],
      supply: [4, 0, 0, -4],
    })
    expect(s.status).toBe('optimal')
    expect(s.cost).toBe(14)
  })
})
