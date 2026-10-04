import { useMemo } from 'react'
import { Figure, formatNumber, int, Plot, Points, Raster, Readout, useAxis, useFigureState } from 'aifn-render'

/** A toy interaction matrix: 6 users × 7 items. Users 1–3 share tastes, users 4–6 share others, user 3 bridges. */
const R = [
  [1, 1, 0, 0, 0, 0, 0],
  [0, 1, 1, 0, 0, 0, 0],
  [1, 0, 1, 1, 0, 0, 0],
  [0, 0, 0, 1, 1, 0, 0],
  [0, 0, 0, 0, 1, 1, 1],
  [0, 0, 0, 0, 0, 1, 1],
]
const U = R.length
const M = R[0].length
const N = U + M
const USERS = Array.from({ length: U }, (_, u) => u + 1)
const ITEMS = Array.from({ length: M }, (_, i) => i + 1)
const OBS_X = R.flatMap((row) => row.flatMap((r, i) => (r ? [i + 1] : [])))
const OBS_Y = R.flatMap((row, u) => row.flatMap((r) => (r ? [u + 1] : [])))

/** Symmetrically normalised adjacency of the bipartite graph, D^{-1/2} A D^{-1/2}, with users first. */
function normalisedAdjacency(): number[][] {
  const A = Array.from({ length: N }, () => new Array(N).fill(0))
  R.forEach((row, u) =>
    row.forEach((r, i) => {
      A[u][U + i] = r
      A[U + i][u] = r
    }),
  )
  const deg = A.map((row) => row.reduce((a, b) => a + b, 0))
  return A.map((row, a) => row.map((x, b) => (x ? x / Math.sqrt(deg[a] * deg[b]) : 0)))
}

const matmul = (X: number[][], Y: number[][]) =>
  X.map((row) => Y[0].map((_, j) => row.reduce((s, x, k) => s + x * Y[k][j], 0)))

const AHAT = normalisedAdjacency()
const EYE: number[][] = Array.from({ length: N }, (_, a) => Array.from({ length: N }, (_, b) => (a === b ? 1 : 0)))

/** P = Σ_{k=0}^{K} α_k Â^k with α_k = 1/(K+1): row u says how much of each node's layer-0 embedding reaches user u. */
function propagation(K: number): number[][] {
  let power: number[][] = EYE
  let total = EYE.map((row) => row.map((x) => x / (K + 1)))
  for (let k = 1; k <= K; k++) {
    power = matmul(power, AHAT)
    total = total.map((row, a) => row.map((x, b) => x + power[a][b] / (K + 1)))
  }
  return total
}

const cosine = (a: number[], b: number[]) => {
  const dot = a.reduce((s, x, i) => s + x * b[i], 0)
  const n = Math.sqrt(a.reduce((s, x) => s + x * x, 0) * b.reduce((s, x) => s + x * x, 0))
  return n ? dot / n : 0
}

/** Which item embeddings reach each user after K layers of LightGCN propagation with uniform layer weights. */
export function Propagation() {
  const state = useFigureState({
    layers: int(1, { min: 0, max: 8, step: 1, label: 'propagation layers K', format: (v) => String(v) }),
  })
  const K = state.layers
  const P = useMemo(() => propagation(K), [K])
  const z = useMemo(() => P.slice(0, U).map((row) => row.slice(U)), [P])
  const reached = z[0].filter((x) => x > 1e-9).length
  const similarity = useMemo(() => {
    const rows = P.slice(0, U)
    let s = 0
    let n = 0
    for (let a = 0; a < U; a++)
      for (let b = a + 1; b < U; b++) {
        s += cosine(rows[a], rows[b])
        n++
      }
    return s / n
  }, [P])
  const max = Math.max(1e-6, ...z.flat())

  const xAxis = useAxis({ label: 'item' })
  const yAxis = useAxis({ label: 'user' })
  return (
    <Figure
      title="What K layers of propagation reach"
      state={state}
      caption="Six users and seven items; markers are observed interactions. Each cell shows the weight with which item i's layer-0 embedding enters user u's final LightGCN embedding, Σ α_k (Â^k)_{u,i} with α_k = 1/(K+1). With K = 1 a user receives only the items they interacted with; with K = 3 also items reached through a path user → item → user → item, which is how user 1 is linked to item 3. Step K up with the arrows: as K grows, every user's weights spread over the whole graph and users become alike (over-smoothing)."

      readouts={
        <>
          <Readout label="items reaching user 1" value={`${reached} of ${M}`} />
          <Readout label="mean cosine similarity between users' rows" value={formatNumber(similarity)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Raster x={ITEMS} y={USERS} z={z} range={[0, max]} valueLabel={'weight'} />
        <Points name="observed interaction" x={OBS_X} y={OBS_Y} live />
      </Plot>
    </Figure>
  )
}
