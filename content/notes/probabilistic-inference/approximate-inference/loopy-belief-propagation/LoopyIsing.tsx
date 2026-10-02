import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from 'aifn-render'

const SIDE = 3
const N = SIDE * SIDE
const EDGES: [number, number][] = []
for (let r = 0; r < SIDE; r++)
  for (let c = 0; c < SIDE; c++) {
    if (c + 1 < SIDE) EDGES.push([r * SIDE + c, r * SIDE + c + 1])
    if (r + 1 < SIDE) EDGES.push([r * SIDE + c, (r + 1) * SIDE + c])
  }
const NEIGHBOURS: number[][] = Array.from({ length: N }, () => [])
EDGES.forEach(([i, j]) => {
  NEIGHBOURS[i].push(j)
  NEIGHBOURS[j].push(i)
})
const MAX_ITERS = 500
const TOL = 1e-9
const SPINS = [-1, 1]
const NODES = Array.from({ length: N }, (_, i) => i + 1)

/** Exact marginals P(x_i = +1) and log Z by enumerating all 2⁹ spin configurations. */
function exact(J: number, h: number) {
  const weights: number[] = []
  let top = -Infinity
  for (let s = 0; s < 1 << N; s++) {
    const x = (i: number) => ((s >> i) & 1 ? 1 : -1)
    let e = 0
    EDGES.forEach(([i, j]) => (e += J * x(i) * x(j)))
    for (let i = 0; i < N; i++) e += h * x(i)
    weights.push(e)
    top = Math.max(top, e)
  }
  let Z = 0
  const up = new Array<number>(N).fill(0)
  weights.forEach((e, s) => {
    const w = Math.exp(e - top)
    Z += w
    for (let i = 0; i < N; i++) if ((s >> i) & 1) up[i] += w
  })
  return { marginals: up.map((u) => u / Z), logZ: top + Math.log(Z) }
}

/** Parallel (flooding) sum-product on the grid, messages normalised to sum to 1, with damping. */
function loopy(J: number, h: number, damping: number) {
  const psiI = SPINS.map((s) => Math.exp(h * s))
  const psiIJ = SPINS.map((a) => SPINS.map((b) => Math.exp(J * a * b)))
  const key = (i: number, j: number) => i * N + j
  let msg = new Map<number, number[]>()
  EDGES.forEach(([i, j]) => {
    msg.set(key(i, j), [0.5, 0.5])
    msg.set(key(j, i), [0.5, 0.5])
  })
  // Product of the node potential and all incoming messages except the one from `except`.
  const incoming = (m: Map<number, number[]>, i: number, except: number) =>
    SPINS.map((_, a) => NEIGHBOURS[i].reduce((p, k) => (k === except ? p : p * m.get(key(k, i))![a]), psiI[a]))
  let iterations = MAX_ITERS
  for (let t = 0; t < MAX_ITERS; t++) {
    const next = new Map<number, number[]>()
    let change = 0
    msg.forEach((old, k) => {
      const i = Math.floor(k / N)
      const j = k % N
      const pre = incoming(msg, i, j)
      const raw = SPINS.map((_, b) => pre[0] * psiIJ[0][b] + pre[1] * psiIJ[1][b])
      const total = raw[0] + raw[1]
      const m = raw.map((v, b) => (1 - damping) * (v / total) + damping * old[b])
      change = Math.max(change, Math.abs(m[0] - old[0]))
      next.set(k, m)
    })
    msg = next
    if (change < TOL) {
      iterations = t + 1
      break
    }
  }
  const beliefs = Array.from({ length: N }, (_, i) => incoming(msg, i, -1))
  const marginals = beliefs.map((b) => b[1] / (b[0] + b[1]))
  // Bethe approximation to log Z = −F_Bethe, from edge and node beliefs at the final messages.
  let F = 0
  EDGES.forEach(([i, j]) => {
    const bi = incoming(msg, i, j)
    const bj = incoming(msg, j, i)
    const raw = SPINS.map((_, a) => SPINS.map((_, b) => bi[a] * psiIJ[a][b] * bj[b]))
    const total = raw.flat().reduce((s, v) => s + v, 0)
    SPINS.forEach((_, a) =>
      SPINS.forEach((_, b) => {
        const p = raw[a][b] / total
        if (p > 0) F += p * (Math.log(p) - Math.log(psiIJ[a][b] * psiI[a] * psiI[b]))
      }),
    )
  })
  marginals.forEach((p1, i) => {
    const d = NEIGHBOURS[i].length
    ;[1 - p1, p1].forEach((p, a) => {
      if (p > 0) F -= (d - 1) * p * (Math.log(p) - Math.log(psiI[a]))
    })
  })
  return { marginals, iterations, converged: iterations < MAX_ITERS, logZ: -F }
}

export function LoopyIsing() {
  const J = useParam(0.5, { min: -1.5, max: 1.5, step: 0.05 })
  const h = useParam(0.1, { min: -0.5, max: 0.5, step: 0.01 })
  const damping = useParam(0, { min: 0, max: 0.9, step: 0.05 })

  const truth = useMemo(() => exact(J.value, h.value), [J.value, h.value])
  const bp = useMemo(() => loopy(J.value, h.value, damping.value), [J.value, h.value, damping.value])

  const series: XYSeries[] = useMemo(
    () => [
      { name: 'exact P(xᵢ = +1)', type: 'bar', x: NODES, y: truth.marginals, slot: 0 },
      { name: 'loopy BP belief', type: 'scatter', x: NODES, y: bp.marginals, slot: 1 },
    ],
    [truth, bp],
  )
  const maxError = Math.max(...truth.marginals.map((p, i) => Math.abs(p - bp.marginals[i])))

  return (
    <Interactive
      title="Loopy belief propagation on a 3 × 3 Ising grid"
      caption="Spins xᵢ ∈ {−1, +1} on a 3 × 3 grid (nodes numbered row by row; node 5 is the centre), with coupling J between neighbours and field h on every node. Bars are the exact marginals from all 512 configurations; points are loopy BP beliefs after parallel updates. For weak coupling the two agree closely. Strong positive coupling makes loopy BP overconfident, because evidence circulates around the grid's cycles and is counted repeatedly. Strong negative coupling (antiferromagnetic) makes the undamped messages oscillate; damping restores convergence but not accuracy."
      controls={
        <>
          <ParamSlider label="coupling J" param={J} />
          <ParamSlider label="field h" param={h} />
          <ParamSlider label="damping" param={damping} />
        </>
      }
      readout={
        <>
          <Readout label="iterations" value={bp.converged ? bp.iterations : `no convergence in ${MAX_ITERS}`} />
          <Readout label="largest marginal error" value={formatNumber(maxError)} />
          <Readout label="log Z, exact" value={formatNumber(truth.logZ)} />
          <Readout label="log Z, Bethe" value={formatNumber(bp.logZ)} />
        </>
      }
    >
      <XYChart series={series} xLabel="node" yLabel="P(xᵢ = +1)" yRange={[0, 1]} height={280} />
    </Interactive>
  )
}
