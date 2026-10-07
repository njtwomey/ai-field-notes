import { useState } from 'react'
import {
  Bars,
  choice,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Player,
  Plot,
  Points,
  Raster,
  Readout,
  setting,
  slider,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { eigh, inverse } from 'aifn-compute/numerics/linalg'
import { tensor, toArray, toFlat } from 'aifn-compute/foundation/tensor'
import { stream, uniform } from 'aifn-compute/foundation/random'

const G = 12
const N = G * G
const AXIS = Array.from({ length: G }, (_, k) => k / (G - 1))
const PX = Array.from({ length: N }, (_, i) => AXIS[i % G])
const PY = Array.from({ length: N }, (_, i) => AXIS[Math.floor(i / G)])
const SHOWN = 48
const BAR_X = Array.from({ length: SHOWN }, (_, k) => k + 1)

type Kernel = { L: number[][]; lambda: number[]; V: number[][] }

/** L_ij = α q_i q_j exp(−‖x_i − x_j‖² / 2σ²), with quality q_i either 1 or a bump centred at (cx, cy). */
function kernel(sigma: number, alpha: number, hill: boolean, cx: number, cy: number): Kernel {
  const q = PX.map((x, i) => (hill ? 0.25 + 1.5 * Math.exp(-((x - cx) ** 2 + (PY[i] - cy) ** 2) / (2 * 0.2 ** 2)) : 1))
  const L = Array.from({ length: N }, (_, i) =>
    Array.from(
      { length: N },
      (_, j) => alpha * q[i] * q[j] * Math.exp(-((PX[i] - PX[j]) ** 2 + (PY[i] - PY[j]) ** 2) / (2 * sigma * sigma)),
    ),
  )
  const { values, vectors } = eigh(tensor(L))
  const lambda = toFlat(values).map((v) => Math.max(0, v))
  const rows = toArray(vectors) as number[][]
  // Columns of the eigenvector matrix, one array per eigenvector.
  const V = lambda.map((_, n) => rows.map((row) => row[n]))
  return { L, lambda, V }
}

type Frame = {
  /** The field over the grid drawn at this step (row-major, G × G). */
  field: number[][]
  picks: number[]
  label: string
}

const toGrid = (v: ArrayLike<number>) =>
  Array.from({ length: G }, (_, r) => Array.from({ length: G }, (_, c) => v[r * G + c]))

/** Draw an index from probabilities p (summing to 1) with one uniform number. */
function draw(p: ArrayLike<number>, u: number): number {
  let i = 0
  while (i < N - 1 && u >= p[i]) u -= p[i++]
  return i
}

/**
 * The spectral sampler (Hough et al.; Kulesza and Taskar, Algorithm 1). Phase 1 keeps eigenvector n with probability
 * λ_n / (λ_n + 1). Phase 2 picks one point per kept vector, in one of two equivalent ways:
 * - `basis`: point i with probability Σ_v v_i² / |V|, then replace the kept vectors by an orthonormal basis of their
 *   span orthogonal to e_i (elimination, then Gram–Schmidt over the N-long columns);
 * - `rows`: Gram–Schmidt over the k-long rows instead (Tremblay et al., 2018). Point i's probability is the squared
 *   norm of its row's residual after projecting out the rows already picked, divided by the number of picks left.
 * Both give point i the same conditional probability at every step, so the same random numbers give the same sample.
 */
function sampleTrace(
  k: Kernel,
  seed: number,
  phase2: 'basis' | 'rows',
): { frames: Frame[]; kept: boolean[]; marginal: number[] } {
  const r = stream(seed)
  const { lambda, V } = k
  const marginal = Array.from({ length: N }, (_, i) =>
    lambda.reduce((s, l, n) => s + (l / (l + 1)) * V[n][i] * V[n][i], 0),
  )
  const kept = lambda.map((l) => uniform(r) < l / (l + 1))
  const frames: Frame[] = [{ field: toGrid(marginal), picks: [], label: 'P(i ∈ Y)' }]
  const picks: number[] = []
  const label = 'P(next = i)'

  if (phase2 === 'rows') {
    const cols = V.filter((_, n) => kept[n])
    const size = cols.length
    const rows = Array.from({ length: N }, (_, i) => cols.map((v) => v[i]))
    const residual = rows.map((y) => y.reduce((t, x) => t + x * x, 0))
    const done: number[][] = [] // orthonormalised rows of the picked points
    const next = () => residual.map((v) => Math.max(0, v) / (size - picks.length))
    if (size) frames.push({ field: toGrid(next()), picks: [], label })
    while (picks.length < size) {
      const i = draw(next(), uniform(r))
      picks.push(i)
      const e = [...rows[i]]
      for (const d of done) {
        const c = d.reduce((t, x, m) => t + x * rows[i][m], 0)
        for (let m = 0; m < size; m++) e[m] -= c * d[m]
      }
      const nrm = Math.sqrt(e.reduce((t, x) => t + x * x, 0))
      for (let m = 0; m < size; m++) e[m] /= nrm
      done.push(e)
      for (let j = 0; j < N; j++) residual[j] -= rows[j].reduce((t, x, m) => t + x * e[m], 0) ** 2
      frames.push({ field: toGrid(picks.length < size ? next() : new Float64Array(N)), picks: [...picks], label })
    }
    return { frames, kept, marginal }
  }

  let basis = V.filter((_, n) => kept[n]).map((v) => Float64Array.from(v))
  const next = () => {
    const p = new Float64Array(N)
    for (const v of basis) for (let i = 0; i < N; i++) p[i] += (v[i] * v[i]) / basis.length
    return p
  }
  if (basis.length) frames.push({ field: toGrid(next()), picks: [], label })
  while (basis.length) {
    const i = draw(next(), uniform(r))
    picks.push(i)
    // Eliminate e_i: use the vector with the largest i-th entry to zero the others' i-th entries, then drop it.
    let j = 0
    for (let m = 1; m < basis.length; m++) if (Math.abs(basis[m][i]) > Math.abs(basis[j][i])) j = m
    const vj = basis[j]
    basis = basis
      .filter((_, m) => m !== j)
      .map((v) => {
        const f = v[i] / vj[i]
        return v.map((x, t) => x - f * vj[t])
      })
    // Modified Gram–Schmidt restores an orthonormal basis.
    for (let a = 0; a < basis.length; a++) {
      for (let b = 0; b < a; b++) {
        let d = 0
        for (let t = 0; t < N; t++) d += basis[a][t] * basis[b][t]
        for (let t = 0; t < N; t++) basis[a][t] -= d * basis[b][t]
      }
      let nrm = 0
      for (let t = 0; t < N; t++) nrm += basis[a][t] ** 2
      nrm = Math.sqrt(nrm)
      for (let t = 0; t < N; t++) basis[a][t] /= nrm
    }
    frames.push({ field: toGrid(basis.length ? next() : new Float64Array(N)), picks: [...picks], label })
  }
  return { frames, kept, marginal }
}

/**
 * The sequential sampler without an eigendecomposition (Poulson, 2019; Launay et al., 2020). It visits the points in
 * order. Point j joins with its current conditional probability A_jj; the rest of A is then conditioned on that
 * outcome by one step of an LU factorisation whose pivot is A_jj if j joined and A_jj − 1 if it did not.
 */
function sequentialTrace(k: Kernel, seed: number): { frames: Frame[]; visits: { p: number; joined: boolean }[] } {
  const r = stream(seed)
  // K = I − (L + I)⁻¹: one inverse, no eigendecomposition.
  const inv = toArray(inverse(tensor(k.L.map((row, i) => row.map((v, j) => v + (i === j ? 1 : 0)))))) as number[][]
  const A = inv.map((row, i) => row.map((v, j) => (i === j ? 1 : 0) - v))
  const picks: number[] = []
  const visits: { p: number; joined: boolean }[] = []
  const label = 'P(i ∈ Y | so far)'
  const field = (from: number) => Array.from({ length: N }, (_, i) => (i < from ? NaN : A[i][i]))
  const frames: Frame[] = [{ field: toGrid(field(0)), picks: [], label }]
  for (let j = 0; j < N; j++) {
    const p = Math.min(1, Math.max(0, A[j][j]))
    const joined = uniform(r) < p
    visits.push({ p, joined })
    if (joined) picks.push(j)
    const pivot = joined ? A[j][j] : A[j][j] - 1
    if (Math.abs(pivot) > 1e-12)
      for (let i = j + 1; i < N; i++) {
        const f = A[i][j] / pivot
        if (f !== 0) for (let m = j + 1; m < N; m++) A[i][m] -= f * A[j][m]
      }
    frames.push({ field: toGrid(field(j + 1)), picks: [...picks], label })
  }
  return { frames, visits }
}

/**
 * Greedy MAP inference with an incremental Cholesky factor (Chen et al., 2018). d_i² = det(L_{S∪i}) / det(L_S) is the
 * factor by which adding i multiplies the determinant; each step adds the largest, up to a budget of k points.
 */
function mapTrace(k: Kernel, budget: number): { frames: Frame[]; gains: number[] } {
  const { L } = k
  const d2 = L.map((row, i) => row[i])
  const c: number[][] = Array.from({ length: N }, () => [])
  const picks: number[] = []
  const gains: number[] = []
  const frames: Frame[] = [{ field: toGrid(d2), picks: [], label: 'gain d_i²' }]
  for (;;) {
    let j = -1
    for (let i = 0; i < N; i++) if (!picks.includes(i) && (j < 0 || d2[i] > d2[j])) j = i
    if (picks.length >= budget || j < 0 || d2[j] <= 1e-12) break
    const dj = Math.sqrt(d2[j])
    picks.push(j)
    gains.push(d2[j])
    for (let i = 0; i < N; i++) {
      if (picks.includes(i)) continue
      let dot = 0
      for (let t = 0; t < c[j].length; t++) dot += c[j][t] * c[i][t]
      const e = (L[j][i] - dot) / dj
      c[i].push(e)
      d2[i] = Math.max(0, d2[i] - e * e)
    }
    c[j].push(dj)
    const shown = d2.map((v, i) => (picks.includes(i) ? NaN : v))
    frames.push({ field: toGrid(shown), picks: [...picks], label: 'gain d_i²' })
  }
  return { frames, gains }
}

export function DppSampler() {
  const state = useFigureState({
    mode: choice(
      [
        { value: 'sample', label: 'sample: spectral, basis elimination' },
        { value: 'rows', label: 'sample: spectral, row Gram–Schmidt' },
        { value: 'sequential', label: 'sample: sequential, no eigendecomposition' },
        { value: 'map', label: 'most likely set (greedy MAP)' },
      ],
      'sample',
      { label: 'algorithm' },
    ),
    sigma: float(0.12, {
      min: 0.04,
      max: 0.4,
      step: 0.01,
      label: 'similarity length σ',
      suggestions: [0.06, 0.12, 0.25],
    }),
    alpha: float(0.5, { min: 0.05, max: 50, label: 'kernel scale α', suggestions: [0.3, 1, 5, 20] }),
    hill: setting(false, 'quality hill'),
    cx: slider(0, 1, 0.3, { step: 0.01, onChart: true }),
    cy: slider(0, 1, 0.65, { step: 0.01, onChart: true }),
    seed: int(1, { min: 1, max: 9999, label: 'seed' }),
    compare: setting(false, 'compare with independent points'),
  })
  const { mode, sigma, alpha, hill, cx, cy, seed } = state

  const k = useComputed(() => kernel(sigma, alpha, hill, cx, cy), [sigma, alpha, hill, cx, cy])
  const run = useComputed(() => {
    const kk = k.value
    return mode === 'map'
      ? {
          ...mapTrace(kk, Math.max(1, Math.round(kk.lambda.reduce((t, l) => t + l / (l + 1), 0)))),
          kind: 'map' as const,
        }
      : mode === 'sequential'
        ? { ...sequentialTrace(kk, seed), kind: 'sequential' as const }
        : { ...sampleTrace(kk, seed, mode === 'rows' ? 'rows' : 'basis'), kind: 'sample' as const }
  }, [k.value, mode, seed])
  const { frames } = run.value

  const key = `${mode}|${sigma}|${alpha}|${hill}|${cx}|${cy}|${seed}`
  const [pos, setPos] = useState({ key, step: 0 })
  const step = pos.key === key ? Math.min(pos.step, frames.length - 1) : 0
  const frame = frames[step]

  const lambda = k.value.lambda
  const expected = lambda.reduce((s, l) => s + l / (l + 1), 0)
  const inclusion = lambda.slice(0, SHOWN).map((l) => l / (l + 1))
  const kept = run.value.kind === 'sample' ? run.value.kept : null
  const keptX = kept && step >= 1 ? BAR_X.filter((_, n) => kept[n]) : []
  const keptY = kept && step >= 1 ? inclusion.filter((_, n) => kept[n]) : []

  const indep = (() => {
    if (!state.compare || run.value.kind !== 'sample') return { x: [] as number[], y: [] as number[] }
    const r = stream(seed + 7919)
    const m = run.value.marginal
    const idx = m.map((p, i) => (uniform(r) < p ? i : -1)).filter((i) => i >= 0)
    return { x: idx.map((i) => PX[i]), y: idx.map((i) => PY[i]) }
  })()

  // Anchor the colour scale at zero, so a near-uniform field reads as a level rather than as empty.
  const fieldMax = Math.max(1e-9, ...frame.field.flat().filter(Number.isFinite))
  const fieldRange: [number, number] = [0, fieldMax]
  const last = frame.picks.at(-1)
  const px = frame.picks.map((i) => PX[i])
  const py = frame.picks.map((i) => PY[i])
  const logdet =
    run.value.kind === 'map' ? run.value.gains.slice(0, frame.picks.length).reduce((s, g) => s + Math.log(g), 0) : NaN

  const gx = useAxis({ label: 'x₁', range: [-0.05, 1.05] })
  const gy = useAxis({ label: 'x₂', range: [-0.05, 1.05], equal: gx })
  const bx = useAxis({ label: 'eigenvector n (largest λ first)', range: [0, SHOWN + 1], integer: true })
  const by = useAxis({ label: 'λₙ / (λₙ + 1)', range: [0, 1] })
  const mx = useAxis({ label: 'pick', range: [0, undefined], integer: true })
  const my = useAxis({ label: 'log gain, log d²' })
  const vx = useAxis({ label: 'visit (row by row)', range: [0, N + 1], integer: true })
  const vy = useAxis({ label: 'probability at the visit', range: [0, 1] })
  const ix = useAxis({ label: 'x₁', range: [-0.05, 1.05] })
  const iy = useAxis({ label: 'x₂', range: [-0.05, 1.05], equal: ix })

  const total = frames.length - 1
  const phase =
    run.value.kind === 'map'
      ? step === 0
        ? 'gains before any pick'
        : `pick ${step}: largest gain`
      : run.value.kind === 'sequential'
        ? step === 0
          ? 'before any visit'
          : `visited ${step} of ${N}: ${run.value.visits[step - 1].joined ? 'joined' : 'left out'}`
        : step === 0
          ? 'the process: inclusion probabilities'
          : step === 1
            ? `phase 1: kept ${keptX.length} eigenvectors`
            : `phase 2: pick ${step - 1} of ${total - 1}`

  return (
    <Figure
      title="Sampling a determinantal point process, step by step"
      defaultSize="L"
      state={state}
      caption={
        run.value.kind === 'sample'
          ? `A DPP over a 12 × 12 grid with a Gaussian similarity kernel. Step 0 colours each point by its inclusion probability. Step 1 is phase 1: each eigenvector of L is kept independently with probability λ/(λ + 1) (right, kept bars coloured), which fixes the sample size. Each later step picks one point with probability proportional to the colour, then projects away from it: the colour drains around every pick, so the next pick lands elsewhere. ${mode === 'rows' ? 'This version orthonormalises the picked points\u2019 rows of the eigenvector matrix instead of the kept vectors; switch between the two with the same seed and the samples are identical.' : 'The row Gram\u2013Schmidt version computes the same probabilities more cheaply; with the same seed it gives the same sample.'} Lengthen σ to widen the repulsion, raise α for larger samples, and switch on the quality hill (drag its centre) to trade quality against spread. Compare shows independent points with the same inclusion probabilities, which clump and leave gaps.`
          : run.value.kind === 'sequential'
            ? 'The sequential sampler needs no eigendecomposition. It visits the points row by row; the colour is each unvisited point\u2019s inclusion probability given what has been decided so far. The visited point joins with that probability (right: each visit\u2019s probability, coloured if it joined). Joining lowers the probabilities of similar points; being left out raises them. One LU-style update conditions the rest of the kernel on each outcome.'
            : 'Greedy MAP inference for the same kernel. The colour is the gain d_i², the factor by which adding point i multiplies det(L_S); it is the squared distance from the feature vector of point i to the span of those already chosen. Each step adds the largest gain, and the gains of nearby points collapse. The search stops after k picks, k the expected size of a sample, as diversified ranking fixes the length of a slate; a gain below 1 means the point lowers det(L_S) and would be skipped by an unconstrained MAP search.'
      }
      controls={
        <Player
          value={step}
          onChange={(s) => setPos({ key, step: s })}
          count={frames.length}
          label="step"
          format={() => phase}
        />
      }
      readouts={
        <>
          <Readout label="expected size Σ λ/(λ+1)" value={formatNumber(expected)} />
          <Readout label="points chosen" value={String(frame.picks.length)} />
          {run.value.kind === 'map' ? (
            <Readout label="log det L_S" value={formatNumber(logdet)} />
          ) : (
            <Readout label="independent points" value={state.compare ? String(indep.x.length) : '—'} />
          )}
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-[2fr_1fr]">
        <Plot x={gx} y={gy} height={440}>
          <Raster
            x={AXIS}
            y={AXIS}
            z={frame.field}
            scale="sequential"
            range={fieldRange}
            valueLabel={frame.label}
            stale={run.stale}
          />
          <Points name="chosen" x={px} y={py} emphasis size={10} />
          {last !== undefined && <Points name="latest" x={[PX[last]]} y={[PY[last]]} slot={1} size={14} />}
          {hill && <Handle {...state.handle(['cx', 'cy'], { label: 'quality peak' })} />}
        </Plot>
        <div className="grid gap-4">
          {run.value.kind === 'sample' ? (
            <Plot x={bx} y={by} height={state.compare ? 210 : 440}>
              <Bars name="λ/(λ + 1)" x={BAR_X} y={inclusion} muted />
              <Bars name="kept" x={keptX} y={keptY} slot={0} />
            </Plot>
          ) : run.value.kind === 'sequential' ? (
            <Plot x={vx} y={vy} height={440}>
              <Bars
                name="left out"
                x={run.value.visits.slice(0, step).flatMap((v, n) => (v.joined ? [] : [n + 1]))}
                y={run.value.visits.slice(0, step).flatMap((v) => (v.joined ? [] : [v.p]))}
                muted
              />
              <Bars
                name="joined"
                x={run.value.visits.slice(0, step).flatMap((v, n) => (v.joined ? [n + 1] : []))}
                y={run.value.visits.slice(0, step).flatMap((v) => (v.joined ? [v.p] : []))}
                slot={0}
              />
            </Plot>
          ) : (
            <Plot x={mx} y={my} height={440}>
              <Bars
                name="log d² of each pick"
                x={run.value.gains.slice(0, frame.picks.length).map((_, n) => n + 1)}
                y={run.value.gains.slice(0, frame.picks.length).map(Math.log)}
                slot={0}
              />
            </Plot>
          )}
          {state.compare && run.value.kind === 'sample' && (
            <Plot x={ix} y={iy} height={210}>
              <Points name="independent" x={indep.x} y={indep.y} slot={2} size={8} />
            </Plot>
          )}
        </div>
      </div>
    </Figure>
  )
}
