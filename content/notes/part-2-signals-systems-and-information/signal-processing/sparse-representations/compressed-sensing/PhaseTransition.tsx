import { useEffect, useMemo, useState } from 'react'
import { basisPursuit, orthogonalMatchingPursuitSteps } from 'aifn-compute/signal/sparse'
import { child, normals, permutation, stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { live } from 'aifn-compute/foundation/trace'
import {
  Bars,
  choice,
  Curve,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Raster,
  Readout,
  slider,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'

const N = 64
const G = 15
/** Measurements m = 4, 8, …, 60, so δ = m/n runs over the cell centres 1/16, …, 15/16. */
const MS = Array.from({ length: G }, (_, a) => 4 * (a + 1))
const DELTAS = MS.map((m) => m / N)
const RHOS = Array.from({ length: G }, (_, b) => (b + 1) / G)
const INDEX = Array.from({ length: N }, (_, j) => j)
const sparsity = (m: number, rho: number) => Math.max(1, Math.round(rho * m))

type Method = 'omp' | 'bp'

/** One trial: a Gaussian m × n matrix with variance 1/m entries and an s-sparse vector with Gaussian entries. */
function trial(m: number, s: number, seed: number, a: number, b: number, t: number) {
  const r = child(stream(seed), 'cell', a, b, t)
  const flat = toFlat(normals(child(r, 'A'), [m, N], 0, 1 / Math.sqrt(m)))
  const A = Array.from({ length: m }, (_, i) => flat.slice(i * N, (i + 1) * N))
  const support = toFlat(permutation(child(r, 'support'), N)).slice(0, s)
  const values = toFlat(normals(child(r, 'values'), s))
  const x = new Array<number>(N).fill(0)
  support.forEach((j, i) => (x[j] = values[i]))
  const y = A.map((row) => row.reduce((acc, v, j) => acc + v * x[j], 0))
  return { A, x, y, support }
}

const relError = (x: number[], xh: ArrayLike<number>) =>
  Math.sqrt(x.reduce((acc, v, j) => acc + (v - xh[j]) ** 2, 0) / x.reduce((acc, v) => acc + v * v, 0))

/** Solve one trial; orthogonal matching pursuit stops at its first atom outside the true support, a certain failure. */
function solve(method: Method, p: ReturnType<typeof trial>): { xh: number[]; ok: boolean } {
  if (method === 'bp') {
    const xh = toFlat(basisPursuit(p.A, p.y, { method: 'interior-point' }).x).map((v) => (Number.isFinite(v) ? v : 0))
    return { xh, ok: relError(p.x, xh) < 1e-4 }
  }
  const inSupport = new Set(p.support)
  let xh = new Array<number>(N).fill(0)
  for (const { state } of live(orthogonalMatchingPursuitSteps(p.A, p.y, { sparsity: p.support.length }), undefined)) {
    xh = toFlat(state.x)
    const last = state.support.at(-1)
    if (last !== undefined && !inSupport.has(last)) return { xh, ok: false }
  }
  return { xh, ok: relError(p.x, xh) < 1e-4 }
}

type Progress = { key: string; z: number[][]; done: number }

/** The success-rate grid, filled a few trials at a time between frames so the page stays responsive. */
function usePhaseGrid(method: Method, trials: number, seed: number): Progress {
  const key = `${method}|${trials}|${seed}`
  const [grid, setGrid] = useState<Progress>({ key: '', z: [], done: 0 })
  useEffect(() => {
    let cancelled = false
    const wins = RHOS.map(() => MS.map(() => 0))
    const total = G * G * trials
    let unit = 0
    const tick = () => {
      if (cancelled) return
      const start = performance.now()
      // Column by column in δ; every trial of a cell before the next cell.
      while (unit < total && performance.now() - start < 12) {
        const cell = Math.floor(unit / trials)
        const t = unit % trials
        const a = Math.floor(cell / G)
        const b = cell % G
        const m = MS[a]
        if (solve(method, trial(m, sparsity(m, RHOS[b]), seed, a, b, t)).ok) wins[b][a] += 1
        unit += 1
      }
      const filled = Math.floor(unit / trials)
      const z = RHOS.map((_, b) => MS.map((_, a) => (a * G + b < filled ? wins[b][a] / trials : NaN)))
      setGrid({ key, z, done: unit / total })
      if (unit < total) setTimeout(tick, 0)
    }
    const id = setTimeout(tick, 0)
    return () => {
      cancelled = true
      clearTimeout(id)
    }
  }, [key, method, trials, seed])
  return grid.key === key ? grid : { key, z: RHOS.map(() => MS.map(() => NaN)), done: 0 }
}

export function PhaseTransition() {
  const state = useFigureState({
    method: choice(
      [
        { value: 'omp', label: 'orthogonal matching pursuit' },
        { value: 'bp', label: 'basis pursuit (slower to fill)' },
      ],
      'omp',
      { label: 'recovery' },
    ),
    trials: int(8, { min: 1, max: 40, label: 'trials per cell', suggestions: [4, 8, 16] }),
    seed: int(1, { min: 1, max: 9999, label: 'seed' }),
    delta: slider(DELTAS[0], DELTAS[G - 1], 0.5, { step: 1 / N, onChart: true }),
    rho: slider(RHOS[0], 1, 0.2, { step: 0.01, onChart: true }),
  })
  const { method, trials, seed, delta, rho } = state
  const grid = usePhaseGrid(method, trials, seed)

  // The cell nearest the handle, and its first trial.
  const a = Math.min(G - 1, Math.max(0, Math.round(delta * (N / 4)) - 1))
  const b = Math.min(G - 1, Math.max(0, Math.round(rho * G) - 1))
  const m = MS[a]
  const s = sparsity(m, RHOS[b])
  const one = useComputed(() => {
    const p = trial(m, s, seed, a, b, 0)
    return { p, ...solve(method, p) }
  }, [m, s, seed, a, b, method]).value
  const truthJ = useMemo(() => one.p.support.slice().sort((u, v) => u - v), [one])
  const rate = grid.z[b]?.[a]
  const top = Math.max(...one.p.x.map(Math.abs), ...one.xh.map(Math.abs)) * 1.1

  const dx = useAxis({ label: 'δ = m / n (measurements per unknown)', range: [0, 1] })
  const dy = useAxis({ label: 'ρ = s / m (non-zeros per measurement)', range: [0, 1.03] })
  const jx = useAxis({ label: 'entry j', range: [-1, N] })
  const jy = useAxis({ label: 'value', range: [-top, top], key: `${m}|${s}|${seed}` })

  return (
    <Figure
      title="Where recovery switches on: the phase transition"
      state={state}
      defaultSize="L"
      caption="Left: the share of trials in which an s-sparse vector of length n = 64 is recovered exactly from m Gaussian measurements, for each δ = m/n and ρ = s/m (colour bar: 0 never, 1 always). The grid fills column by column. The solid contour marks a 50% success rate; the dashed line is m = 2s, above which some s-sparse vectors share their measurements with another, so no method recovers every one; a typical one can still be recovered. Drag the point to pick a cell; right: its first trial, the true entries (dots) and the recovered ones (bars). Switch to basis pursuit: its boundary lies higher, most clearly at large δ, where it crosses the dashed line."
      readouts={
        <>
          <Readout label="cell" value={`m = ${m}, s = ${s}`} />
          <Readout
            label="success rate in the cell"
            value={rate === undefined || Number.isNaN(rate) ? '…' : formatNumber(rate)}
          />
          <Readout label="this trial" value={one.ok ? 'recovered' : 'failed'} />
          <Readout label="filled" value={`${Math.round(grid.done * 100)}%`} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
        <Plot x={dx} y={dy} height={320}>
          <Raster
            x={DELTAS}
            y={RHOS}
            z={grid.z}
            range={[0, 1]}
            boundary={0.5}
            valueLabel="success rate"
            scaleTicks={[0, 0.5, 1]}
          />
          <Curve name="m = 2s" x={[0, 1]} y={[0.5, 0.5]} dashed emphasis silent />
          <Handle {...state.handle(['delta', 'rho'], { label: 'cell' })} />
        </Plot>
        <Plot x={jx} y={jy} height={320}>
          <Bars name="recovered" x={INDEX} y={one.xh} slot={0} width={0.8} />
          <Points name="true" x={truthJ} y={truthJ.map((j) => one.p.x[j])} emphasis size={7} />
        </Plot>
      </div>
    </Figure>
  )
}
