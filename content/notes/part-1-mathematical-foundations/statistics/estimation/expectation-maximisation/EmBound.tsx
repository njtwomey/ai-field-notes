import { useMemo, useState } from 'react'
import {
  choice,
  Figure,
  formatNumber,
  Handle,
  Player,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { iterate, linkage, mixture, rate, type EmProblem } from './problems'
import { linspace, toFlat } from 'aifn/foundation/tensor'

const PROBLEMS = { linkage, mixture }
type Name = keyof typeof PROBLEMS
/** Iterations precomputed from the start; the player walks through their half-steps. */
const ITERATIONS = 40
const GRID_POINTS = 241
/** Previous bounds kept on screen, faintly, so the sequence toward the limit stays visible. */
const TRAIL = 6

/**
 * The EM bound picture. Half-steps alternate: an E-step draws the bound 𝓛(q_t, θ), which touches ℓ at θ_t; an M-step
 * moves to the bound's maximum θ_{t+1}. Half-step k shows θ_{⌊k/2⌋} and the bounds built at θ₀ … θ_{⌈k/2⌉−1}.
 */
export function EmBound() {
  const state = useFigureState({
    problem: choice<Name>(
      [
        { value: 'linkage', label: 'genetic linkage' },
        { value: 'mixture', label: 'symmetric mixture' },
      ],
      'linkage',
      { label: 'problem' },
    ),
  })
  const name = state.problem
  const problem: EmProblem = PROBLEMS[name]
  // The dragged start belongs to its problem; another problem opens at its own start.
  const [picked, setPicked] = useState({ name, start: linkage.start })
  const start = picked.name === name ? picked.start : problem.start
  // The walk-through restarts at half-step 0 whenever the problem or the start changes.
  const walk = `${name}:${start}`
  const [position, setPosition] = useState({ walk, k: 0 })
  const k = position.walk === walk ? position.k : 0

  const thetas = useMemo(() => iterate(problem, start, ITERATIONS), [problem, start])
  // The limit from this start: run on until the iterates stop moving (a start exactly at a stationary point stays).
  const limit = useMemo(() => iterate(problem, start, 2000).at(-1)!, [problem, start])
  const convergenceRate = rate(problem, limit)

  const t = Math.floor(k / 2)
  const bounds = Math.ceil(k / 2)
  const theta = thetas[t]
  const current = bounds > 0 ? thetas[bounds - 1] : undefined

  const grid = useMemo(() => toFlat(linspace(problem.range[0], problem.range[1], GRID_POINTS)), [problem])
  const llGrid = useMemo(() => grid.map(problem.ll), [grid, problem])
  const top = Math.max(...llGrid)

  const series = useMemo((): SeriesSpec[] => {
    const out: SeriesSpec[] = []
    // Older bounds first, muted, under one legend entry; the current bound on top in colour.
    for (let b = Math.max(0, bounds - 1 - TRAIL); b < bounds - 1; b++) {
      const q = thetas[b]
      out.push({
        name: 'earlier bounds',
        type: 'line',
        x: grid,
        y: grid.map((g) => problem.ll(g) - problem.gap(q, g)),
        muted: true,
      })
    }
    out.push({ name: 'log-likelihood ℓ', type: 'line', x: grid, y: llGrid, slot: 0 })
    if (current !== undefined) {
      out.push({
        name: 'current bound 𝓛(q, ·)',
        type: 'line',
        x: grid,
        y: grid.map((g) => problem.ll(g) - problem.gap(current, g)),
        slot: 1,
      })
    }
    out.push({
      name: `current ${problem.parameter}`,
      type: 'scatter',
      x: [theta],
      y: [problem.ll(theta)],
      emphasis: true,
    })
    return out
  }, [bounds, thetas, grid, llGrid, problem, current, theta])

  const history = thetas.slice(0, t + 1)
  const trace = useMemo((): { ll: SeriesSpec[]; error: SeriesSpec[] } => {
    const iters = history.map((_, i) => i)
    const floor = 1e-16
    const err = history.map((v) => Math.log10(Math.max(Math.abs(v - limit), floor)))
    const e0 = Math.abs(history[0] - limit)
    return {
      ll: [{ name: `ℓ(${problem.parameter}ₜ)`, type: 'line', x: iters, y: history.map(problem.ll), slot: 0 }],
      error: [
        { name: 'observed', type: 'line', x: iters, y: err, slot: 0 },
        ...(e0 > floor && convergenceRate > 0
          ? [
              {
                name: `rate ${formatNumber(convergenceRate)} per step`,
                type: 'line' as const,
                x: iters,
                y: iters.map((i) => Math.log10(e0) + i * Math.log10(convergenceRate)),
                slot: 1,
                dashed: true,
              },
            ]
          : []),
      ],
    }
  }, [history, limit, problem, convergenceRate])

  const gap = current !== undefined ? problem.gap(current, theta) : undefined
  const stage =
    k === 0
      ? 'start'
      : k % 2 === 1
        ? 'E-step: bound tight at the current value'
        : 'M-step: moved to the bound’s maximum'

  const xAxis = useAxis({ label: problem.parameter, range: problem.range })
  const yAxis = useAxis({ label: 'log-likelihood', range: [top - problem.span, top + 2] })
  const xAxis2 = useAxis({ label: 'iteration', hold: 'union' })
  const yAxis2 = useAxis({ label: 'ℓ', hold: 'union' })
  const xAxis3 = useAxis({ label: 'iteration', hold: 'union' })
  const yAxis3 = useAxis({ label: 'log₁₀ error', hold: 'union' })
  return (
    <Figure
      title="The bound and its limit"
      state={state}
      caption="Step alternates the two halves of EM. The E-step draws the lower bound 𝓛(q, ·) that touches the log-likelihood at the current value; the M-step jumps to that bound's maximum, where the likelihood is at least as high. Faint lines are earlier bounds. Genetic linkage has one maximum. The symmetric mixture has two, at ±μ̂, and a stationary point at 0: drag the start to see the limit change sides, and to 0 to see EM stay put. The right-hand chart shows the error shrinking by a constant factor per step, the fraction of missing information."
      controls={
        <Player
          value={k}
          onChange={(v) => setPosition({ walk, k: v })}
          count={2 * ITERATIONS + 1}
          label="half-step"
          format={(v) => (v === 0 ? 'start' : `${Math.ceil(v / 2)}${v % 2 ? ' E' : ' M'}`)}
        />
      }
      readouts={
        <>
          <Readout label="stage" value={stage} />
          <Readout label="iteration t" value={t} />
          <Readout label={`${problem.parameter}ₜ`} value={formatNumber(theta)} />
          <Readout label={`ℓ(${problem.parameter}ₜ)`} value={formatNumber(problem.ll(theta))} />
          {gap !== undefined && (
            <Readout label="bound at the current value" value={formatNumber(problem.ll(theta) - gap)} />
          )}
          {gap !== undefined && <Readout label="gap KL(q ‖ posterior)" value={formatNumber(gap)} />}
          <Readout label={`limit ${problem.parameter}̂`} value={formatNumber(limit)} />
          <Readout label="rate |M′|" value={formatNumber(convergenceRate)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={340}>
        {seriesLayers(series)}
        <Handle
          kind="x"
          at={start}
          label={`start ${problem.parameter}₀`}
          onDrag={(v) => {
            const [lo, hi] = problem.range
            setPicked({ name, start: Math.round(Math.min(hi, Math.max(lo, v)) * 100) / 100 })
          }}
        />
      </Plot>
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis2} y={yAxis2} height={220}>
          {seriesLayers(trace.ll)}
        </Plot>
        <Plot x={xAxis3} y={yAxis3} height={220}>
          {seriesLayers(trace.error)}
        </Plot>
      </div>
    </Figure>
  )
}
