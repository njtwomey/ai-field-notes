import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  Readout,
  StepControls,
  XYChart,
  formatNumber,
  type Handle,
  type XYSeries,
} from 'aifn-render'
import { linspace } from '@/lib/math'
import { iterate, linkage, mixture, rate, type EmProblem } from './problems'

const PROBLEMS = { linkage, mixture }
type Name = keyof typeof PROBLEMS
/** Iterations precomputed from the start; Run jumps to the last. */
const ITERATIONS = 40
const GRID_POINTS = 241
/** Previous bounds kept on screen, faintly, so the sequence toward the limit stays visible. */
const TRAIL = 6

/**
 * The EM bound picture. Half-steps alternate: an E-step draws the bound 𝓛(q_t, θ), which touches ℓ at θ_t; an M-step
 * moves to the bound's maximum θ_{t+1}. Half-step k shows θ_{⌊k/2⌋} and the bounds built at θ₀ … θ_{⌈k/2⌉−1}.
 */
export function EmBound() {
  const [name, setName] = useState<Name>('linkage')
  const problem: EmProblem = PROBLEMS[name]
  const [start, setStart] = useState<number>(linkage.start)
  const [k, setK] = useState(0)

  const thetas = useMemo(() => iterate(problem, start, ITERATIONS), [problem, start])
  // The limit from this start: run on until the iterates stop moving (a start exactly at a stationary point stays).
  const limit = useMemo(() => iterate(problem, start, 2000).at(-1)!, [problem, start])
  const convergenceRate = rate(problem, limit)

  const t = Math.floor(k / 2)
  const bounds = Math.ceil(k / 2)
  const theta = thetas[t]
  const current = bounds > 0 ? thetas[bounds - 1] : undefined

  const grid = useMemo(() => linspace(problem.range[0], problem.range[1], GRID_POINTS), [problem])
  const llGrid = useMemo(() => grid.map(problem.ll), [grid, problem])
  const top = Math.max(...llGrid)

  const series = useMemo((): XYSeries[] => {
    const out: XYSeries[] = []
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
  const trace = useMemo((): { ll: XYSeries[]; error: XYSeries[] } => {
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

  const handles: Handle[] = [
    {
      kind: 'x',
      at: start,
      label: `start ${problem.parameter}₀`,
      onDrag: (v) => {
        const [lo, hi] = problem.range
        setStart(Math.round(Math.min(hi, Math.max(lo, v)) * 100) / 100)
        setK(0)
      },
    },
  ]

  const gap = current !== undefined ? problem.gap(current, theta) : undefined
  const stage =
    k === 0
      ? 'start'
      : k % 2 === 1
        ? 'E-step: bound tight at the current value'
        : 'M-step: moved to the bound’s maximum'

  return (
    <Interactive
      title="The bound and its limit"
      caption="Step alternates the two halves of EM. The E-step draws the lower bound 𝓛(q, ·) that touches the log-likelihood at the current value; the M-step jumps to that bound's maximum, where the likelihood is at least as high. Faint lines are earlier bounds. Genetic linkage has one maximum. The symmetric mixture has two, at ±μ̂, and a stationary point at 0: drag the start to see the limit change sides, and to 0 to see EM stay put. The right-hand chart shows the error shrinking by a constant factor per step, the fraction of missing information."
      controls={
        <>
          <ParamChoice
            label="problem"
            value={name}
            onChange={(v) => {
              setName(v)
              setStart(PROBLEMS[v].start)
              setK(0)
            }}
            options={[
              { value: 'linkage', label: 'genetic linkage' },
              { value: 'mixture', label: 'symmetric mixture' },
            ]}
          />
          <StepControls
            onStep={() => setK((v) => Math.min(v + 1, 2 * ITERATIONS))}
            onRun={() => setK(2 * ITERATIONS)}
            onReset={() => setK(0)}
            done={k >= 2 * ITERATIONS}
          />
        </>
      }
      readout={
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
      <XYChart
        series={series}
        xLabel={problem.parameter}
        yLabel="log-likelihood"
        xRange={problem.range}
        yRange={[top - problem.span, top + 2]}
        handles={handles}
        height={340}
      />
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart series={trace.ll} xLabel="iteration" yLabel="ℓ" height={220} />
        <XYChart series={trace.error} xLabel="iteration" yLabel="log₁₀ error" height={220} />
      </div>
    </Interactive>
  )
}
