import { useMemo, useState, type ReactNode } from 'react'
import {
  Area,
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Player,
  Plot,
  Plots,
  Points,
  Readout,
  Segments,
  setting,
  useAxis,
  useFigureState,
} from 'aifn-render'
import {
  bayesianOptimisationSteps,
  ucbBeta,
  type AcquisitionName,
  type BayesianOptimisationState,
} from 'aifn-methods/optimisation/bayesian'
import { forrester } from 'aifn-methods/data/objectives'
import { stream } from 'aifn-compute/foundation/random'
import { tensor, toFlat } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { FixedHeight } from './FixedHeight'

/** The objective: Forrester, Sóbester and Keane's function on [0, 1], minimum −6.0207 at 0.7572. */
const F = forrester()
const F_MIN = F.minimumValue
/** Evaluations in the initial design (a Latin hypercube drawn from the run's stream). */
export const INIT = 3
/** Candidates the acquisition is maximised over: a regular grid of 201 points on [0, 1]. */
export const GRID = 201
/** The vertical range of the surrogate panel, wide enough for the ±2 sd band. */
const Y_RANGE = [-12, 20] as const

type RunSpec = {
  acquisition: AcquisitionName
  seed: number
  budget: number
  xi?: number
  beta?: number | 'srinivas'
  maxValues?: number
}

type Run = { grid: number[]; truth: number[]; states: readonly BayesianOptimisationState[] }

/** One Bayesian optimisation run on the Forrester function, every state kept so a Player can scrub it. */
function useBoRun(spec: RunSpec): Run {
  const { acquisition, seed, budget, xi, beta, maxValues } = spec
  return useMemo(() => {
    const alg = bayesianOptimisationSteps(F.value, F, {
      acquisition,
      init: INIT,
      grid: GRID,
      xi: xi ?? 0,
      beta: beta ?? 4,
      maxValues: maxValues ?? 10,
    })
    const tr = trace(alg, undefined, budget, { stream: stream(seed), keep: 'all' })
    const grid = toFlat(alg.grid)
    const truth = grid.map((x) => F.value(tensor([x])))
    return { grid, truth, states: tr.steps }
  }, [acquisition, seed, budget, xi, beta, maxValues])
}

/** The Player's position, reset to 0 whenever the run changes. */
function useCursor(key: string) {
  const [cursor, setCursor] = useState({ key, step: 0 })
  const step = cursor.key === key ? cursor.step : 0
  return [step, (v: number) => setCursor({ key, step: v })] as const
}

type ViewProps = {
  title: string
  caption: ReactNode
  state: Parameters<typeof Figure>[0]['state']
  run: Run
  runKey: string
  /** What the lowest panel shows: the best value so far, or the cumulative regret. */
  bottom: 'best' | 'regret'
  acquisitionLabel: string
  extraReadouts?: (s: BayesianOptimisationState) => ReactNode
}

/** The three panels: surrogate and evaluations, the acquisition, and progress by evaluation. */
function BoRunView({ title, caption, state, run, runKey, bottom, acquisitionLabel, extraReadouts }: ViewProps) {
  const [step, setStep] = useCursor(runKey)
  const at = Math.min(step, run.states.length - 1)
  const s = run.states[at]

  const panels = useMemo(() => {
    const mean = toFlat(s.mean)
    const sd = toFlat(s.sd)
    const upper = mean.map((m, i) => m + 2 * sd[i])
    const lower = mean.map((m, i) => m - 2 * sd[i])
    const acq = toFlat(s.acquisition)
    const xs = toFlat(s.x)
    const ys = toFlat(s.y)
    const nx = toFlat(s.next)[0]
    const draw = s.draw ? toFlat(s.draw) : null
    const minima = s.maxValues ? toFlat(s.maxValues) : []
    const lastIndex = xs.length - 1
    const acqFloor = Math.min(0, ...acq)
    return { mean, upper, lower, acq, acqFloor, xs, ys, nx, draw, minima, lastIndex }
  }, [s])

  const progress = useMemo(() => {
    const ys = toFlat(run.states[run.states.length - 1].y)
    const n = INIT + at
    const evals = Array.from({ length: n }, (_, i) => i + 1)
    let best = Infinity
    const bestSoFar = ys.slice(0, n).map((v) => (best = Math.min(best, v)))
    let total = 0
    const cumulative = ys.slice(0, n).map((v, i) => (i < INIT ? 0 : (total += v - F_MIN)))
    return { evals, bestSoFar, cumulative, total }
  }, [run, at])

  const lam = useAxis({ label: 'hyperparameter λ', range: [0, 1] })
  const fAxis = useAxis({ label: 'f(λ)', range: Y_RANGE })
  const aAxis = useAxis({ label: acquisitionLabel, hold: 'union', key: runKey })
  const nAxis = useAxis({ label: 'evaluations', range: [1, INIT + run.states.length - 1], integer: true })
  const pAxis = useAxis({
    label: bottom === 'best' ? 'best value so far' : 'cumulative regret',
    hold: 'union',
    key: runKey,
  })

  const p = panels
  const previous = { x: p.xs.slice(0, p.lastIndex), y: p.ys.slice(0, p.lastIndex) }
  return (
    <Figure
      title={title}
      state={state}
      caption={caption}
      controls={
        <Player
          value={at}
          onChange={setStep}
          count={run.states.length}
          label="step"
          format={(v) => `${INIT + v} evaluations`}
        />
      }
      readouts={
        <>
          <Readout label="evaluations t" value={String(s.evaluations)} />
          <Readout label="incumbent f*ₜ" value={formatNumber(s.best)} />
          <Readout label="at λ" value={formatNumber(toFlat(s.bestX)[0])} />
          <Readout label="next λ" value={formatNumber(p.nx)} />
          <Readout label="true minimum" value={`${formatNumber(F_MIN)} at 0.757`} />
          <Readout label="cumulative regret" value={formatNumber(progress.total)} />
          <Readout label="length-scale (unit box)" value={formatNumber(Number(s.hyper.lengthscale))} />
          {extraReadouts?.(s)}
        </>
      }
    >
      <FixedHeight height={660}>
        <Plots rows={3} heights={[3, 2, 2]}>
          <Plot x={lam} y={fAxis} title="surrogate">
            <Area name="μₜ ± 2σₜ" x={run.grid} y={p.upper} base={p.lower} slot={0} opacity={0.18} line={false} />
            <Curve name="objective f" x={run.grid} y={run.truth} muted dashed />
            <Curve name="posterior mean μₜ" x={run.grid} y={p.mean} slot={0} />
            {p.draw && <Curve name="posterior draw" x={run.grid} y={p.draw} slot={2} width={1.5} />}
            {p.minima.length > 0 && (
              <Segments
                name="samples of min f"
                segments={p.minima.map((v) => ({ from: [0, v] as const, to: [1, v] as const }))}
                slot={2}
                dashed
              />
            )}
            {previous.x.length > 0 && <Points name="evaluations" x={previous.x} y={previous.y} slot={1} size={7} />}
            <Points name="latest evaluation" x={[p.xs[p.lastIndex]]} y={[p.ys[p.lastIndex]]} slot={1} size={11} />
            <Segments name="next λ" segments={[{ from: [p.nx, Y_RANGE[0]], to: [p.nx, Y_RANGE[1]] }]} emphasis dashed />
          </Plot>
          <Plot x={lam} y={aAxis} title="acquisition">
            <Area name={acquisitionLabel} x={run.grid} y={p.acq} base={p.acqFloor} slot={3} />
            <Points name="its maximum" x={[p.nx]} y={[p.acq[s.nextIndex]]} emphasis size={9} />
          </Plot>
          <Plot x={nAxis} y={pAxis} title={bottom === 'best' ? 'progress' : 'regret'}>
            {bottom === 'best' ? (
              <>
                <Curve name="true minimum" x={[1, INIT + run.states.length - 1]} y={[F_MIN, F_MIN]} muted dashed />
                <Curve name="best value so far" x={progress.evals} y={progress.bestSoFar} slot={1} showPoints />
              </>
            ) : (
              <Curve
                name="cumulative regret Σ (yᵢ − min f)"
                x={progress.evals}
                y={progress.cumulative}
                slot={1}
                showPoints
              />
            )}
          </Plot>
        </Plots>
      </FixedHeight>
    </Figure>
  )
}

const BUDGET = {
  label: 'evaluations after the initial design',
  min: 1,
  max: 40,
  suggestions: [10, 15, 25, 40],
} as const
const SEED = { label: 'seed (initial design)', min: 0, max: 99, suggestions: [1, 2, 3, 4] } as const

const ACQUISITIONS = [
  { value: 'ei', label: 'expected improvement' },
  { value: 'pi', label: 'probability of improvement' },
  { value: 'ucb', label: 'confidence bound, β = 4' },
  { value: 'thompson', label: 'Thompson sampling' },
  { value: 'mes', label: 'max-value entropy search' },
] as const
const ACQUISITION_LABEL: Record<AcquisitionName, string> = {
  ei: 'EI',
  pi: 'PI',
  ucb: 'negated lower bound',
  thompson: 'negated draw',
  mes: 'MES',
}

function CompareWalkthrough() {
  const state = useFigureState({
    acquisition: choice<AcquisitionName>(ACQUISITIONS, 'pi', { label: 'acquisition' }),
    budget: int(15, BUDGET),
    seed: int(1, SEED),
  })
  const run = useBoRun({ acquisition: state.acquisition, seed: state.seed, budget: state.budget })
  return (
    <BoRunView
      title="Bayesian optimisation, one evaluation at a time"
      state={state}
      run={run}
      runKey={`${state.acquisition}|${state.seed}|${state.budget}`}
      bottom="best"
      acquisitionLabel={ACQUISITION_LABEL[state.acquisition]}
      caption="Top: the Forrester objective (dashed), the Gaussian-process posterior mean and its ±2 sd band, the evaluations so far and the next one (vertical line). Middle: the acquisition over the grid of 201 candidates; its maximum is the next evaluation. Bottom: the best value observed against the number of evaluations. Step through with the player, then switch the acquisition with the same seed: probability of improvement creeps a grid step at a time, the confidence bound and Thompson sampling visit the ends of the interval, and expected improvement balances the two. Change the seed to start from another initial design."
    />
  )
}

function EiWalkthrough() {
  const state = useFigureState({
    xi: float(0, { label: 'margin ξ (units of f)', ge: 0, le: 5, suggestions: [0, 0.01, 0.1, 0.5, 1, 2] }),
    budget: int(15, BUDGET),
    seed: int(0, SEED),
  })
  const run = useBoRun({ acquisition: 'ei', seed: state.seed, budget: state.budget, xi: state.xi })
  return (
    <BoRunView
      title="Expected improvement with a margin"
      state={state}
      run={run}
      runKey={`ei|${state.xi}|${state.seed}|${state.budget}`}
      bottom="best"
      acquisitionLabel="EI"
      caption="Expected improvement on the Forrester function, with the margin ξ subtracted from the incumbent before the improvement is computed. Top: objective (dashed), posterior mean with ±2 sd band, evaluations and the next one. Middle: EI over the 201 candidates. Bottom: best value so far. With seed 0 and ξ = 0, evaluations 4 to 8 stay around the shallow minimum near λ = 0.15 before the search moves on and finds the global one. With ξ = 0.5 it leaves the shallow well after one evaluation, but it refines the global minimum less and ends further from it. The margin trades early escape against late precision."
    />
  )
}

function UcbWalkthrough() {
  const state = useFigureState({
    schedule: setting(false, { label: 'Srinivas schedule βₜ (δ = 0.1)' }),
    beta: float(4, {
      label: 'exploration weight β',
      ge: 0,
      le: 100,
      suggestions: [0, 1, 4, 16, 36],
      when: (v) => !v.schedule,
    }),
    budget: int(20, BUDGET),
    seed: int(4, SEED),
  })
  const beta = state.schedule ? ('srinivas' as const) : state.beta
  const run = useBoRun({ acquisition: 'ucb', seed: state.seed, budget: state.budget, beta })
  return (
    <BoRunView
      title="The confidence bound and its exploration weight"
      state={state}
      run={run}
      runKey={`ucb|${String(beta)}|${state.seed}|${state.budget}`}
      bottom="regret"
      acquisitionLabel="−(μₜ − √β σₜ)"
      extraReadouts={(s) => (
        <Readout
          label="β used"
          value={formatNumber(state.schedule ? ucbBeta(s.t + 1, { candidates: GRID }) : state.beta)}
        />
      )}
      caption="The confidence-bound acquisition −(μₜ − √β σₜ) on the Forrester function. Top: objective (dashed), posterior mean with ±2 sd band, evaluations and the next one. Middle: the acquisition. Bottom: the cumulative regret, the sum over evaluations after the initial design of how far each value was above the true minimum. With seed 4, β = 0 and β = 1 settle in the shallow minimum near λ = 0.14 and the regret grows linearly; β = 4 escapes. The Srinivas schedule (β from 16 to 29 here) never stops exploring the ends, so its regret keeps growing, but more and more slowly."
    />
  )
}

function MesWalkthrough() {
  const state = useFigureState({
    samples: int(10, { label: 'samples K of min f', min: 1, max: 100, suggestions: [1, 3, 10, 30] }),
    budget: int(15, BUDGET),
    seed: int(1, SEED),
  })
  const run = useBoRun({ acquisition: 'mes', seed: state.seed, budget: state.budget, maxValues: state.samples })
  return (
    <BoRunView
      title="Max-value entropy search, one evaluation at a time"
      state={state}
      run={run}
      runKey={`mes|${state.samples}|${state.seed}|${state.budget}`}
      bottom="best"
      acquisitionLabel="MES (nats)"
      caption="Max-value entropy search on the Forrester function. Top: objective (dashed), posterior mean with ±2 sd band, evaluations, the next one, and the K samples of the minimum value min f drawn from the posterior by the Gumbel approximation (horizontal lines). Middle: the acquisition, the expected information an evaluation gives about min f, in nats. It is largest where the predictive distribution reaches down to the sampled minima. Bottom: best value so far. Watch the samples tighten under the incumbent as the search proceeds; with K = 1 the rule is a probability of improvement with a random target."
    />
  )
}

/** A Bayesian optimisation run on the Forrester function, in one of four teaching modes. */
export function BoWalkthrough({ mode }: { mode: 'compare' | 'ei' | 'ucb' | 'mes' }) {
  if (mode === 'ei') return <EiWalkthrough />
  if (mode === 'ucb') return <UcbWalkthrough />
  if (mode === 'mes') return <MesWalkthrough />
  return <CompareWalkthrough />
}
