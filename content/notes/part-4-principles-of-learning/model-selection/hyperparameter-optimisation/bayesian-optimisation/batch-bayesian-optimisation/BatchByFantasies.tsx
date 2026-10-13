import { useMemo, useState } from 'react'
import {
  Area,
  choice,
  Curve,
  Figure,
  formatNumber,
  int,
  Player,
  Plot,
  Plots,
  Points,
  Readout,
  Segments,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { bayesianOptimisationSteps, expectedImprovement } from 'aifn-methods/optimisation/bayesian'
import { gpPosterior } from 'aifn-methods/learning/gaussian-processes'
import { forrester } from 'aifn-methods/data/objectives'
import { matern } from 'aifn-compute/learning/kernels'
import { stream } from 'aifn-compute/foundation/random'
import { sqrt, tensor, toFlat } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import { FixedHeight } from '../_shared/FixedHeight'

/** The objective: Forrester, Sóbester and Keane's function on [0, 1], minimum −6.0207 at 0.7572. */
const F = forrester()
const GRID = 201
const Y_RANGE = [-12, 20] as const

type Strategy = 'none' | 'believer' | 'liar-min' | 'liar-mean' | 'liar-max'
const STRATEGIES = [
  { value: 'none', label: 'no fantasy (EI q times)' },
  { value: 'believer', label: 'kriging believer (lie μₜ)' },
  { value: 'liar-min', label: 'constant liar, L = min y' },
  { value: 'liar-mean', label: 'constant liar, L = mean y' },
  { value: 'liar-max', label: 'constant liar, L = max y' },
] as const

type Slot = {
  mean: number[]
  upper: number[]
  lower: number[]
  ei: number[]
  next: number
  nextIndex: number
  /** The value the next point is pretended to return. */
  lie: number
  /** The incumbent the EI of this slot was computed against: the best of the real and pretended values. */
  incumbent: number
}

/**
 * A batch of q evaluations chosen by expected improvement with fantasised outcomes (Ginsbourger, Le Riche and Carraro,
 * 2010): the Gaussian process is fitted once to the real evaluations, then each chosen point is added with a pretended
 * value and the posterior is recomputed with the same hyperparameters before the next point is chosen.
 */
export function BatchByFantasies() {
  const state = useFigureState({
    strategy: choice<Strategy>(STRATEGIES, 'liar-min', { label: 'pending points are given' }),
    q: int(4, { label: 'batch size q', min: 1, max: 8, suggestions: [2, 4, 6, 8] }),
    init: int(5, { label: 'evaluations so far', min: 3, max: 12, suggestions: [4, 5, 8] }),
    seed: int(4, { label: 'seed (design)', min: 0, max: 99, suggestions: [3, 4, 5, 7] }),
  })
  const { strategy, q, init, seed } = state

  const batch = useMemo(() => {
    // The engine's loop fits the surrogate to the real evaluations: hyperparameters by marginal likelihood, on the unit
    // box and on targets standardised by `shift` and `scale`.
    const alg = bayesianOptimisationSteps(F.value, F, { init, grid: GRID })
    const s = run(alg, undefined, 0, { stream: stream(seed) })
    // The same posterior in the objective's units: the kernel variance and noise scaled back, the prior mean the shift.
    const kernel = matern(2.5, {
      lengthscale: Number(s.hyper.lengthscale),
      variance: Number(s.hyper.variance) * s.scale ** 2,
    })
    const options = { noiseVariance: s.noiseVariance * s.scale ** 2, mean: s.shift }
    const realX = Array.from(toFlat(s.x))
    const realY = Array.from(toFlat(s.y))
    const lies: Record<Exclude<Strategy, 'none' | 'believer'>, number> = {
      'liar-min': Math.min(...realY),
      'liar-mean': realY.reduce((a, b) => a + b, 0) / realY.length,
      'liar-max': Math.max(...realY),
    }
    const xs = realX.slice()
    const ys = realY.slice()
    const slots: Slot[] = []
    for (let j = 0; j < q; j++) {
      const p = gpPosterior(kernel, tensor(xs), tensor(ys), options).predict(alg.grid)
      const sd = sqrt(p.variance)
      const incumbent = Math.min(...ys)
      const ei = Array.from(toFlat(expectedImprovement(p.mean, sd, incumbent)))
      const nextIndex = ei.reduce((arg, v, i) => (v > ei[arg] ? i : arg), 0)
      const mean = Array.from(toFlat(p.mean))
      const sdv = toFlat(sd)
      const next = toFlat(alg.grid)[nextIndex]
      const lie = strategy === 'believer' || strategy === 'none' ? mean[nextIndex] : lies[strategy]
      slots.push({
        mean,
        upper: mean.map((m, i) => m + 2 * sdv[i]),
        lower: mean.map((m, i) => m - 2 * sdv[i]),
        ei,
        next,
        nextIndex,
        lie,
        incumbent,
      })
      if (strategy !== 'none') {
        xs.push(next)
        ys.push(lie)
      }
    }
    const grid = Array.from(toFlat(alg.grid))
    return { grid, truth: grid.map((x) => F.value(tensor([x]))), realX, realY, slots }
  }, [strategy, q, init, seed])

  const key = `${strategy}|${q}|${init}|${seed}`
  const [cursor, setCursor] = useState({ key, step: 0 })
  const step = Math.min(cursor.key === key ? cursor.step : 0, q - 1)
  const slot = batch.slots[step]
  const chosen = batch.slots.slice(0, step)

  const lam = useAxis({ label: 'hyperparameter λ', range: [0, 1] })
  const fAxis = useAxis({ label: 'f(λ)', range: Y_RANGE })
  const aAxis = useAxis({ label: 'EI', hold: 'union', key })
  return (
    <Figure
      title="Filling a batch by pretending pending points have returned"
      state={state}
      caption="Top: the Forrester objective (dashed), the evaluations so far (orange), the points already in the batch at the values they are pretended to return (green), and the Gaussian-process posterior given both, with its ±2 sd band. Bottom: expected improvement over 201 candidates; its maximum (ink) joins the batch next. Step through the batch with the player. With no fantasy the posterior never changes and all q points coincide. A constant liar with the lie L = min y keeps the batch near the incumbent; L = max y pushes it into unexplored regions. The kriging believer pretends each point returns its posterior mean, which leaves the posterior mean unchanged and shrinks the band, and can lower the incumbent for later points."
      controls={
        <Player
          value={step}
          onChange={(v) => setCursor({ key, step: v })}
          count={q}
          label="point"
          format={(v) => `point ${v + 1} of ${q}`}
        />
      }
      readouts={
        <>
          <Readout label="incumbent used" value={formatNumber(slot.incumbent)} />
          <Readout label={`point ${step + 1} at λ`} value={formatNumber(slot.next)} />
          <Readout label="its EI" value={formatNumber(slot.ei[slot.nextIndex])} />
          <Readout label="pretended value" value={strategy === 'none' ? 'none' : formatNumber(slot.lie)} />
          <Readout
            label="batch so far"
            value={batch.slots
              .slice(0, step + 1)
              .map((b) => formatNumber(b.next))
              .join(', ')}
          />
        </>
      }
    >
      <FixedHeight height={520}>
        <Plots rows={2} heights={[3, 2]}>
          <Plot x={lam} y={fAxis} title="posterior with fantasies">
            <Area name="μ ± 2σ" x={batch.grid} y={slot.upper} base={slot.lower} slot={0} opacity={0.18} line={false} />
            <Curve name="objective f" x={batch.grid} y={batch.truth} muted dashed />
            <Curve name="posterior mean μ" x={batch.grid} y={slot.mean} slot={0} />
            <Points name="evaluations" x={batch.realX} y={batch.realY} slot={1} size={8} />
            {chosen.length > 0 && strategy !== 'none' && (
              <Points
                name="pending, pretended value"
                x={chosen.map((b) => b.next)}
                y={chosen.map((b) => b.lie)}
                slot={2}
                size={9}
              />
            )}
            <Segments
              name="next point of the batch"
              segments={[{ from: [slot.next, Y_RANGE[0]], to: [slot.next, Y_RANGE[1]] }]}
              emphasis
              dashed
            />
          </Plot>
          <Plot x={lam} y={aAxis} title="expected improvement">
            <Area name="EI" x={batch.grid} y={slot.ei} slot={3} />
            <Points name="its maximum" x={[slot.next]} y={[slot.ei[slot.nextIndex]]} emphasis size={9} />
          </Plot>
        </Plots>
      </FixedHeight>
    </Figure>
  )
}
