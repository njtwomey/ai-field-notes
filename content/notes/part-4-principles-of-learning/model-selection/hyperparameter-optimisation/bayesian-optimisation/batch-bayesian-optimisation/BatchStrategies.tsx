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
import { bayesianOptimisationSteps, type BatchStrategy } from 'aifn-methods/optimisation/bayesian'
import { forrester } from 'aifn-methods/data/objectives'
import { stream } from 'aifn-compute/foundation/random'
import { tensor, toArray, toFlat } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import { FixedHeight } from '../_shared/FixedHeight'

/** The objective: Forrester, Sóbester and Keane's function on [0, 1], minimum −6.0207 at 0.7572. */
const F = forrester()
const GRID = 201
const Y_RANGE = [-12, 20] as const

type Strategy = 'none' | BatchStrategy
const STRATEGIES = [
  { value: 'none', label: 'none (EI q times)' },
  { value: 'kriging-believer', label: 'kriging believer (lie μₜ)' },
  { value: 'constant-liar-min', label: 'constant liar, L = min y' },
  { value: 'constant-liar-mean', label: 'constant liar, L = mean y' },
  { value: 'constant-liar-max', label: 'constant liar, L = max y' },
  { value: 'local-penalisation', label: 'local penalisation' },
  { value: 'thompson', label: 'Thompson sampling, q draws' },
] as const
const SCORE_LABEL: Record<Strategy, string> = {
  none: 'EI',
  'kriging-believer': 'EI after the lies',
  'constant-liar-min': 'EI after the lies',
  'constant-liar-mean': 'EI after the lies',
  'constant-liar-max': 'EI after the lies',
  'local-penalisation': 'penalised EI',
  thompson: 'negated posterior draw',
}

/**
 * One batch of q evaluations chosen by the engine's batch loop (Ginsbourger, Le Riche and Carraro, 2010; González et
 * al., 2016): the Gaussian process is fitted to the real evaluations, then each point of the batch is chosen by the
 * score left after the points before it, which a strategy changes by a pretended value, a penaliser or a fresh draw.
 */
export function BatchStrategies() {
  const state = useFigureState({
    strategy: choice<Strategy>(STRATEGIES, 'constant-liar-min', { label: 'pending points are handled by' }),
    q: int(4, { label: 'batch size q', min: 1, max: 8, suggestions: [2, 4, 6, 8] }),
    init: int(5, { label: 'evaluations so far', min: 3, max: 12, suggestions: [4, 5, 8] }),
    seed: int(4, { label: 'seed (design)', min: 0, max: 99, suggestions: [3, 4, 5, 7] }),
  })
  const { strategy, q, init, seed } = state

  const batch = useMemo(() => {
    const alg = bayesianOptimisationSteps(F.value, F, {
      init,
      grid: GRID,
      ...(strategy === 'none' ? {} : { batch: { size: q, strategy } }),
    })
    const s = run(alg, undefined, 0, { stream: stream(seed) })
    const grid = Array.from(toFlat(alg.grid))
    const mean = Array.from(toFlat(s.mean))
    const sd = toFlat(s.sd)
    // Without a strategy every slot sees the same acquisition, so all q points are its maximiser.
    const scores = s.batchAcquisition
      ? (toArray(s.batchAcquisition) as number[][])
      : Array.from({ length: q }, () => Array.from(toFlat(s.acquisition)))
    const indices = s.batchIndices ?? Array.from({ length: q }, () => s.nextIndex)
    const fantasies = s.fantasies ? Array.from(toFlat(s.fantasies)) : null
    return {
      grid,
      truth: grid.map((x) => F.value(tensor([x]))),
      mean,
      upper: mean.map((m, i) => m + 2 * sd[i]),
      lower: mean.map((m, i) => m - 2 * sd[i]),
      x: Array.from(toFlat(s.x)),
      y: Array.from(toFlat(s.y)),
      best: s.best,
      scores,
      indices,
      // A pending point is drawn at its pretended value, or at the posterior mean when it has none.
      shown: indices.map((k, j) => (fantasies ? fantasies[j] : mean[k])),
    }
  }, [strategy, q, init, seed])

  const key = `${strategy}|${q}|${init}|${seed}`
  const [cursor, setCursor] = useState({ key, step: 0 })
  const step = Math.min(cursor.key === key ? cursor.step : 0, q - 1)
  const score = batch.scores[step]
  const index = batch.indices[step]
  const next = batch.grid[index]
  const pending = batch.indices.slice(0, step)
  const draw = strategy === 'thompson' ? score.map((v) => -v) : null
  const label = SCORE_LABEL[strategy]
  const floor = Math.min(0, ...score)

  const lam = useAxis({ label: 'hyperparameter λ', range: [0, 1] })
  const fAxis = useAxis({ label: 'f(λ)', range: Y_RANGE })
  const aAxis = useAxis({ label, hold: 'union', key })
  return (
    <Figure
      title="Filling a batch: lies, penalties and draws"
      state={state}
      caption="Top: the Forrester objective (dashed), the evaluations so far (orange), the Gaussian-process posterior fitted to them with its ±2 sd band, and the points already in the batch (green), drawn at the value a liar pretends they returned or, without a lie, at the posterior mean. Bottom: the score that chooses the next point, after the points before it, with the first point's score (grey) for comparison; its maximum (ink) joins the batch. Step through the batch with the player. With no strategy all q points coincide. A constant liar with L = min y keeps the batch near the incumbent; L = max y pushes it into unexplored regions. Local penalisation multiplies EI by a factor that is near 0 next to each pending point and leaves the posterior alone. Thompson sampling minimises a fresh posterior draw (green curve) for each point."
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
          <Readout label="incumbent" value={formatNumber(batch.best)} />
          <Readout label={`point ${step + 1} at λ`} value={formatNumber(next)} />
          <Readout label={`its ${label}`} value={formatNumber(score[index])} />
          {strategy.startsWith('constant-liar') || strategy === 'kriging-believer' ? (
            <Readout label="pretended value" value={formatNumber(batch.shown[step])} />
          ) : null}
          <Readout
            label="batch so far"
            value={batch.indices
              .slice(0, step + 1)
              .map((k) => formatNumber(batch.grid[k]))
              .join(', ')}
          />
        </>
      }
    >
      <FixedHeight height={520}>
        <Plots rows={2} heights={[3, 2]}>
          <Plot x={lam} y={fAxis} title="posterior and pending points">
            <Area
              name="μ ± 2σ"
              x={batch.grid}
              y={batch.upper}
              base={batch.lower}
              slot={0}
              opacity={0.18}
              line={false}
            />
            <Curve name="objective f" x={batch.grid} y={batch.truth} muted dashed />
            <Curve name="posterior mean μ" x={batch.grid} y={batch.mean} slot={0} />
            {draw && <Curve name="posterior draw" x={batch.grid} y={draw} slot={2} width={1.5} />}
            <Points name="evaluations" x={batch.x} y={batch.y} slot={1} size={8} />
            {pending.length > 0 && strategy !== 'none' && (
              <Points
                name="pending points"
                x={pending.map((k) => batch.grid[k])}
                y={batch.shown.slice(0, step)}
                slot={2}
                size={9}
              />
            )}
            <Segments
              name="next point of the batch"
              segments={[{ from: [next, Y_RANGE[0]], to: [next, Y_RANGE[1]] }]}
              emphasis
              dashed
            />
          </Plot>
          <Plot x={lam} y={aAxis} title={label}>
            {step > 0 && <Curve name="score of point 1" x={batch.grid} y={batch.scores[0]} muted />}
            <Area name={label} x={batch.grid} y={score} base={floor} slot={3} />
            <Points name="its maximum" x={[next]} y={[score[index]]} emphasis size={9} />
          </Plot>
        </Plots>
      </FixedHeight>
    </Figure>
  )
}
