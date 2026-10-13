import { useMemo, useState } from 'react'
import {
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
  useAxis,
  useFigureState,
} from 'aifn-render'
import { treeParzenSteps, type TreeParzenState } from 'aifn-methods/optimisation/tree-parzen'
import { forrester } from 'aifn-methods/data/objectives'
import { real, space } from 'aifn-compute/foundation/space'
import { stream } from 'aifn-compute/foundation/random'
import { linspace, tensor, toFlat } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { FixedHeight } from '../_shared/FixedHeight'

const F = forrester()
const SPACE = space({ lambda: real(0, 1) })
const objective = ({ lambda }: { lambda: number }) => F.value(tensor([lambda]))
const XS = toFlat(linspace(0, 1, 201))
const TRUTH = XS.map((x) => F.value(tensor([x])))
/** Random evaluations before the densities take over. */
const INIT = 10

/** TPE on the Forrester function: the split into good and bad, the two Parzen densities, and their ratio. */
export function TpeWalkthrough() {
  const state = useFigureState({
    gamma: float(0.15, { label: 'good fraction γ', gt: 0, le: 1, suggestions: [0.1, 0.15, 0.25, 0.5] }),
    candidates: int(24, { label: 'candidates drawn from ℓ', min: 1, max: 500, suggestions: [1, 4, 24, 100] }),
    budget: int(30, { label: 'evaluations after the random start', min: 1, max: 100, suggestions: [20, 30, 60] }),
    seed: int(3, { label: 'seed', min: 0, max: 99, suggestions: [1, 2, 3, 4] }),
  })

  const run = useMemo(() => {
    const alg = treeParzenSteps(objective, SPACE, {
      gamma: state.gamma,
      candidates: state.candidates,
      init: INIT,
    })
    const tr = trace(alg, {}, state.budget, { stream: stream(state.seed), keep: 'all' })
    return tr.steps as readonly TreeParzenState[]
  }, [state.gamma, state.candidates, state.budget, state.seed])

  const key = `${state.gamma}|${state.candidates}|${state.budget}|${state.seed}`
  const [cursor, setCursor] = useState({ key, step: 0 })
  const at = Math.min(cursor.key === key ? cursor.step : 0, run.length - 1)
  const s = run[at]

  const view = useMemo(() => {
    const lam = s.observations.map((o) => Number(o.lambda))
    const loss = toFlat(s.losses)
    const isGood = new Set(s.good)
    const pick = (keep: boolean) => {
      const idx = lam.map((_, i) => i).filter((i) => isGood.has(i) === keep)
      return { x: idx.map((i) => lam[i]), y: idx.map((i) => loss[i]) }
    }
    const m = s.marginals[0]
    const grid = toFlat(m.at)
    const cand = toFlat(s.candidates)
    const ratioAt = toFlat(s.scores).map((v) => Math.exp(v))
    const chosen = Number(s.next.lambda)
    const chosenIndex = cand.findIndex((u) => Math.abs(u - toFlat(s.nextU)[0]) < 1e-12)
    return {
      good: pick(true),
      bad: pick(false),
      grid,
      ell: toFlat(m.good),
      g: toFlat(m.bad),
      ratio: toFlat(m.ratio),
      cand,
      ratioAt,
      chosen,
      chosenRatio: chosenIndex >= 0 ? ratioAt[chosenIndex] : Math.max(...ratioAt),
    }
  }, [s])

  const lamAxis = useAxis({ label: 'hyperparameter λ', range: [0, 1] })
  const lossAxis = useAxis({ label: 'loss f(λ)', range: [-8, 17] })
  const dAxis = useAxis({ label: 'density', hold: 'union', key })
  const rAxis = useAxis({ label: 'ratio ℓ/g', log: true, hold: 'union', key })
  const v = view
  return (
    <Figure
      title="The tree-structured Parzen estimator, step by step"
      state={state}
      caption={`TPE minimising the Forrester function over λ in [0, 1], after ${INIT} random evaluations. Top: the objective (dashed), the evaluations split at the γ-quantile y* of their losses into a good group (the best ⌈γn⌉) and a bad group, and the next evaluation (vertical line). Middle: the Parzen densities ℓ of the good group and g of the bad group, each a mixture of truncated Gaussian kernels plus a broad prior component. Bottom: the ratio ℓ/g on a log scale, the candidates drawn from ℓ (dots at their ratio) and the one chosen, the candidate with the largest ratio. Step through with the player. A small γ makes ℓ sharp around the best few evaluations and the search exploits; γ = 0.5 spreads ℓ and the search explores. With one candidate per step TPE samples from ℓ; with hundreds it maximises the ratio almost exactly.`}
      controls={
        <Player
          value={at}
          onChange={(p) => setCursor({ key, step: p })}
          count={run.length}
          label="step"
          format={(p) => `${INIT + p} evaluations`}
        />
      }
      readouts={
        <>
          <Readout label="evaluations n" value={String(s.observations.length)} />
          <Readout label="good group ⌈γn⌉" value={String(s.good.length)} />
          <Readout label="split y*" value={formatNumber(s.threshold)} />
          <Readout label="best loss" value={formatNumber(s.best)} />
          <Readout label="at λ" value={formatNumber(Number(s.incumbent.lambda))} />
          <Readout label="next λ" value={formatNumber(v.chosen)} />
          <Readout label="true minimum" value={`${formatNumber(F.minimumValue)} at 0.757`} />
        </>
      }
    >
      <FixedHeight height={640}>
        <Plots rows={3} heights={[3, 2, 2]}>
          <Plot x={lamAxis} y={lossAxis} title="evaluations">
            <Curve name="objective f" x={XS} y={TRUTH} muted dashed />
            <Curve name="split y*" x={[0, 1]} y={[s.threshold, s.threshold]} slot={1} dashed width={1} />
            {v.bad.x.length > 0 && <Points name="bad group" x={v.bad.x} y={v.bad.y} slot={0} size={7} />}
            {v.good.x.length > 0 && <Points name="good group" x={v.good.x} y={v.good.y} slot={1} size={9} />}
            <Segments name="next λ" segments={[{ from: [v.chosen, -8], to: [v.chosen, 17] }]} emphasis dashed />
          </Plot>
          <Plot x={lamAxis} y={dAxis} title="Parzen densities">
            <Curve name="ℓ(λ), good" x={v.grid} y={v.ell} slot={1} />
            <Curve name="g(λ), bad" x={v.grid} y={v.g} slot={0} />
          </Plot>
          <Plot x={lamAxis} y={rAxis} title="ratio">
            <Curve name="ℓ(λ)/g(λ)" x={v.grid} y={v.ratio} slot={3} />
            <Points name="candidates from ℓ" x={v.cand} y={v.ratioAt} slot={3} size={6} />
            <Points name="chosen" x={[v.chosen]} y={[v.chosenRatio]} emphasis size={10} />
          </Plot>
        </Plots>
      </FixedHeight>
    </Figure>
  )
}
