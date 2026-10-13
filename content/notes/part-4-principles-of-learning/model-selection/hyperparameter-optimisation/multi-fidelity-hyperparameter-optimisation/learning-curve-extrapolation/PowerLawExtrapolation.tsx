import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { stream } from 'aifn-compute/foundation/random'
import { kendallTau } from 'aifn-compute/probability/stats'
import { toArray, toFlat } from 'aifn-compute/foundation/tensor'
import { learningCurves } from 'aifn-methods/data/synthetic'
import { fitPowerLaw } from 'aifn-methods/optimisation/multi-fidelity'

const R = 81
const N = 27
const BUDGETS = Array.from({ length: R }, (_, i) => i + 1)

/**
 * A power law a + b t^(−c) fitted to the first t₀ epochs of each learning curve and extrapolated to R: one curve in
 * detail, and every curve's prediction against its true final loss, beside the naive prediction "the loss at t₀".
 */
export function PowerLawExtrapolation() {
  const state = useFigureState({
    cutoff: int(9, { ge: 3, le: R, label: 'epochs observed t₀', onChart: true }),
    shape: choice(
      [
        { value: 'power', label: 'power laws' },
        { value: 'exponential', label: 'exponential decays' },
        { value: 'mixed', label: 'a mixture' },
      ],
      'power',
      { label: 'true curves' },
    ),
    crossing: slider(0, 1, 0.5, { step: 0.05, label: 'crossing κ' }),
    noise: float(0.01, { ge: 0, le: 0.05, suggestions: [0, 0.01, 0.02], label: 'noise sd' }),
    curve: int(0, { ge: 0, le: N - 1, label: 'curve shown' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  })
  const { cutoff, shape, crossing, noise, curve, seed } = state

  const family = useMemo(() => {
    const c = learningCurves(stream(seed), { n: N, budgets: BUDGETS, crossing, noise, shape })
    return {
      observed: toArray(c.values) as number[][],
      clean: toArray(c.clean) as number[][],
      final: Array.from(toFlat(c.final)),
      best: c.best,
    }
  }, [shape, crossing, noise, seed])

  const fits = useMemo(
    () => family.observed.map((row) => fitPowerLaw(BUDGETS.slice(0, cutoff), row.slice(0, cutoff))),
    [family, cutoff],
  )
  const predicted = fits.map((f) => f.at(R))
  const current = family.observed.map((row) => row[cutoff - 1])
  const fit = fits[curve]
  const mae = (xs: number[]) => xs.reduce((a, x, i) => a + Math.abs(x - family.final[i]), 0) / N
  const pickBy = (xs: number[]) => xs.indexOf(Math.min(...xs))
  const tail = BUDGETS.filter((t) => t >= cutoff)

  const tAxis = useAxis({ label: 'epoch t', log: true, range: [1, R] })
  const lossAxis = useAxis({ label: 'validation loss', range: [0, 1.05] })
  const trueAxis = useAxis({ label: 'true final loss at R = 81', range: [0, 0.6] })
  const predAxis = useAxis({ label: 'predicted final loss', range: [-0.1, 0.8] })
  return (
    <Figure
      title="Extrapolating learning curves with a power law"
      state={state}
      caption="Left: one learning curve of a seeded family whose true form is known. The dots are the losses seen in the first t₀ epochs; the dashed line is the power law a + b·t^(−c) fitted to them by least squares and extrapolated to 81 epochs; the thin grey line is the true curve. Drag the vertical line to change t₀. Right: every curve's predicted final loss against its true final loss. Orange: the power-law extrapolation. Blue: the naive prediction, the loss already reached at t₀. Points on the diagonal are perfect. With few epochs the noise dominates the fit; with power-law curves the extrapolation then becomes accurate quickly. With exponential curves the model is wrong: individual predictions can be far off, although the fits may still rank the curves better than the naive prediction."
      readouts={
        <>
          <Readout
            label="fit: a, b, c"
            value={`${formatNumber(fit.a)}, ${formatNumber(fit.b)}, ${formatNumber(fit.c)}`}
          />
          <Readout
            label="this curve at R: predicted / true"
            value={`${formatNumber(predicted[curve])} / ${formatNumber(family.final[curve])}`}
          />
          <Readout
            label="mean |error|: power law / loss at t₀"
            value={`${formatNumber(mae(predicted))} / ${formatNumber(mae(current))}`}
          />
          <Readout
            label="Kendall τ with the truth: power law / loss at t₀"
            value={`${formatNumber(kendallTau(predicted, family.final))} / ${formatNumber(kendallTau(current, family.final))}`}
          />
          <Readout
            label="true final of the curve each picks / best"
            value={`${formatNumber(family.final[pickBy(predicted)])}, ${formatNumber(family.final[pickBy(current)])} / ${formatNumber(family.final[family.best])}`}
          />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={tAxis} y={lossAxis} height={340}>
          <Curve name="true curve" x={BUDGETS} y={family.clean[curve]} muted thin />
          <Points
            name="observed"
            x={BUDGETS.slice(0, cutoff)}
            y={family.observed[curve].slice(0, cutoff)}
            slot={0}
            size={5}
          />
          <Curve name="power-law fit" x={BUDGETS} y={BUDGETS.map((t) => fit.at(t))} slot={1} dashed />
          <Curve name="extrapolation" x={tail} y={tail.map((t) => fit.at(t))} slot={1} />
          <Handle
            kind="x"
            at={cutoff}
            label="t₀"
            onDrag={(x) => state.set('cutoff', Math.max(3, Math.min(R, Math.round(x))))}
          />
        </Plot>
        <Plot x={trueAxis} y={predAxis} height={340}>
          <Curve name="perfect prediction" x={[0, 0.6]} y={[0, 0.6]} muted dashed />
          <Points name="loss at t₀" x={family.final} y={current} slot={0} thin />
          <Points name="power-law prediction" x={family.final} y={predicted} slot={1} />
          <Points name="curve shown" x={[family.final[curve]]} y={[predicted[curve]]} emphasis size={9} />
        </Plot>
      </div>
    </Figure>
  )
}
