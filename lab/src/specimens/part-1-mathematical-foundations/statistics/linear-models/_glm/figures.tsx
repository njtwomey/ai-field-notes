import { useMemo, useState } from 'react'
import {
  binomialFamily,
  gammaFamily,
  gaussianFamily,
  inverseGaussianFamily,
  link,
  negativeBinomialFamily,
  poissonFamily,
  type Family,
  type LinkName,
} from 'aifn-compute/probability/likelihoods'
import { dataset } from 'aifn-compute/learning/estimators'
import { glm } from 'aifn-methods/learning/generalised/glm'
import { child, stream, uniform } from 'aifn-compute/foundation/random'
import { fromData, linspace, matmul, tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { Player } from 'aifn-render/controls'
import { ControlRow, Figure } from 'aifn-render/layout'
import { int, choice, useFigureState, variants } from 'aifn-render/state'
import { Curve, Plot, Plots, Points, Readout, useAxis } from 'aifn-render/viz'
import { formatValue } from '@lab/views'

type Setup = {
  label: string
  family: () => Family
  links: LinkName[]
  truth: (x: number) => number
  dispersion: number
}

const SETUPS = {
  gaussian: {
    label: 'Gaussian',
    family: gaussianFamily,
    links: ['identity', 'log'],
    truth: (x) => 1 + 0.8 * x,
    dispersion: 0.5,
  },
  bernoulli: {
    label: 'Bernoulli',
    family: binomialFamily,
    links: ['logit', 'probit', 'cloglog'],
    truth: (x) => 1 / (1 + Math.exp(-(0.3 + 1.5 * x))),
    dispersion: 1,
  },
  poisson: {
    label: 'Poisson',
    family: poissonFamily,
    links: ['log', 'identity', 'sqrt'],
    truth: (x) => Math.exp(0.5 + 0.6 * x),
    dispersion: 1,
  },
  gamma: {
    label: 'gamma',
    family: gammaFamily,
    links: ['log', 'inverse', 'identity'],
    truth: (x) => Math.exp(0.8 + 0.5 * x),
    dispersion: 0.3,
  },
  inverseGaussian: {
    label: 'inverse Gaussian',
    family: inverseGaussianFamily,
    links: ['log', 'inverse-squared'],
    truth: (x) => Math.exp(0.4 + 0.4 * x),
    dispersion: 0.2,
  },
  negativeBinomial: {
    label: 'negative binomial (θ = 2)',
    family: () => negativeBinomialFamily(2),
    links: ['log', 'sqrt'],
    truth: (x) => Math.exp(0.8 + 0.6 * x),
    dispersion: 1,
  },
} satisfies Record<string, Setup>
type SetupName = keyof typeof SETUPS

/** The family, with the links it supports (the first is the default) and the data seed every family shares. */
const MODEL = variants(
  Object.fromEntries(
    (Object.keys(SETUPS) as SetupName[]).map((k) => {
      const links = SETUPS[k].links as LinkName[]
      return [k, { label: SETUPS[k].label, params: { link: choice(links, links[0], { label: 'link' }) } }]
    }),
  ) as Record<SetupName, { label: string; params: { link: ReturnType<typeof choice<LinkName>> } }>,
  {
    label: '1 · model',
    choiceLabel: 'family',
    initial: 'poisson',
    shared: { seed: int(1, { ge: 1, le: 20, label: 'data seed' }) },
  },
)
const N = 80
const GRID = linspace(-2, 2, 101)
const GRID_X = toFlat(GRID)
const GRID_DESIGN = fromData(Float64Array.from(GRID_X.flatMap((v) => [v, 1])), [GRID_X.length, 2])

/** One GLM per family, fitted by IRLS, stepped iteration by iteration. */
export function IrlsSteps() {
  const figure = useFigureState({ model: MODEL })
  const name = figure.model.key as SetupName
  const setup: Setup = SETUPS[name]
  const chosenLink = figure.model.values.link as LinkName
  const seed = figure.model.values.seed as number
  const [step, setStep] = useState(0)
  const data = useMemo(() => {
    const s = stream(`glm-${name}-${seed}`)
    const x = toFlat(uniform(child(s, 'x'), -2, 2, { shape: [N] }) as Tensor)
    const mu = tensor(x.map(setup.truth))
    // Responses drawn from the family's own predictive distribution at the true means.
    const y = toFlat(setup.family().predictive(mu, setup.dispersion).sample(child(s, 'y'), { shape: [] }) as Tensor)
    return { x, y }
  }, [name, seed, setup])
  const model = useMemo(
    () =>
      glm({ family: setup.family(), link: chosenLink, maxSteps: 25 }).fit(
        dataset(fromData(Float64Array.from(data.x), [N, 1]), tensor(data.y)),
      ),
    [data, setup, chosenLink],
  )
  const steps = model.training.steps
  const k = Math.min(step, steps.length - 1)
  const state = steps[k]
  const curve = useMemo(() => {
    if (!state.coefficients) return null
    const eta = matmul(GRID_DESIGN, state.coefficients) as Tensor
    return toFlat(link(chosenLink).inverse(eta) as Tensor)
  }, [state, chosenLink])
  const truth = useMemo(() => GRID_X.map(setup.truth), [setup])
  const start = useMemo(() => (k === 0 ? toFlat(state.mu) : null), [k, state])
  // The deviance above its final value, on a log axis: the gap closes quadratically near the optimum.
  const gap = useMemo(() => {
    const deviances = toFlat(model.training.series.deviance)
    return { x: Array.from(model.training.index), y: deviances.map((v) => Math.max(v - model.deviance, 1e-12)) }
  }, [model])
  const now = useMemo(() => ({ x: [k], y: [gap.y[k]] }), [k, gap])
  const xa = useAxis({ label: 'x' })
  const ya = useAxis({ label: 'y, μ(x)', hold: 'initial', key: `${name}-${seed}` })
  const stepAxis = useAxis({ label: 'IRLS step', hold: 'initial', key: model })
  const gapAxis = useAxis({ label: 'deviance gap', log: true, hold: 'initial', key: model })
  const coef = state.coefficients ? toFlat(state.coefficients) : [NaN, NaN]
  return (
    <Figure
      title="IRLS for six families"
      defaultSize="L"
      purpose="Each IRLS step is a weighted least-squares fit to the working response z = η + (y − μ)/μ′(η); a few steps take the constant start to the maximum-likelihood fit."
      state={figure}
      controls={
        <ControlRow label="2 · iterations">
          <Player value={k} onChange={setStep} count={steps.length} format={(i) => `step ${i}`} />
        </ControlRow>
      }
      readouts={
        <>
          <Readout label="slope β₁" value={formatValue(coef[0])} />
          <Readout label="intercept β₀" value={formatValue(coef[1])} />
          <Readout label="deviance" value={formatValue(state.deviance)} />
          <Readout label="step halvings" value={state.halvings} />
          <Readout label="converged after" value={`${model.steps} steps`} />
          <Readout label="dispersion φ̂" value={formatValue(model.dispersion)} />
          <Readout label="p-value of β₁" value={formatValue(toFlat(model.pValues)[0])} />
        </>
      }
      caption="Step through the iterations: step 0 is the family's starting mean, and each step refits. The canonical link converges in a handful of steps; a non-canonical one (identity for Poisson, inverse for gamma) may halve a step that would leave the mean space. The lower panel shows how far the deviance is above its final value, on a log scale; it never rises, and near the optimum the gap shrinks quadratically (the digits double each step). Gaps below 10⁻¹² are drawn at 10⁻¹²."
    >
      <Plots rows={2} heights={[1.6, 1]}>
        <Plot x={xa} y={ya}>
          <Points name="data" x={data.x} y={data.y} muted />
          <Curve name="true mean" x={GRID_X} y={truth} slot={2} dashed />
          {curve && <Curve name={`μ after step ${k}`} x={GRID_X} y={curve} slot={0} />}
          {start && <Points name="starting means μ₀" x={data.x} y={start} slot={0} />}
        </Plot>
        <Plot x={stepAxis} y={gapAxis} legend={false}>
          <Curve name="deviance − final deviance" x={gap.x} y={gap.y} slot={0} showPoints />
          <Points name="current" x={now.x} y={now.y} emphasis />
        </Plot>
      </Plots>
    </Figure>
  )
}
