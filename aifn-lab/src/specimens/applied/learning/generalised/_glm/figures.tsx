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
} from 'aifn/probability/likelihoods'
import { dataset } from 'aifn/learning/estimators'
import { glm } from 'aifn-applied/learning/generalised/glm'
import { child, stream, uniform } from 'aifn/foundation/random'
import { fromData, linspace, matmul, tensor, toFlat, type Tensor } from 'aifn/foundation/tensor'
import { Player, Select, Slider } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { Panel, Readout, Subplots, XYChart, type XYSeries } from '@lab/viz'
import { formatValue } from '@lab/views'

type Setup = { family: () => Family; links: LinkName[]; truth: (x: number) => number; dispersion: number }

const SETUPS: Record<string, Setup> = {
  Gaussian: { family: gaussianFamily, links: ['identity', 'log'], truth: (x) => 1 + 0.8 * x, dispersion: 0.5 },
  Bernoulli: {
    family: binomialFamily,
    links: ['logit', 'probit', 'cloglog'],
    truth: (x) => 1 / (1 + Math.exp(-(0.3 + 1.5 * x))),
    dispersion: 1,
  },
  Poisson: {
    family: poissonFamily,
    links: ['log', 'identity', 'sqrt'],
    truth: (x) => Math.exp(0.5 + 0.6 * x),
    dispersion: 1,
  },
  gamma: {
    family: gammaFamily,
    links: ['log', 'inverse', 'identity'],
    truth: (x) => Math.exp(0.8 + 0.5 * x),
    dispersion: 0.3,
  },
  'inverse Gaussian': {
    family: inverseGaussianFamily,
    links: ['log', 'inverse-squared'],
    truth: (x) => Math.exp(0.4 + 0.4 * x),
    dispersion: 0.2,
  },
  'negative binomial (θ = 2)': {
    family: () => negativeBinomialFamily(2),
    links: ['log', 'sqrt'],
    truth: (x) => Math.exp(0.8 + 0.6 * x),
    dispersion: 1,
  },
}
const NAMES = Object.keys(SETUPS)
const N = 80
const GRID = linspace(-2, 2, 101)
const GRID_X = toFlat(GRID)
const GRID_DESIGN = fromData(Float64Array.from(GRID_X.flatMap((v) => [v, 1])), [GRID_X.length, 2])

/** One GLM per family, fitted by IRLS, stepped iteration by iteration. */
export function IrlsSteps() {
  const [name, setName] = useState('Poisson')
  const setup = SETUPS[name]
  const [linkName, setLinkName] = useState<LinkName>(setup.links[0])
  const [step, setStep] = useState(1)
  const [seed, setSeed] = useState(1)
  const chosenLink = setup.links.includes(linkName) ? linkName : setup.links[0]
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
  const top: XYSeries[] = [
    { name: 'data', type: 'scatter', x: data.x, y: data.y, muted: true },
    { name: 'true mean', type: 'line', x: GRID_X, y: GRID_X.map(setup.truth), slot: 2, dashed: true },
    ...(curve ? [{ name: `μ after step ${k}`, type: 'line' as const, x: GRID_X, y: curve, slot: 0 }] : []),
    ...(k === 0
      ? [{ name: 'starting means μ₀', type: 'scatter' as const, x: data.x, y: toFlat(state.mu), slot: 0 }]
      : []),
  ]
  // The deviance above its final value, on a log axis: the gap closes quadratically near the optimum.
  const deviances = toFlat(model.training.series.deviance)
  const gap = deviances.map((v) => Math.max(v - model.deviance, 1e-12))
  const bottom: XYSeries[] = [
    {
      name: 'deviance − final deviance',
      type: 'line',
      x: Array.from(model.training.index),
      y: gap,
      slot: 0,
      showPoints: true,
    },
    { name: 'current', type: 'scatter', x: [k], y: [gap[k]], emphasis: true },
  ]
  const coef = state.coefficients ? toFlat(state.coefficients) : [NaN, NaN]
  return (
    <Figure
      title="IRLS for six families"
      defaultSize="L"
      description="Each IRLS step is a weighted least-squares fit to the working response z = η + (y − μ)/μ′(η); a few steps take the constant start to the maximum-likelihood fit."
      controls={
        <>
          <ControlRow label="Model">
            <Select
              label="family"
              value={name}
              onChange={(v) => {
                setName(v)
                setLinkName(SETUPS[v].links[0])
                setStep(1)
              }}
              options={NAMES}
            />
            <Select
              label="link"
              value={chosenLink}
              onChange={(v) => setLinkName(v as LinkName)}
              options={setup.links}
            />
            <Slider label="data seed" value={seed} onChange={setSeed} min={1} max={20} step={1} />
          </ControlRow>
          <ControlRow label="Iterations">
            <Player value={k} onChange={setStep} count={steps.length} format={(i) => `step ${i}`} />
          </ControlRow>
        </>
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
      <Subplots rows={2} heightRatios={[1.6, 1]}>
        <Panel rescaleOnChange={false} axisKey={`${name}-${seed}`}>
          <XYChart series={top} xLabel="x" yLabel="y, μ(x)" />
        </Panel>
        <Panel>
          <XYChart series={bottom} xLabel="IRLS step" yLabel="deviance gap" integerX yLog />
        </Panel>
      </Subplots>
    </Figure>
  )
}
