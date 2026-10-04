import { useMemo, useState } from 'react'
import {
  Figure,
  ControlGroup,
  NumberSelector,
  Player,
  Plots,
  Plot,
  Bars,
  Curve,
  Annotation,
  Points,
  Readout,
  useAxis,
} from 'aifn-render'
import { latentFactors, latentFactorModel } from 'aifn-methods/data/synthetic'
import { factorAnalysis, probabilisticPca, latentGaussianSteps } from 'aifn-methods/unsupervised/embedding/linear'
import { dataset } from 'aifn/learning/estimators'
import { stream } from 'aifn/foundation/random'
import { toFlat, type Tensor } from 'aifn/foundation/tensor'
import { trace } from 'aifn/foundation/trace'

export function FactorAnalysisExplorer() {
  const [n, setN] = useState(400)
  const [spread, setSpread] = useState(2.0)
  const [seed, setSeed] = useState(0)
  const [step, setStep] = useState(100)
  const iterations = 100
  const d = 8

  const run = useMemo(() => {
    const data = latentFactors(stream(`fa-${seed}`), { n, d, latent: 2, spread })
    const truth = latentFactorModel({ d, latent: 2, spread })
    const tr = trace(
      latentGaussianSteps(data.x, { latent: 2, noise: 'diagonal', tolerance: 0 }),
      undefined,
      iterations,
      { stream: stream(`fa-em-${seed}`) },
    )
    const communality = (W: Tensor) => {
      const w = toFlat(W)
      return Array.from({ length: d }, (_, i) => w[i * 2] ** 2 + w[i * 2 + 1] ** 2)
    }
    const fa = factorAnalysis({ latent: 2, maxSteps: iterations, tolerance: 0 }).fit(dataset(data.x), {
      stream: stream(`fa-em-${seed}`),
    })
    const pp = probabilisticPca({ latent: 2 }).fit(dataset(data.x))

    return {
      steps: tr.steps.map((s) => ({
        t: s.t,
        ll: s.logLikelihood,
        noise: Array.from(toFlat(s.noise)),
        communality: communality(s.loadings),
      })),
      trueNoise: Array.from(toFlat(truth.noise)),
      trueCommunality: communality(truth.loadings),
      ppcaNoise: pp.noiseVariance,
      ppcaCommunality: communality(pp.loadings),
      ppcaLl: pp.logLikelihood,
      faLl: fa.logLikelihood,
    }
  }, [n, spread, seed])

  const currentStep = Math.min(step, run.steps.length - 1)
  const now = run.steps[currentStep]

  const features = Array.from({ length: d }, (_, i) => i + 1)
  const fAxis = useAxis({ label: 'feature index', range: [0.4, d + 0.6], integer: true })

  const top = Math.max(...run.trueNoise, ...run.steps.flatMap((s) => s.noise), run.ppcaNoise) * 1.15
  const ctop =
    Math.max(...run.trueCommunality, ...run.ppcaCommunality, ...run.steps.flatMap((s) => s.communality)) * 1.15

  const noiseAxis = useAxis({ label: 'noise variance ψᵢ', range: [0, top], key: `${spread}-${seed}-${n}` })
  const commAxis = useAxis({ label: 'communality Σⱼ Wᵢⱼ²', range: [0, ctop], key: `${spread}-${seed}-${n}` })
  const itAxis = useAxis({ label: 'EM iteration', range: [0, Math.max(1, iterations)] })
  const llAxis = useAxis({ label: 'log-likelihood per row', hold: 'union', key: `${spread}-${seed}-${n}` })

  const shift = (k: number) => features.map((f) => f + (k - 1) * 0.27)

  return (
    <Figure
      title="Factor analysis vs Probabilistic PCA under heteroskedastic noise"
      purpose="Both fit x = Wz + ε with two latent factors; Factor Analysis models each feature's idiosyncratic noise variance ψᵢ, whereas Probabilistic PCA forces an isotropic shared σ²I, wrongly absorbing excess sensor noise into loadings on high-noise features."
      defaultSize="L"
      controls={
        <>
          <ControlGroup title="1 · Data and noise heteroskedasticity">
            <NumberSelector
              label="largest noise variance ψ_d"
              value={spread}
              onChange={setSpread}
              min={0.1}
              max={4.0}
              step={0.1}
              suggestions={[0.5, 1.0, 2.0, 3.5]}
            />
            <NumberSelector
              label="sample count n"
              value={n}
              onChange={setN}
              min={50}
              max={1500}
              step={50}
              suggestions={[100, 400, 1000]}
            />
            <NumberSelector
              label="seed"
              value={seed}
              onChange={setSeed}
              min={0}
              max={20}
              step={1}
              suggestions={[0, 1, 2, 5]}
            />
          </ControlGroup>
          <ControlGroup title="2 · EM iterations">
            <Player value={currentStep} onChange={setStep} count={run.steps.length} label="iteration" />
          </ControlGroup>
        </>
      }
      readouts={
        <>
          <Readout label="EM iteration" value={`${now.t} of ${run.steps.length - 1}`} />
          <Readout label="FA log-likelihood / row" value={now.ll.toFixed(4)} />
          <Readout label="PPCA closed-form maximum" value={run.ppcaLl.toFixed(4)} />
          <Readout label="PPCA shared noise σ²" value={run.ppcaNoise.toFixed(3)} />
        </>
      }
      caption="8 features generated with 2 latent factors. Noise variances grow geometrically from 0.05 (feature 1) to the chosen spread (feature 8). Top: Noise variance estimated for each feature (black: ground truth, blue: Factor Analysis at active iteration, orange: PPCA shared isotropic σ²). Middle: Communality (variance explained by factors). Bottom: Factor Analysis EM log-likelihood curve converging monotonically above the constrained PPCA maximum."
    >
      <Plots rows={3} heights={[35, 35, 30]}>
        <Plot x={fAxis} y={noiseAxis} title="Feature-specific noise variance ψᵢ">
          <Bars name="true ψᵢ" x={shift(0)} y={run.trueNoise} width={0.25} emphasis />
          <Bars name="Factor Analysis ψᵢ" x={shift(1)} y={now.noise} width={0.25} slot={0} />
          <Bars name="PPCA shared σ²" x={shift(2)} y={features.map(() => run.ppcaNoise)} width={0.25} slot={1} />
        </Plot>
        <Plot x={fAxis} y={commAxis} title="Feature communality Σⱼ Wᵢⱼ²">
          <Bars name="true communality" x={shift(0)} y={run.trueCommunality} width={0.25} emphasis />
          <Bars name="Factor Analysis" x={shift(1)} y={now.communality} width={0.25} slot={0} />
          <Bars name="PPCA" x={shift(2)} y={run.ppcaCommunality} width={0.25} slot={1} />
        </Plot>
        <Plot x={itAxis} y={llAxis} title="EM log-likelihood convergence">
          <Curve name="Factor Analysis (EM)" x={run.steps.map((s) => s.t)} y={run.steps.map((s) => s.ll)} slot={0} />
          <Annotation y={run.ppcaLl} dashed text="PPCA maximum" />
          <Points name="current iteration" x={[now.t]} y={[now.ll]} emphasis />
        </Plot>
      </Plots>
    </Figure>
  )
}
