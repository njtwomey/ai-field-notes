import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamButton,
  ParamSlider,
  ParamSwitch,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from '@/components/viz'
import { boundaryLine, corrupt, fitLogistic, lineAnchors, parametricTest, twoGaussians } from './noiseTests'

const XS = [-4.5, 4.5]
const SHOWN = 400
const logit = (p: number) => Math.log(p / (1 - p))

/**
 * Two unit-variance Gaussians at (1, 1) and (−1, −1), labels flipped with rates α (positives) and β (negatives),
 * logistic regression fitted to the noisy labels. With `withTest`, anchors on the true boundary (posterior 1/2) and
 * the z-test of Poyiadzi et al. (2022) for H₀: α = β.
 */
export function NoisyBoundary({ withTest = false }: { withTest?: boolean }) {
  const alpha = useParam(0, { min: 0, max: 0.4, step: 0.01 })
  const beta = useParam(0.15, { min: 0, max: 0.4, step: 0.01 })
  const n = useParam(1000, { min: 200, max: 5000, step: 100 })
  const k = useParam(8, { min: 1, max: 32, step: 1 })
  const [tied, setTied] = useState(false)
  const [seed, setSeed] = useState(1)
  const b = tied ? alpha.value : beta.value

  const sample = useMemo(() => twoGaussians(n.value, seed), [n.value, seed])
  const noisy = useMemo(() => corrupt(sample, alpha.value, b), [sample, alpha.value, b])
  const fit = useMemo(() => fitLogistic(sample.X, noisy), [sample, noisy])
  const anchors = useMemo(() => lineAnchors(k.value, seed + 7), [k.value, seed])
  const test = useMemo(() => parametricTest(fit, anchors), [fit, anchors])

  // Clean posterior: logit η = 2(x₁ + x₂). The noisy Bayes rule thresholds η̃ at 1/2, i.e. η at (1/2 − β)/(1 − α − β).
  const etaStar = (0.5 - b) / (1 - alpha.value - b)
  const noisyBayes = logit(Math.min(Math.max(etaStar, 1e-6), 1 - 1e-6)) / 2

  const series: XYSeries[] = [
    {
      name: 'points',
      type: 'scatter',
      x: sample.X.slice(0, SHOWN).map((p) => p[0]),
      y: sample.X.slice(0, SHOWN).map((p) => p[1]),
      group: noisy.slice(0, SHOWN),
      groupNames: ['noisy label 0', 'noisy label 1'],
    },
    { name: 'clean Bayes boundary', type: 'line', x: XS, y: XS.map((x) => -x), dashed: true, muted: true },
    { name: 'noisy Bayes boundary', type: 'line', x: XS, y: XS.map((x) => noisyBayes - x), dashed: true, slot: 2 },
    { name: 'fitted to noisy labels', type: 'line', x: XS, y: boundaryLine(fit.theta, XS), emphasis: true },
  ]
  if (withTest) {
    series.push({
      name: 'anchors',
      type: 'scatter',
      x: anchors.map((a) => a[0]),
      y: anchors.map((a) => a[1]),
      emphasis: true,
    })
  }

  return (
    <Interactive
      title={withTest ? 'Testing for class-conditional noise at anchors' : 'Where the noisy decision boundary goes'}
      caption={
        withTest
          ? 'Logistic regression is fitted to labels flipped with rates α (positives) and β (negatives). The anchors lie on the clean boundary, where the true posterior is 1/2. Under uniform noise (α = β) the fitted posterior there stays near 1/2; under class-conditional noise it moves towards (1 − α + β)/2 (further, in this example, because the logistic model is misspecified under noise), and the z-score grows with the number of points, the number of anchors and |α − β|. Tie β to α to see the null hold.'
          : 'Labels are flipped with rate α for positives and β for negatives. Under uniform noise (tie β to α) the noisy Bayes boundary, where the noisy posterior is 1/2, stays on the clean one and so does the fitted logistic regression. Under class-conditional noise both move into the region of the class whose labels are flipped more. The fitted line moves further than the noisy Bayes boundary, because the noisy posterior is no longer a logistic function of a linear score and the model is misspecified.'
      }
      controls={
        <>
          <ParamSlider label="α = P(flip | y = 1)" param={alpha} />
          <ParamSlider
            label="β = P(flip | y = 0)"
            param={tied ? { ...beta, value: alpha.value, set: alpha.set } : beta}
          />
          <ParamSwitch label="uniform noise (β = α)" checked={tied} onChange={setTied} />
          <ParamSlider label="training points N" param={n} format={(v) => String(v)} />
          {withTest && <ParamSlider label="anchors k" param={k} format={(v) => String(v)} />}
          <ParamButton onClick={() => setSeed((s) => s + 1)}>New sample</ParamButton>
        </>
      }
      readout={
        withTest ? (
          <>
            <Readout label="mean fitted posterior at anchors" value={formatNumber(test.etaBar)} />
            <Readout label="noisy posterior there, (1 − α + β)/2" value={formatNumber((1 - alpha.value + b) / 2)} />
            <Readout label="standard error under H₀" value={formatNumber(test.se)} />
            <Readout label="z" value={formatNumber(test.z)} />
            <Readout label="two-sided p-value" value={test.p < 1e-4 ? '< 0.0001' : formatNumber(test.p)} />
          </>
        ) : (
          <>
            <Readout label="clean η on the noisy Bayes boundary" value={formatNumber(etaStar)} />
            <Readout label="noisy Bayes boundary crosses x₁ = x₂ at" value={formatNumber(noisyBayes / 2)} />
            <Readout
              label="fitted boundary crosses x₁ = x₂ at"
              value={formatNumber(-fit.theta[0] / (fit.theta[1] + fit.theta[2]))}
            />
          </>
        )
      }
    >
      <XYChart
        series={series}
        xLabel="x₁"
        yLabel="x₂"
        xRange={[-4.5, 4.5]}
        yRange={[-4.5, 4.5]}
        equalAspect
        ariaLabel="Two-class scatter with clean, noisy and fitted decision boundaries"
      />
    </Interactive>
  )
}
