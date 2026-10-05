import { useMemo } from 'react'
import {
  Figure,
  float,
  formatNumber,
  int,
  MathText,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { fitShared1d, nonParallelSample } from '../_shared/ordinal'
import { brant } from './brant'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { sigmoid } from 'aifn-compute/numerics/special'

const X = toFlat(linspace(-3, 3, 121))
const K = 4
const BINS = 10

/**
 * One feature, four classes, with split-specific slopes β_j = 1.2(1 + δ(j − 1)). At δ = 0 proportional odds holds.
 * Solid curves are separate binary logits per split; dashed curves the proportional-odds fit with one shared slope.
 */
export function SlopeTest() {
  const state = useFigureState({
    delta: float(0, { min: -0.6, max: 0.6, step: 0.02, label: 'slope spread δ' }),
    n: int(400, { min: 100, max: 2000, step: 50, label: 'sample size n' }),
  })

  const result = useMemo(() => {
    const { x, y } = nonParallelSample(state.n, state.delta, 11)
    const test = brant(x, y)
    const pom = fitShared1d(x, y, K, 'cumulative-logit')
    const series: SeriesSpec[] = []
    // Empirical exceedance proportions in equal-width bins of x.
    const edges = toFlat(linspace(-3, 3, BINS + 1))
    const centres = edges.slice(0, -1).map((e, i) => (e + edges[i + 1]) / 2)
    ;[0, 1, 2].forEach((j) => {
      const f = test.fits[j]
      series.push({
        name: `P(y > ${j + 1}), separate`,
        type: 'line',
        x: X,
        y: X.map((v) => sigmoid(f.a + f.b * v)),
        slot: j,
      })
      series.push({
        name: `P(y > ${j + 1}), shared slope`,
        type: 'line',
        x: X,
        y: X.map((v) => sigmoid(pom.w * v - pom.theta[j])),
        slot: j,
        dashed: true,
      })
      const props = centres.map((_, b) => {
        const inBin = x.flatMap((v, i) => (v >= edges[b] && v < edges[b + 1] ? [y[i] > j ? 1 : 0] : []))
        return inBin.length ? inBin.reduce((s: number, t) => s + t, 0) / inBin.length : NaN
      })
      series.push({ name: `binned y > ${j + 1}`, type: 'scatter', x: centres, y: props, slot: j })
    })
    return { test, pom, series }
  }, [state.delta, state.n])

  const { test, pom } = result
  const xAxis = useAxis({ label: 'feature x', range: [-3, 3] })
  const yAxis = useAxis({ label: 'P(y > j | x)', range: [0, 1] })
  return (
    <Figure
      title="Testing whether the cumulative splits share a slope"
      state={state}
      caption={
        <MathText text="Data come from four classes whose cumulative splits have their own slopes, $P(y > j \mid x) = \sigma(\beta_j x - \theta_j)$ with $\beta_j = 1.2\,(1 + \delta(j - 2))$ for $j = 1, 2, 3$. Solid curves are separate logistic regressions for each split $y > j$; dashed curves are the proportional-odds fit, which forces one slope. Points are binned proportions. At $\delta = 0$ the separate slopes agree up to noise and Brant's test rejects in about 5% of samples at the 0.05 level; as $|\delta|$ grows the slopes separate and the p-value falls. A larger sample detects a smaller spread." />
      }

      readouts={
        <>
          {test.slopes.map((b, j) => (
            <Readout
              key={j}
              label={`slope, split ${j + 1}`}
              value={`${formatNumber(b)} ± ${formatNumber(test.se[j])}`}
            />
          ))}
          <Readout label="shared slope" value={formatNumber(pom.w)} />
          <Readout label="Brant χ² (2 df)" value={formatNumber(test.chi2)} />
          <Readout label="p-value" value={test.p < 1e-4 ? '< 0.0001' : formatNumber(test.p)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={320} ariaLabel={'Separate and shared-slope cumulative logit curves'}>
        {seriesLayers(result.series)}
      </Plot>
    </Figure>
  )
}
