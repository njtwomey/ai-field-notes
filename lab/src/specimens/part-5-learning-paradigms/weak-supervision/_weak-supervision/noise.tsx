/**
 * Showcase: testing for class-conditional label noise from anchor points (Poyiadzi, Yang, Twomey and
 * Santos-Rodriguez 2022; Yang et al. 2024, the local maximum-likelihood version). Data, anchors, the test, its power
 * and the simulation are aifn's (`classConditionalNoise`, `noiseLayoutAnchors`, `classConditionalNoiseTest`,
 * `noiseTestPower`, `noiseTestSimulation`); the lab draws them.
 */
import { useMemo } from 'react'
import { child, stream } from 'aifn-compute/foundation/random'
import { fromRows, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { Normal } from 'aifn-compute/probability/distributions'
import { classConditionalNoise, noiseLayoutAnchors, type NoiseLayout } from 'aifn-methods/data/synthetic'
import {
  classConditionalNoiseTest,
  noiseTestPower,
  type NoiseTestModel,
  type NoiseTestSimulation,
} from 'aifn-methods/learning/weak-supervision'
import { Figure } from 'aifn-render/layout'
import { call, choice, float, int, row, useFigureState, when, type Task } from 'aifn-render/state'
import { formatValue, TrainControls, useTrainedRun } from '@lab/views'
import { Area, Bars, Curve, Handle, Plot, Plots, Points, Raster, Readout, useAxis } from 'aifn-render/viz'

const f3 = (v: number) => (Number.isFinite(v) ? formatValue(Number(v.toPrecision(3))) : '—')
const LAYOUTS = [
  { value: 'gaussians', label: 'two Gaussians (2022 paper)' },
  { value: 'xor', label: 'symmetric XOR (2024 paper)' },
  { value: 'asymmetric-xor', label: 'asymmetric XOR (2024 paper)' },
]
const MODELS = [
  { value: 'parametric', label: 'parametric: logistic MLE' },
  { value: 'local', label: 'local likelihood (2024)' },
]
const RATES = Array.from({ length: 21 }, (_, i) => (0.4 * i) / 20)
const density = (mean: number, sd: number, xs: number[]) => {
  const law = Normal(mean, sd)
  return xs.map((x) => Math.exp(Number(law.logProb(x))))
}

export function NoiseTestFigure() {
  const state = useFigureState({
    data: row('1 · data', {
      layout: choice(LAYOUTS, 'gaussians', { label: 'classes' }),
      n: int(1000, { ge: 50, le: 5000, suggestions: [500, 1000, 2000], label: 'training points N' }),
      seed: int(1, { ge: 0, le: 9999, label: 'seed' }),
    }),
    noise: row('2 · noise rates (or drag the point in the α–β plane)', {
      alpha: float(0, { ge: 0, le: 0.4, suggestions: [0, 0.05, 0.1, 0.2], label: 'α = P(ỹ = 0 | y = 1)' }),
      beta: float(0.1, { ge: 0, le: 0.4, suggestions: [0, 0.05, 0.1, 0.2], label: 'β = P(ỹ = 1 | y = 0)' }),
    }),
    test: row('3 · test', {
      model: choice(MODELS, 'parametric', { label: 'model' }),
      bandwidth: float(1, {
        gt: 0,
        scale: 'log10',
        suggestions: [0.5, 1, 2],
        label: 'bandwidth h',
        when: when('model', 'local'),
      }),
      k: int(4, { ge: 1, le: 64, suggestions: [1, 2, 4, 8, 16], label: 'anchors k' }),
      delta: float(0, { ge: 0, lt: 0.5, suggestions: [0, 0.05, 0.1], label: 'anchor relaxation δ' }),
    }),
  })
  const { layout, n, seed } = state.data
  const { alpha, beta } = state.noise
  const { model, bandwidth, k, delta } = state.test
  const data = useMemo(
    () =>
      classConditionalNoise(stream(`lab/noise/${layout}/${seed}`), {
        n,
        layout: layout as NoiseLayout,
        alpha,
        beta,
        truth: false,
      }),
    [layout, n, seed, alpha, beta],
  )
  const anchors = useMemo(
    () => noiseLayoutAnchors(stream(`lab/noise/anchors/${seed}`), layout as NoiseLayout, k, delta),
    [layout, seed, k, delta],
  )
  const result = useMemo(
    () =>
      classConditionalNoiseTest(data.x, toFlat(data.y), fromRows(anchors), {
        model: model as NoiseTestModel,
        bandwidth,
      }),
    [data, anchors, model, bandwidth],
  )
  const rows = useMemo(() => toRows(data.x as Tensor) as number[][], [data])
  const noisy = useMemo(() => Array.from(toFlat(data.y)), [data])
  const flipped = useMemo(() => Array.from(toFlat(data.flipped)).flatMap((f, i) => (f ? [i] : [])), [data])
  const v = result.variance
  // The analytic power over the α–β plane, at this dataset's v.
  const power = useMemo(
    () => RATES.map((b) => RATES.map((a) => (a + b < 1 ? noiseTestPower({ variance: v, alpha: a, beta: b }) : NaN))),
    [v],
  )
  const sd = Math.sqrt(v)
  const etaAlt = (1 - alpha + beta) / 2
  const sdAlt = 4 * etaAlt * (1 - etaAlt) * sd
  const lo = Math.min(0.5 - 5 * sd, etaAlt - 5 * sdAlt)
  const hi = Math.max(0.5 + 5 * sd, etaAlt + 5 * sdAlt)
  const xs = Array.from({ length: 201 }, (_, i) => lo + ((hi - lo) * i) / 200)
  const z = 1.959963984540054
  const reject = xs.filter((x) => Math.abs(x - 0.5) > z * sd)
  const ax = useAxis({ label: 'x₁', key: layout })
  const ay = useAxis({ label: 'x₂', equal: ax, key: layout })
  const pa = useAxis({ label: 'α', range: [0, 0.4] })
  const pb = useAxis({ label: 'β', range: [0, 0.4], equal: pa })
  const ex = useAxis({
    label: 'η̄, the mean noisy posterior at the anchors',
    range: [lo, hi],
    key: `${lo.toFixed(3)}/${hi.toFixed(3)}`,
  })
  const ey = useAxis({ label: 'density', hold: 'union', key: `${layout}/${model}/${k}` })
  const nullD = density(0.5, sd, xs)
  return (
    <Figure
      title="Is the label noise class-conditional? A z-test at the anchor points"
      purpose="At an anchor point the clean posterior is ½, so the noisy posterior is (1 − α + β)/2: exactly ½ under uniform noise, off ½ when the classes are flipped at different rates. The test asks whether the fitted posterior averages ½ over the anchors."
      state={state}
      defaultSize="XL"
      readouts={{
        test: (
          <>
            <Readout label="η̄" value={f3(result.meanEstimate)} />
            <Readout label="√v under H₀" value={f3(sd)} />
            <Readout label="z" value={f3(result.statistic)} />
            <Readout label="p-value" value={f3(result.pValue)} />
            <Readout label="at 5%" value={result.pValue < 0.05 ? 'reject uniform noise' : 'retain uniform noise'} />
          </>
        ),
        truth: (
          <>
            <Readout label="(1 − α + β)/2" value={f3(etaAlt)} />
            <Readout label="labels flipped" value={`${flipped.length} of ${n}`} />
            <Readout label="power at (α, β), level 5%" value={f3(noiseTestPower({ variance: v, alpha, beta }))} />
          </>
        ),
      }}
      caption={
        <>
          aifn <code>classConditionalNoise</code> ({n} points, labels flipped at α = {alpha} for class 1 and β = {beta}{' '}
          for class 0) and <code>noiseLayoutAnchors</code> (k = {k},{' '}
          {delta > 0 ? `relaxed to |η − ½| ≤ ${delta}` : 'strict, η = ½'}). <code>classConditionalNoiseTest</code> fits{' '}
          {model === 'parametric'
            ? 'a logistic regression by maximum likelihood (Poyiadzi et al. 2022)'
            : `a local likelihood logistic regression at each anchor, bandwidth ${bandwidth} (Yang et al. 2024)`}{' '}
          to the noisy labels and compares η̄ with N(½, v). Left: colour is the noisy label, red halos the flipped
          points, ink marks the anchors. Middle: the analytic power (<code>noiseTestPower</code>, Eq. 11) over the noise
          rates at this v; the diagonal α = β is the null; drag the point to set α and β. Right: the null law of η̄ with
          the 5% rejection region shaded, the law under these noise rates (dashed) and the observed η̄ (ink).
        </>
      }
    >
      <Plots cols={3} widths={[3, 2, 3]}>
        <Plot x={ax} y={ay}>
          <Points
            name="flipped"
            x={flipped.map((i) => rows[i][0])}
            y={flipped.map((i) => rows[i][1])}
            tone="destructive"
            size={9}
          />
          <Points
            name="points"
            x={rows.map((r) => r[0])}
            y={rows.map((r) => r[1])}
            group={noisy}
            groupNames={['noisy label 0', 'noisy label 1']}
            size={4}
          />
          <Points name="anchors" x={anchors.map((a) => a[0])} y={anchors.map((a) => a[1])} emphasis size={11} />
        </Plot>
        <Plot x={pa} y={pb} title="power over the noise rates">
          <Raster x={RATES} y={RATES} z={power} scale="sequential" range={[0, 1]} valueLabel="power" />
          <Curve name="H₀: α = β" x={[0, 0.4]} y={[0, 0.4]} emphasis dashed width={1} />
          <Handle
            kind="point"
            at={[alpha, beta]}
            onDrag={([a, b]) => {
              state.set('alpha', Math.round(Math.min(0.4, Math.max(0, a)) * 100) / 100)
              state.set('beta', Math.round(Math.min(0.4, Math.max(0, b)) * 100) / 100)
            }}
            label={`(${alpha}, ${beta})`}
          />
        </Plot>
        <Plot x={ex} y={ey} title="η̄ against its null law">
          <Area
            name="rejection region (5%)"
            x={reject.filter((x) => x < 0.5)}
            y={density(
              0.5,
              sd,
              reject.filter((x) => x < 0.5),
            )}
            tone="destructive"
            opacity={0.25}
            line={false}
          />
          <Area
            name="rejection region (5%)"
            x={reject.filter((x) => x > 0.5)}
            y={density(
              0.5,
              sd,
              reject.filter((x) => x > 0.5),
            )}
            tone="destructive"
            opacity={0.25}
            line={false}
          />
          <Curve name="H₀: N(½, v)" x={xs} y={nullD} slot={0} />
          <Curve name="under these α, β" x={xs} y={density(etaAlt, sdAlt, xs)} slot={1} dashed />
          <Bars name="observed η̄" x={[result.meanEstimate]} y={[Math.max(...nullD)]} width={(hi - lo) / 200} emphasis />
        </Plot>
      </Plots>
    </Figure>
  )
}

const KS = [1, 2, 4, 8, 16]
type SimSettings = {
  layout: NoiseLayout
  n: number
  repeats: number
  alpha: number
  beta: number
  tau: number
  model: NoiseTestModel
  bandwidth: number
  delta: number
  seed: number
}

const simTask = (s: SimSettings): Task<NoiseTestSimulation> => {
  const cells = KS.flatMap((k) =>
    (['null', 'alt'] as const).map((which) => ({
      label: `${which} k=${k}`,
      datasets: Array.from({ length: s.repeats }, (_, r) =>
        call(
          'applied/data/synthetic/classConditionalNoise',
          call('foundation/random/stream', `lab/noise-sim/${s.seed}/${which}/${k}/${r}`),
          {
            n: s.n,
            layout: s.layout,
            alpha: which === 'null' ? s.tau : s.alpha,
            beta: which === 'null' ? s.tau : s.beta,
            truth: false,
          },
        ),
      ),
      anchors: Array.from({ length: s.repeats }, (_, r) =>
        call(
          'applied/data/synthetic/noiseLayoutAnchors',
          call('foundation/random/stream', `lab/noise-sim/anchors/${s.seed}/${k}/${r}`),
          s.layout,
          k,
          s.delta,
        ),
      ),
    })),
  )
  return call<NoiseTestSimulation>('applied/learning/weak-supervision/noiseTestSimulation', cells, {
    model: s.model,
    bandwidth: s.bandwidth,
    seed: s.seed,
  })
}

export function NoiseSimulationFigure() {
  const state = useFigureState({
    data: row('1 · data', {
      layout: choice(LAYOUTS, 'gaussians', { label: 'classes' }),
      n: int(500, { ge: 50, le: 5000, suggestions: [500, 1000, 2000], label: 'training points N' }),
      repeats: int(40, { ge: 5, le: 500, suggestions: [40, 100, 500], label: 'datasets per cell' }),
      seed: int(1, { ge: 0, le: 9999, label: 'seed' }),
    }),
    noise: row('2 · noise', {
      tau: float(0.1, { ge: 0, le: 0.45, suggestions: [0, 0.1, 0.2], label: 'uniform τ (null)' }),
      alpha: float(0, { ge: 0, le: 0.45, suggestions: [0, 0.1], label: 'α (alternative)' }),
      beta: float(0.1, { ge: 0, le: 0.45, suggestions: [0.05, 0.1, 0.2], label: 'β (alternative)' }),
    }),
    test: row('3 · test', {
      model: choice(MODELS, 'parametric', { label: 'model' }),
      bandwidth: float(1, {
        gt: 0,
        scale: 'log10',
        suggestions: [0.5, 1, 2],
        label: 'bandwidth h',
        when: when('model', 'local'),
      }),
      delta: float(0, { ge: 0, lt: 0.5, suggestions: [0, 0.05, 0.1], label: 'anchor relaxation δ' }),
    }),
  })
  const settings: SimSettings = {
    layout: state.data.layout as NoiseLayout,
    n: state.data.n,
    repeats: state.data.repeats,
    seed: state.data.seed,
    tau: state.noise.tau,
    alpha: state.noise.alpha,
    beta: state.noise.beta,
    model: state.test.model as NoiseTestModel,
    bandwidth: state.test.bandwidth,
    delta: state.test.delta,
  }
  const trained = useTrainedRun(settings, simTask)
  const r = trained.run.value
  const shown = trained.trained ?? settings
  const kx = useAxis({ label: 'anchors k', range: [0.8, 20], log: true })
  const ky = useAxis({ label: 'share of datasets rejected at 5%', range: [0, 1.02] })
  const hx = useAxis({ label: 'p-value under uniform noise (all k)', range: [0, 1] })
  const hy = useAxis({ label: 'datasets', hold: 'union', key: trained.trained })
  const nullRates = KS.map((_, i) => r?.rejected05[2 * i] ?? NaN)
  const altRates = KS.map((_, i) => r?.rejected05[2 * i + 1] ?? NaN)
  const nullP = r ? KS.flatMap((_, i) => r.pValues[2 * i]) : []
  const edges = Array.from({ length: 11 }, (_, i) => i / 10)
  const counts = edges.slice(0, 10).map((e) => nullP.filter((p) => p >= e && p < e + 0.1).length)
  return (
    <Figure
      title="Size and power by simulation"
      purpose="Under uniform noise the test should reject 5% of datasets at the 5% level (its p-values uniform); under class-conditional noise it should reject more often, and more anchors should make it reject more often still."
      state={state}
      defaultSize="L"
      controls={
        <TrainControls
          run={trained as never}
          progress={(r?.done ?? 0) / Math.max(1, r?.total ?? 1)}
          progressText={`${r?.done ?? 0} / ${r?.total ?? KS.length * 2 * shown.repeats} datasets`}
        />
      }
      readouts={
        r ? (
          <>
            {KS.map((k, i) => (
              <Readout key={k} label={`k = ${k}: size · power`} value={`${f3(nullRates[i])} · ${f3(altRates[i])}`} />
            ))}
          </>
        ) : null
      }
      caption={
        <>
          aifn <code>noiseTestSimulation</code>: for each k, {shown.repeats} datasets of N = {shown.n} under uniform
          noise τ = {shown.tau} (the null) and {shown.repeats} under α = {shown.alpha}, β = {shown.beta}, each with its
          own k anchors (δ = {shown.delta}), tested by the{' '}
          {shown.model === 'parametric' ? 'parametric' : 'local likelihood'} test. Left: rejection rates at 5% (size
          under the null, power under the alternative); the dashed line is the level. Right: the null p-values, which
          should be flat. With anchors drawn far from the data, the local test&apos;s neighbourhoods empty and its power
          can fall as k grows (Yang et al.&apos;s empty-neighbourhood problem).
        </>
      }
    >
      <Plots cols={2}>
        <Plot x={kx} y={ky} title={!trained.trained ? 'press Train to simulate' : undefined}>
          <Curve name="size (uniform noise)" x={KS} y={nullRates} slot={2} showPoints />
          <Curve name="power (class-conditional)" x={KS} y={altRates} slot={1} showPoints />
          <Curve name="level 5%" x={[0.8, 20]} y={[0.05, 0.05]} muted dashed thin />
        </Plot>
        <Plot x={hx} y={hy}>
          <Bars name="null p-values" x={edges.slice(0, 10).map((e) => e + 0.05)} y={counts} width={0.09} slot={2} />
        </Plot>
      </Plots>
    </Figure>
  )
}

void child
