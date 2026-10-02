import {
  autocorrelation,
  bootstrap,
  bootstrapInterval,
  correlation,
  ecdf,
  histogram,
  importanceEffectiveSampleSize,
  kde,
  kendallTau,
  median,
  quantile,
  quantileMethods,
  runningMean,
  runningVariance,
  spearman,
  type BinRule,
  type QuantileMethod,
} from 'aifn/probability/stats'
import { now } from 'aifn/foundation/trace'
import { useMemo } from 'react'
import { Normal } from 'aifn/probability/distributions'
import { child, normals, stream, uniform, type Stream } from 'aifn/foundation/random'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { Equation, Figure, live, tex } from '@lab/layout'
import { choice, row, slider, toggle, useFigureState, variants } from '@lab/state'
import { Annotation, Bars, Curve, Handle, Plot, Points, Readout, useAxis } from '@lab/viz'
import { histogramBars } from '@lab/views'

/** n standard normal draws from a stream, as a Float64Array for array arithmetic in the specimens. */
const standardNormals = (s: Stream, n: number) => Float64Array.from(toFlat(normals(s, n)))

/** A skewed sample: a mixture of two Gaussians (70% at 0 with sd 1, 30% at 3 with sd 0.6). */
function mixtureSample(n: number, seed: number): Float64Array {
  const s = stream(`mixture/${seed}`)
  const z = standardNormals(child(s, 'z'), n)
  const pick = child(s, 'pick')
  return z.map((v) => (uniform(pick) < 0.7 ? v : 3 + 0.6 * v))
}

/** n evenly spaced points from lo to hi, as numbers. */
const grid = (lo: number, hi: number, n: number) => toFlat(linspace(lo, hi, n))

// ---------------------------------------------------------------------------------------------------------------------

export function HistogramSpecimen() {
  const state = useFigureState({
    n: slider(20, 3000, 300, { step: 10, label: 'n' }),
    rule: variants(
      {
        'freedman-diaconis': { label: 'Freedman–Diaconis', params: {} },
        sturges: { label: 'Sturges', params: {} },
        count: { label: 'count', params: { count: slider(1, 80, 20, { step: 1, label: 'bins' }) } },
        width: { label: 'width', params: { width: slider(0.05, 2, 0.4, { step: 0.05, label: 'width' }) } },
      },
      { label: '1 · bins', choiceLabel: 'bin rule' },
    ),
    kde: row('2 · KDE', {
      bandwidth: choice(
        [
          { value: 'scott', label: 'Scott' },
          { value: 'silverman', label: 'Silverman' },
        ],
        'scott',
        { label: 'KDE bandwidth' },
      ),
    }),
  })
  const { n, rule } = state
  const bandwidth = state.kde.bandwidth
  const x = useMemo(() => mixtureSample(n, 1), [n])
  const h = useMemo(() => {
    const bins: BinRule =
      rule.key === 'count' ? rule.values.count : rule.key === 'width' ? { width: rule.values.width } : rule.key
    return histogram(x, { bins })
  }, [x, rule])
  const bars = useMemo(() => histogramBars(h), [h])
  const ts = useMemo(() => grid(-4, 6, 241), [])
  const k = useMemo(() => kde(x, ts, { bandwidth }), [x, ts, bandwidth])
  const kd = useMemo(() => toFlat(k.density), [k])
  const xa = useAxis({ label: 'x', range: [-4, 6] })
  const ya = useAxis({ label: 'density', hold: 'union', key: n })
  return (
    <Figure
      title="Histogram and KDE of a two-component mixture"
      purpose="A bin rule picks the bin width from the data (Sturges from n alone, Freedman–Diaconis from the IQR too); a kernel density estimate replaces the bins by a smooth bump per point of width h."
      state={state}
      readouts={
        <>
          <Readout label="bins" value={bars.x.length} />
          <Readout label="bin width" value={(bars.edges[1] - bars.edges[0]).toPrecision(3)} />
          <Readout label="KDE bandwidth h" value={k.bandwidth.toPrecision(3)} />
          <Readout label="dropped" value={h.dropped} />
        </>
      }
      caption="A skewed mixture: 70% N(0, 1) and 30% N(3, 0.6²). Sturges' rule grows with log n only, so with large n it over-smooths the second mode; Freedman–Diaconis narrows its bins as n grows. Silverman's bandwidth is a little narrower than Scott's."
    >
      <Plot x={xa} y={ya}>
        <Bars name="histogram (density)" x={bars.x} y={bars.density} edges={bars.edges} slot={0} />
        <Curve name={`KDE (${bandwidth})`} x={ts} y={kd} slot={1} />
      </Plot>
    </Figure>
  )
}

const QS = grid(0, 1, 401)

export function QuantileSpecimen() {
  const state = useFigureState({
    n: slider(1, 12, 5, { step: 1, label: 'n' }),
    method: choice(quantileMethods, 'inverted-cdf', { label: 'method (numpy.quantile)' }),
  })
  const n = state.n
  const method = state.method as QuantileMethod
  const x = useMemo(() => Float64Array.from(mixtureSample(n, 7)).sort(), [n])
  const lines = useMemo(() => {
    const e = ecdf(x)
    return {
      px: toFlat(e.probabilities),
      py: toFlat(e.values),
      chosen: toFlat(quantile(x, QS, method)),
      linear: toFlat(quantile(x, QS, 'linear')),
    }
  }, [x, method])
  const xa = useAxis({ label: 'q', range: [0, 1] })
  const ya = useAxis({ label: 'Q(q)', hold: 'union', key: n })
  return (
    <Figure
      title="Sample quantile functions"
      purpose="The thirteen numpy.quantile methods differ only in how they place and interpolate the order statistics; they disagree most for small n and near q = 0 and 1."
      state={state}
      readouts={<Readout label="median" value={quantile(x, 0.5, method).toPrecision(4)} />}
      caption="The dots are the order statistics x₍ₖ₎ at q = k/n (the empirical cdf's jumps). Step methods (inverted-cdf, closest-observation, …) jump between them; the continuous ones interpolate. The dashed line is linear, the default."
    >
      <Plot x={xa} y={ya}>
        <Points name="order statistics at k/n" x={lines.px} y={lines.py} muted />
        <Curve name={method} x={QS} y={lines.chosen} slot={0} />
        {method !== 'linear' && <Curve name="linear" x={QS} y={lines.linear} slot={1} dashed />}
      </Plot>
    </Figure>
  )
}

const MAX_LAG = 40
const LAGS = Array.from({ length: MAX_LAG + 1 }, (_, k) => k)

export function AutocorrelationSpecimen() {
  const state = useFigureState({
    phi: slider(-0.95, 0.95, 0.8, { label: 'φ' }),
    n: slider(200, 20000, 2000, { step: 200, label: 'n' }),
  })
  const { phi, n } = state
  const series = useMemo(() => {
    // AR(1): xₜ = φ xₜ₋₁ + εₜ, whose autocorrelation is φ^k.
    const e = standardNormals(stream('ar1'), n)
    const x = new Float64Array(n)
    x[0] = e[0] / Math.sqrt(1 - phi * phi)
    for (let t = 1; t < n; t++) x[t] = phi * x[t - 1] + e[t]
    return x
  }, [phi, n])
  const result = useMemo(() => {
    const t0 = now()
    const direct = toFlat(autocorrelation(series, { maxLag: n - 1, method: 'direct' }))
    const t1 = now()
    const fft = toFlat(autocorrelation(series, { maxLag: n - 1, method: 'fft' }))
    const t2 = now()
    let maxDiff = 0
    for (let k = 0; k < n; k++) maxDiff = Math.max(maxDiff, Math.abs(direct[k] - fft[k]))
    return { acf: Array.from(fft.slice(0, MAX_LAG + 1)), directMs: t1 - t0, fftMs: t2 - t1, maxDiff }
  }, [series, n])
  const theory = useMemo(() => LAGS.map((k) => phi ** k), [phi])
  const xa = useAxis({ label: 'lag k', range: [-0.5, MAX_LAG + 0.5] })
  const ya = useAxis({ label: 'ρ̂(k)', range: [-1, 1] })
  return (
    <Figure
      title="Sample autocorrelation of an AR(1) series"
      purpose="An AR(1) series has autocorrelation φ^k; the FFT computes all n sample lags in O(n log n) and gives the same values as the O(n²) direct sum."
      state={state}
      readouts={
        <>
          <Readout label="all lags, direct" value={`${result.directMs.toFixed(1)} ms`} />
          <Readout label="all lags, FFT" value={`${result.fftMs.toFixed(1)} ms`} />
          <Readout label="max |direct − FFT|" value={result.maxDiff.toExponential(1)} />
        </>
      }
      caption="xₜ = φ xₜ₋₁ + εₜ. Negative φ alternates the sign of the ACF. Raise n to 20 000 and the direct method's time grows a hundredfold while the FFT's barely moves."
    >
      <Plot x={xa} y={ya}>
        <Bars name="sample ACF" x={LAGS} y={result.acf} slot={0} />
        <Curve name="φ^k" x={LAGS} y={theory} slot={1} dashed />
      </Plot>
    </Figure>
  )
}

export function RunningMomentsSpecimen() {
  const state = useFigureState({ logOffset: slider(0, 12, 8, { step: 1, label: 'offset (log₁₀), to test accuracy' }) })
  const offset = state.logOffset === 0 ? 0 : 10 ** state.logOffset
  const n = 2000
  const { mean, sd, steps } = useMemo(() => {
    const x = standardNormals(stream('welford'), n).map((v) => offset + 2 * v)
    const m = runningMean(x)
    const v = runningVariance(x, { sample: true })
    return {
      steps: Array.from({ length: n }, (_, i) => i + 1),
      mean: toFlat(m).map((v) => v - offset),
      sd: toFlat(v).map(Math.sqrt),
    }
  }, [offset])
  const xa = useAxis({ label: 'n' })
  const ya = useAxis({ label: 'value', range: [-1, 4] })
  return (
    <Figure
      title="Running mean and standard deviation"
      purpose="Welford's update keeps the running variance accurate when every value carries a large offset, where the textbook Σx² − (Σx)²/n would cancel catastrophically."
      state={state}
      readouts={<Readout label="final sd" value={sd[n - 1].toPrecision(6)} />}
      caption="2000 draws of offset + 2z. The defaults add 10⁸ to every value and the running sd still settles on 2. Push it to 10¹²: each value then keeps only about four decimal places, and the running sd still settles on 2."
    >
      <Plot x={xa} y={ya}>
        <Curve name="running mean − offset" x={steps} y={mean} slot={0} />
        <Curve name="running sd (n − 1)" x={steps} y={sd} slot={1} />
        <Annotation y={2} text="true sd" dashed />
      </Plot>
    </Figure>
  )
}

export function BootstrapSpecimen() {
  const state = useFigureState({
    data: row('1 · sample', { n: slider(5, 400, 40, { step: 5, label: 'sample size' }) }),
    boot: row('2 · bootstrap', {
      resamples: slider(100, 20000, 2000, { step: 100, label: 'resamples' }),
      statistic: choice(['median', 'mean'], 'median', { label: 'statistic' }),
    }),
  })
  const n = state.data.n
  const { resamples, statistic } = state.boot
  const x = useMemo(() => mixtureSample(n, 3), [n])
  const b = useMemo(() => {
    const s = stream('bootstrap')
    const stat = statistic === 'median' ? median : (v: Float64Array) => v.reduce((a, c) => a + c, 0) / v.length
    return bootstrap(s, x, stat, resamples)
  }, [x, resamples, statistic])
  const [lo, hi] = bootstrapInterval(b)
  const [blo, bhi] = bootstrapInterval(b, { method: 'basic' })
  const bars = useMemo(() => histogramBars(histogram(b.replicates, { bins: 'freedman-diaconis' })), [b])
  const xa = useAxis({ label: `bootstrap ${statistic}` })
  const ya = useAxis({ label: 'density' })
  return (
    <Figure
      title="Bootstrap distribution with 95% percentile interval"
      purpose="Resampling the data with replacement and recomputing the statistic approximates its sampling distribution, whose spread gives a standard error and intervals."
      state={state}
      readouts={{
        estimate: (
          <>
            <Readout label="estimate" value={b.estimate.toPrecision(4)} />
            <Readout label="standard error" value={b.standardError.toPrecision(3)} />
            <Readout label="bias" value={b.bias.toPrecision(2)} />
          </>
        ),
        '95% intervals': (
          <>
            <Readout label="percentile" value={`[${lo.toPrecision(3)}, ${hi.toPrecision(3)}]`} />
            <Readout label="basic" value={`[${blo.toPrecision(3)}, ${bhi.toPrecision(3)}]`} />
          </>
        ),
      }}
      caption="The percentile interval (dashed lines) is the 2.5% and 97.5% quantiles of the replicates; the basic interval reflects them about the estimate. The median of a small sample has a lumpy bootstrap distribution, because it can only take a few sample values; the mean's is smooth."
    >
      <Plot x={xa} y={ya}>
        <Bars name="replicates" x={bars.x} y={bars.density} edges={bars.edges} slot={0} />
        <Annotation x={lo} dashed />
        <Annotation x={hi} dashed />
        <Points name="estimate" x={[b.estimate]} y={[0]} emphasis />
      </Plot>
    </Figure>
  )
}

export function RankCorrelationSpecimen() {
  const state = useFigureState({
    strength: slider(0, 1, 0.7, { label: 'signal share' }),
    outlier: toggle(true, 'one outlier'),
    ox: slider(-3, 4, 3.2, { onChart: true }),
    oy: slider(-8, 12, -6, { onChart: true }),
  })
  const { strength, outlier, ox, oy } = state
  const { x, y } = useMemo(() => {
    const s = stream('ranks')
    const a = standardNormals(child(s, 'a'), 60)
    const e = standardNormals(child(s, 'e'), 60)
    // A monotone but nonlinear relation, y = exp(x) with noise.
    const x = Array.from(a)
    return { x, y: x.map((v, i) => Math.exp(strength * v + (1 - strength) * e[i])) }
  }, [strength])
  const all = useMemo(() => (outlier ? { x: [...x, ox], y: [...y, oy] } : { x, y }), [x, y, outlier, ox, oy])
  const xa = useAxis({ label: 'x', range: [-3, 4] })
  const ya = useAxis({ label: 'y', range: [-8, 12] })
  return (
    <Figure
      title="y = exp(x) with noise"
      purpose="Rank correlations measure any monotone relation, not only a linear one, and one outlier moves Pearson's r far more than Spearman's ρ or Kendall's τ."
      state={state}
      readouts={
        <>
          <Readout label="Pearson r" value={correlation(all.x, all.y).toFixed(3)} />
          <Readout label="Spearman ρ" value={spearman(all.x, all.y).toFixed(3)} />
          <Readout label="Kendall τ_b" value={kendallTau(all.x, all.y).toFixed(3)} />
        </>
      }
      caption="Drag the outlier (ink): Pearson swings with its value, the rank correlations only with its rank. Switch the outlier off to compare; at signal share 1 the relation is exactly monotone and Spearman and Kendall are 1 while Pearson is not."
    >
      <Plot x={xa} y={ya}>
        <Points name="points" x={x} y={y} slot={0} />
        {outlier && <Points name="outlier" x={[ox]} y={[oy]} emphasis live />}
        {outlier && <Handle {...state.handle(['ox', 'oy'], { label: 'outlier' })} />}
      </Plot>
    </Figure>
  )
}

export function ImportanceSpecimen() {
  const state = useFigureState({ shift: slider(0, 4, 1, { label: 'proposal shift' }) })
  const shift = state.shift
  const n = 2000
  const { ess, weights, xs } = useMemo(() => {
    // Target N(0, 1), proposal N(shift, 1): log w = −x²/2 + (x − shift)²/2.
    const z = standardNormals(stream('importance'), n)
    const xs = Array.from(z, (v) => v + shift)
    const [target, proposal] = [Normal(0, 1), Normal(shift, 1)]
    const logW = xs.map((x) => target.logProb(x) - proposal.logProb(x))
    const ess = importanceEffectiveSampleSize(logW, { log: true })
    const top = Math.max(...logW)
    return { ess, weights: logW.map((l) => Math.exp(l - top)), xs }
  }, [shift])
  const theory = Math.exp(-shift * shift) * n
  const xa = useAxis({ label: 'draw x', range: [-4, 8] })
  const ya = useAxis({ label: 'weight / max weight', range: [0, 1] })
  return (
    <Figure
      title="Importance weights, target N(0, 1), proposal N(shift, 1)"
      purpose="As the proposal moves away from the target, a few draws take almost all the weight, and Kish's effective sample size falls like n·exp(−shift²)."
      state={state}
      equation={
        <Equation>
          {tex`\mathrm{ESS} = \frac{(\sum_i w_i)^2}{\sum_i w_i^2} = ${live(ess, { digits: 4, strong: true })} \quad\text{against}\quad n\, e^{-\text{shift}^2} = ${live(theory, { digits: 4 })}`}
        </Equation>
      }
      caption="2000 draws from N(shift, 1) weighted by p(x)/q(x) for the target p = N(0, 1), scaled so the largest weight is 1. At shift 0 every weight is 1; at shift 3 a handful of draws in the proposal's left tail carry the estimate."
    >
      <Plot x={xa} y={ya}>
        <Points name="importance weights" x={xs} y={weights} slot={0} thin size={4} />
      </Plot>
    </Figure>
  )
}
