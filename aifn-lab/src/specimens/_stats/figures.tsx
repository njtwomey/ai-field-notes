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
} from 'aifn/stats'
import { now } from 'aifn/trace'
import { useMemo, useState } from 'react'
import { Normal } from 'aifn/distributions'
import { normals, stream, type Stream } from 'aifn/random'
import { linspace, toFlat } from 'aifn/tensor'
import { Select, Slider, Switch } from '@lab/controls'
import { Figure } from '@lab/layout'
import { Readout, XYChart, type XYSeries } from '@lab/viz'

/** n standard normal draws from a stream, as a Float64Array for array arithmetic in the specimens. */
const standardNormals = (s: Stream, n: number) => Float64Array.from(toFlat(normals(s, n)))

/** A skewed sample: a mixture of two Gaussians (70% at 0 with sd 1, 30% at 3 with sd 0.6). */
function mixtureSample(n: number, seed: number): Float64Array {
  const s = stream(`mixture/${seed}`)
  const z = standardNormals(s.child('z'), n)
  const pick = s.child('pick')
  return z.map((v) => (pick.uniform() < 0.7 ? v : 3 + 0.6 * v))
}

/** n evenly spaced points from lo to hi, as numbers. */
const grid = (lo: number, hi: number, n: number) => toFlat(linspace(lo, hi, n))

// ---------------------------------------------------------------------------------------------------------------------

type RuleName = 'count' | 'width' | 'sturges' | 'freedman-diaconis'

export function HistogramSpecimen() {
  const [n, setN] = useState(300)
  const [rule, setRule] = useState<RuleName>('freedman-diaconis')
  const [count, setCount] = useState(20)
  const [width, setWidth] = useState(0.4)
  const [bandwidth, setBandwidth] = useState<'scott' | 'silverman'>('scott')
  const x = useMemo(() => mixtureSample(n, 1), [n])
  const h = useMemo(() => {
    const bins: BinRule = rule === 'count' ? count : rule === 'width' ? { width } : rule
    return histogram(x, { bins })
  }, [x, rule, count, width])
  const ts = useMemo(() => grid(-4, 6, 241), [])
  const k = useMemo(() => kde(x, ts, { bandwidth }), [x, ts, bandwidth])
  const series = useMemo(
    (): XYSeries[] => [
      {
        name: 'histogram (density)',
        type: 'bar',
        x: Array.from({ length: h.counts.length }, (_, i) => (h.edges[i] + h.edges[i + 1]) / 2),
        y: Array.from(h.density),
        slot: 0,
      },
      { name: `KDE (${bandwidth})`, type: 'line', x: ts, y: Array.from(k.density), slot: 1 },
    ],
    [h, k, ts, bandwidth],
  )
  return (
    <Figure
      title="Histogram and KDE of a two-component mixture"
      controls={
        <>
          <Slider label="n" value={n} min={20} max={3000} step={10} onChange={setN} />
          <Select
            label="bin rule"
            value={rule}
            onChange={setRule}
            options={[
              { value: 'count', label: 'count' },
              { value: 'width', label: 'width' },
              { value: 'sturges', label: 'Sturges' },
              { value: 'freedman-diaconis', label: 'F–D' },
            ]}
          />
          {rule === 'count' && <Slider label="bins" value={count} min={1} max={80} step={1} onChange={setCount} />}
          {rule === 'width' && <Slider label="width" value={width} min={0.05} max={2} onChange={setWidth} />}
          <Select
            label="KDE bandwidth"
            value={bandwidth}
            onChange={setBandwidth}
            options={[
              { value: 'scott', label: 'Scott' },
              { value: 'silverman', label: 'Silverman' },
            ]}
          />
        </>
      }
      readouts={
        <>
          <Readout label="bins" value={h.counts.length} />
          <Readout label="bin width" value={(h.edges[1] - h.edges[0]).toPrecision(3)} />
          <Readout label="KDE bandwidth h" value={k.bandwidth.toPrecision(3)} />
          <Readout label="dropped" value={h.dropped} />
        </>
      }
    >
      <XYChart series={series} xLabel="x" yLabel="density" xRange={[-4, 6]} />
    </Figure>
  )
}

export function QuantileSpecimen() {
  const [n, setN] = useState(5)
  const [method, setMethod] = useState<QuantileMethod>('linear')
  const x = useMemo(() => Float64Array.from(mixtureSample(n, 7)).sort(), [n])
  const qs = useMemo(() => grid(0, 1, 401), [])
  const series = useMemo((): XYSeries[] => {
    const e = ecdf(x)
    const out: XYSeries[] = [
      {
        name: 'order statistics at k/n',
        type: 'scatter',
        x: toFlat(e.probabilities),
        y: toFlat(e.values),
        muted: true,
      },
      { name: method, type: 'line', x: qs, y: toFlat(quantile(x, qs, method)), slot: 0 },
    ]
    if (method !== 'linear')
      out.push({ name: 'linear', type: 'line', x: qs, y: toFlat(quantile(x, qs, 'linear')), slot: 1, dashed: true })
    return out
  }, [x, qs, method])
  return (
    <Figure
      title="Sample quantile functions"
      controls={
        <>
          <Slider label="n" value={n} min={1} max={12} step={1} onChange={setN} />
          <Select label="method (numpy.quantile)" value={method} onChange={setMethod} options={quantileMethods} />
        </>
      }
      readouts={<Readout label="median" value={quantile(x, 0.5, method).toPrecision(4)} />}
    >
      <XYChart series={series} xLabel="q" yLabel="Q(q)" xRange={[0, 1]} />
    </Figure>
  )
}

export function AutocorrelationSpecimen() {
  const [phi, setPhi] = useState(0.8)
  const [n, setN] = useState(2000)
  const series = useMemo(() => {
    // AR(1): xₜ = φ xₜ₋₁ + εₜ, whose autocorrelation is φ^k.
    const e = standardNormals(stream('ar1'), n)
    const x = new Float64Array(n)
    x[0] = e[0] / Math.sqrt(1 - phi * phi)
    for (let t = 1; t < n; t++) x[t] = phi * x[t - 1] + e[t]
    return x
  }, [phi, n])
  const maxLag = 40
  const result = useMemo(() => {
    const t0 = now()
    const direct = toFlat(autocorrelation(series, { maxLag: n - 1, method: 'direct' }))
    const t1 = now()
    const fft = toFlat(autocorrelation(series, { maxLag: n - 1, method: 'fft' }))
    const t2 = now()
    let maxDiff = 0
    for (let k = 0; k < n; k++) maxDiff = Math.max(maxDiff, Math.abs(direct[k] - fft[k]))
    return { acf: fft.slice(0, maxLag + 1), directMs: t1 - t0, fftMs: t2 - t1, maxDiff }
  }, [series, n])
  const lags = useMemo(() => Array.from({ length: maxLag + 1 }, (_, k) => k), [])
  return (
    <Figure
      title="Sample autocorrelation of an AR(1) series"
      controls={
        <>
          <Slider label="φ" value={phi} min={-0.95} max={0.95} onChange={setPhi} />
          <Slider label="n" value={n} min={200} max={20000} step={200} onChange={setN} />
        </>
      }
      readouts={
        <>
          <Readout label="all lags, direct" value={`${result.directMs.toFixed(1)} ms`} />
          <Readout label="all lags, FFT" value={`${result.fftMs.toFixed(1)} ms`} />
          <Readout label="max |direct − FFT|" value={result.maxDiff.toExponential(1)} />
        </>
      }
    >
      <XYChart
        integerX
        xLabel="lag k"
        yLabel="ρ̂(k)"
        yRange={[-1, 1]}
        series={[
          { name: 'sample ACF', type: 'bar', x: lags, y: Array.from(result.acf), slot: 0 },
          { name: 'φ^k', type: 'line', x: lags, y: lags.map((k) => phi ** k), slot: 1, dashed: true },
        ]}
      />
    </Figure>
  )
}

export function RunningMomentsSpecimen() {
  const [offset, setOffset] = useState(0)
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
  return (
    <Figure
      title="Running mean and standard deviation"
      controls={
        <Slider
          label="offset (log₁₀), to test accuracy"
          value={offset === 0 ? 0 : Math.log10(offset)}
          min={0}
          max={12}
          step={1}
          onChange={(v) => setOffset(v === 0 ? 0 : 10 ** v)}
        />
      }
      readouts={<Readout label="final sd" value={sd[n - 1].toPrecision(6)} />}
    >
      <XYChart
        xLabel="n"
        yRange={[-1, 4]}
        series={[
          { name: 'running mean − offset', type: 'line', x: steps, y: mean, slot: 0 },
          { name: 'running sd (n − 1)', type: 'line', x: steps, y: sd, slot: 1 },
          { name: 'true sd', type: 'line', x: [1, n], y: [2, 2], slot: 1, dashed: true },
        ]}
      />
    </Figure>
  )
}

export function BootstrapSpecimen() {
  const [n, setN] = useState(40)
  const [resamples, setResamples] = useState(2000)
  const [statistic, setStatistic] = useState<'median' | 'mean'>('median')
  const x = useMemo(() => mixtureSample(n, 3), [n])
  const b = useMemo(() => {
    const s = stream('bootstrap')
    const stat = statistic === 'median' ? median : (v: Float64Array) => v.reduce((a, c) => a + c, 0) / v.length
    return bootstrap(s, x, stat, resamples)
  }, [x, resamples, statistic])
  const [lo, hi] = bootstrapInterval(b)
  const [blo, bhi] = bootstrapInterval(b, { method: 'basic' })
  const h = useMemo(() => histogram(b.replicates, { bins: 'freedman-diaconis' }), [b])
  const top = Math.max(...h.density)
  return (
    <Figure
      title="Bootstrap distribution with 95% percentile interval"
      controls={
        <>
          <Slider label="sample size" value={n} min={5} max={400} step={5} onChange={setN} />
          <Slider label="resamples" value={resamples} min={100} max={20000} step={100} onChange={setResamples} />
          <Select label="statistic" value={statistic} onChange={setStatistic} options={['median', 'mean']} />
        </>
      }
      readouts={
        <>
          <Readout label="estimate" value={b.estimate.toPrecision(4)} />
          <Readout label="standard error" value={b.standardError.toPrecision(3)} />
          <Readout label="bias" value={b.bias.toPrecision(2)} />
          <Readout label="95% percentile" value={`[${lo.toPrecision(3)}, ${hi.toPrecision(3)}]`} />
          <Readout label="95% basic" value={`[${blo.toPrecision(3)}, ${bhi.toPrecision(3)}]`} />
        </>
      }
    >
      <XYChart
        xLabel={`bootstrap ${statistic}`}
        yLabel="density"
        segments={[
          { from: [lo, 0], to: [lo, top] },
          { from: [hi, 0], to: [hi, top] },
        ]}
        series={[
          {
            name: 'replicates',
            type: 'bar',
            x: Array.from({ length: h.counts.length }, (_, i) => (h.edges[i] + h.edges[i + 1]) / 2),
            y: Array.from(h.density),
            slot: 0,
          },
          { name: 'estimate', type: 'scatter', x: [b.estimate], y: [0], emphasis: true },
        ]}
      />
    </Figure>
  )
}

export function RankCorrelationSpecimen() {
  const [outlier, setOutlier] = useState(true)
  const [strength, setStrength] = useState(0.7)
  const { x, y } = useMemo(() => {
    const s = stream('ranks')
    const a = standardNormals(s.child('a'), 60)
    const e = standardNormals(s.child('e'), 60)
    // A monotone but nonlinear relation, y = exp(x) with noise.
    const x = Array.from(a)
    const y = x.map((v, i) => Math.exp(strength * v + (1 - strength) * e[i]))
    if (outlier) {
      x.push(3.2)
      y.push(-6)
    }
    return { x, y }
  }, [outlier, strength])
  return (
    <Figure
      title="y = exp(x) with noise"
      controls={
        <>
          <Slider label="signal share" value={strength} min={0} max={1} onChange={setStrength} />
          <Switch label="add one outlier" checked={outlier} onChange={setOutlier} />
        </>
      }
      readouts={
        <>
          <Readout label="Pearson r" value={correlation(x, y).toFixed(3)} />
          <Readout label="Spearman ρ" value={spearman(x, y).toFixed(3)} />
          <Readout label="Kendall τ_b" value={kendallTau(x, y).toFixed(3)} />
        </>
      }
    >
      <XYChart xLabel="x" yLabel="y" series={[{ name: 'points', type: 'scatter', x, y, slot: 0 }]} />
    </Figure>
  )
}

export function ImportanceSpecimen() {
  const [shift, setShift] = useState(1)
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
  return (
    <Figure
      title="Importance weights, target N(0, 1), proposal N(shift, 1)"
      controls={<Slider label="proposal shift" value={shift} min={0} max={4} onChange={setShift} />}
      readouts={
        <>
          <Readout label="ESS" value={`${ess.toFixed(1)} of ${n}`} />
          <Readout label="theory exp(−shift²)·n" value={(Math.exp(-shift * shift) * n).toFixed(1)} />
        </>
      }
    >
      <XYChart
        xLabel="draw x"
        yLabel="weight / max weight"
        series={[{ name: 'importance weights', type: 'scatter', x: xs, y: weights, slot: 0 }]}
      />
    </Figure>
  )
}
