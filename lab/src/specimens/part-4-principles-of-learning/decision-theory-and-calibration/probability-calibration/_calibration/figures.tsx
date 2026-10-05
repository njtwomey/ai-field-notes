/**
 * Calibration and conformal prediction on model outputs with a known truth from `aifn-methods/data`
 * (`classifierOutputs`, `quantileModelOutputs`): reliability diagrams before and after each calibration map of
 * `aifn/learning/calibration`, and the coverage and size of split-conformal, CQR and classification sets from
 * `aifn/learning/conformal`, with the Beta law of coverage over calibration draws. Each data set is split into a
 * calibration half (where maps and quantiles are fitted) and a test half (where they are judged).
 */
import { useMemo } from 'react'
import { classifierOutputs, quantileModelOutputs } from 'aifn-methods/data/synthetic'
import { child, stream } from 'aifn-compute/foundation/random'
import { tensor, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import {
  betaCalibration,
  dirichletCalibration,
  histogramBinning,
  isotonicCalibration,
  plattScaling,
  temperatureScaling,
  topLabelConfidence,
} from 'aifn-compute/learning/calibration'
import {
  conformalClassification,
  conformalisedQuantileRegression,
  intervalCoverage,
  setCoverage,
  splitConformalRegression,
  type ClassificationScore,
} from 'aifn-compute/learning/conformal'
import { expectedCalibrationError, maximumCalibrationError, reliabilityDiagram } from 'aifn-compute/learning/metrics'
import { softmax } from 'aifn-compute/numerics/special'
import { Beta } from 'aifn-compute/probability/distributions'
import { Figure } from 'aifn-render/layout'
import { choice, float, int, row, slider, useComputed, useFigureState } from 'aifn-render/state'
import {
  Annotation,
  Area,
  Bars,
  Curve,
  Density,
  formatNumber,
  Histogram,
  Plot,
  Plots,
  Points,
  Readout,
  useAxis,
} from 'aifn-render/viz'

const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(digits))) : '—')
const flat = (t: Tensor) => Array.from(toFlat(t))

/** Rows [from, to) of a matrix as rows, and the matching labels. */
const rowsOf = (t: Tensor, from: number, to: number) => (toRows(t) as number[][]).slice(from, to)

// ── 1 · Reliability before and after each map ────────────────────────────────────────────────────────────────────────

const MAPS = [
  { value: 'temperature', label: 'temperature scaling' },
  { value: 'dirichlet', label: 'Dirichlet calibration' },
  { value: 'histogram', label: 'histogram binning (top label)' },
  { value: 'isotonic', label: 'isotonic (top label)' },
  { value: 'beta', label: 'beta calibration (top label)' },
  { value: 'platt', label: 'Platt scaling (top label)' },
] as const
type MapName = (typeof MAPS)[number]['value']

export function ReliabilityMapsSpecimen() {
  const state = useFigureState({
    model: row('1 · model', {
      classes: int(4, { ge: 2, le: 20, suggestions: [2, 4, 10], label: 'classes K' }),
      temperature: float(2.5, {
        gt: 0,
        scale: 'log10',
        suggestions: [0.5, 1, 2.5, 5],
        label: 'overconfidence (logits ×)',
      }),
      bias: slider(-1.5, 1.5, 0, { step: 0.05, label: 'per-class offset' }),
      n: int(2000, { ge: 100, le: 50000, suggestions: [500, 2000, 10000], label: 'cases (half to calibrate)' }),
      seed: slider(0, 20, 0, { step: 1, label: 'seed' }),
    }),
    map: row('2 · map', {
      map: choice(MAPS, 'temperature', { label: 'calibration map' }),
      bins: int(12, { ge: 2, le: 50, suggestions: [5, 10, 15, 20], label: 'bins M' }),
    }),
  })
  const { classes: K, temperature, bias, n, seed } = state.model
  const { map, bins } = state.map
  const result = useComputed(
    () => {
      const d = classifierOutputs(stream(`calibration/outputs/${seed}`), { n, classes: K, temperature, bias })
      const half = Math.floor(n / 2)
      const y = flat(d.y!)
      const calZ = rowsOf(d.x, 0, half)
      const testZ = rowsOf(d.x, half, n)
      const calY = y.slice(0, half)
      const testY = y.slice(half)
      const calP = softmax(tensor(calZ))
      const testP = softmax(tensor(testZ))
      const before = topLabelConfidence(testP, testY)
      const cal = topLabelConfidence(calP, calY)
      let after: number[]
      const m = map as MapName
      if (m === 'temperature' || m === 'dirichlet') {
        const P =
          m === 'temperature'
            ? temperatureScaling(calZ, calY).apply(testZ)
            : dirichletCalibration(calP, calY).apply(testP)
        after = flat(topLabelConfidence(P, testY).confidence)
      } else {
        const s = flat(cal.confidence)
        const c = flat(cal.correct)
        const t = flat(before.confidence)
        if (m === 'histogram') after = flat(histogramBinning(s, c, { bins }).apply(t))
        else if (m === 'isotonic') after = flat(isotonicCalibration(s, c).apply(t))
        else if (m === 'beta') after = flat(betaCalibration(s, c).apply(t))
        else after = flat(plattScaling(tensor(s), tensor(c)).probability(tensor(t)))
      }
      const correct = flat(before.correct)
      const conf = flat(before.confidence)
      return {
        correct,
        conf,
        after,
        dBefore: reliabilityDiagram(correct, conf, { bins }),
        dAfter: reliabilityDiagram(correct, after, { bins }),
        eceBefore: expectedCalibrationError(correct, conf, { bins }),
        eceAfter: expectedCalibrationError(correct, after, { bins }),
        mceBefore: maximumCalibrationError(correct, conf, { bins }),
        mceAfter: maximumCalibrationError(correct, after, { bins }),
        accuracy: correct.reduce((a, b) => a + b, 0) / correct.length,
      }
    },
    [K, temperature, bias, n, seed, map, bins],
    { mode: 'release' },
  )
  const r = result.value
  const valid = (d: typeof r.dBefore) => {
    const x = flat(d.x)
    const yv = flat(d.y)
    const keep = x.map((_, i) => Number.isFinite(x[i]) && Number.isFinite(yv[i]))
    return { x: x.filter((_, i) => keep[i]), y: yv.filter((_, i) => keep[i]) }
  }
  const b = valid(r.dBefore)
  const a = valid(r.dAfter)
  const confAxis = useAxis({ label: 'top-label confidence', range: [0, 1] })
  const accAxis = useAxis({ label: 'accuracy in the bin', range: [0, 1] })
  const countAxis = useAxis({ label: 'cases', hold: 'union', key: `${n}/${bins}` })
  const edges = Array.from({ length: bins + 1 }, (_, k) => k / bins)
  return (
    <Figure
      title="Reliability before and after a calibration map"
      purpose="A calibrated classifier is right a fraction p of the time when it says p, so its reliability diagram follows the diagonal; a map fitted on held-out cases moves an overconfident model's points back onto it."
      state={state}
      defaultSize="L"
      readouts={{
        'test half': (
          <>
            <Readout label="accuracy" value={fmt(r.accuracy)} />
            <Readout label="ECE before → after" value={`${fmt(r.eceBefore)} → ${fmt(r.eceAfter)}`} />
            <Readout label="MCE before → after" value={`${fmt(r.mceBefore)} → ${fmt(r.mceAfter)}`} />
          </>
        ),
      }}
      caption="The model's logits are the true logits times the overconfidence factor (plus per-class offsets), so its confidence runs ahead of its accuracy and the grey points fall below the diagonal. Every map is fitted on the calibration half and judged on the test half. Temperature scaling undoes a pure scale error exactly and never changes the predicted class; with per-class offsets a Dirichlet map does better. The top-label maps act on the confidence only: histogram binning and isotonic are free-form steps, beta and Platt smooth curves. Recomputed on release."
    >
      <Plots rows={2} heights={[72, 28]} hoverGroup>
        <Plot x={confAxis} y={accAxis}>
          <Curve name="perfect calibration" x={[0, 1]} y={[0, 1]} dashed emphasis />
          <Curve name="before" x={b.x} y={b.y} muted showPoints stale={result.stale} />
          <Curve name="after" x={a.x} y={a.y} slot={0} showPoints width={2.5} stale={result.stale} />
        </Plot>
        <Plot x={confAxis} y={countAxis}>
          <Histogram name="confidence before" values={r.conf} bins={edges} range={[0, 1]} normalize="count" muted />
          <Histogram name="confidence after" values={r.after} bins={edges} range={[0, 1]} normalize="count" slot={0} />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── 2 · Conformal intervals for regression ───────────────────────────────────────────────────────────────────────────

const BANDS = [
  { value: 'model', label: 'the model’s own interval' },
  { value: 'split', label: 'split conformal: ŷ ± q̂' },
  { value: 'cqr', label: 'conformalised quantile regression' },
] as const

export function ConformalRegressionSpecimen() {
  const state = useFigureState({
    data: row('1 · data and model', {
      heteroscedastic: slider(0, 6, 3, { step: 0.1, label: 'noise growth to the right' }),
      scale: slider(0.2, 2, 0.6, { step: 0.05, label: 'model interval width (× right)' }),
      homoscedastic: slider(0, 1, 0, { step: 0.05, label: 'model ignores the growth' }),
      calibration: int(200, { ge: 5, le: 20000, suggestions: [20, 100, 200, 1000], label: 'calibration cases n' }),
      seed: slider(0, 20, 0, { step: 1, label: 'seed' }),
    }),
    conformal: row('2 · conformal', {
      alpha: float(0.1, { gt: 0, lt: 1, suggestions: [0.05, 0.1, 0.2], label: 'miscoverage α' }),
      band: choice(BANDS, 'cqr', { label: 'show' }),
    }),
  })
  const { heteroscedastic, scale, homoscedastic, calibration: nCal, seed } = state.data
  const { alpha, band } = state.conformal
  const nTest = 600
  const r = useMemo(() => {
    const d = quantileModelOutputs(stream(`conformal/regression/${seed}`), {
      n: nCal + nTest,
      heteroscedastic,
      scale,
      homoscedastic,
      alpha,
    })
    const x = flat(d.x)
    const y = flat(d.y!)
    const f = flat(d.prediction)
    const lo = flat(d.lower)
    const hi = flat(d.upper)
    const cal = {
      targets: y.slice(0, nCal),
      predictions: f.slice(0, nCal),
      lower: lo.slice(0, nCal),
      upper: hi.slice(0, nCal),
    }
    const order = Array.from({ length: nTest }, (_, i) => nCal + i).sort((i, j) => x[i] - x[j])
    const tx = order.map((i) => x[i])
    const ty = order.map((i) => y[i])
    const tf = order.map((i) => f[i])
    const tlo = order.map((i) => lo[i])
    const thi = order.map((i) => hi[i])
    const split = splitConformalRegression(cal, tf, alpha)
    const cqr = conformalisedQuantileRegression(cal, { lower: tlo, upper: thi }, alpha)
    const bands = {
      model: { lower: tlo, upper: thi },
      split: { lower: flat(split.lower), upper: flat(split.upper) },
      cqr: { lower: flat(cqr.lower), upper: flat(cqr.upper) },
    }
    const coverage = Object.fromEntries(
      (Object.keys(bands) as (keyof typeof bands)[]).map((k) => [
        k,
        intervalCoverage(bands[k].lower, bands[k].upper, ty),
      ]),
    ) as Record<keyof typeof bands, ReturnType<typeof intervalCoverage>>
    return { tx, ty, tf, bands, coverage, qSplit: split.quantile, qCqr: cqr.quantile }
  }, [heteroscedastic, scale, homoscedastic, nCal, seed, alpha])
  const shown = r.bands[band as keyof typeof r.bands]
  const covered = flat(r.coverage[band as keyof typeof r.coverage].covered)
  const inX = r.tx.filter((_, i) => covered[i])
  const inY = r.ty.filter((_, i) => covered[i])
  const outX = r.tx.filter((_, i) => !covered[i])
  const outY = r.ty.filter((_, i) => !covered[i])
  const xAxis = useAxis({ label: 'x', range: [0, 1] })
  const yAxis = useAxis({ label: 'y', hold: 'union', key: `${heteroscedastic}/${seed}` })
  return (
    <Figure
      title="Conformal intervals reach the target coverage"
      purpose="Calibrated on held-out cases, split conformal widens ŷ ± q̂ until 1 − α of new cases are covered, with one width everywhere; conformalised quantile regression adjusts the model's own interval and keeps its shape."
      state={state}
      readouts={{
        [`coverage on ${nTest} test cases (target ${fmt(1 - alpha)})`]: (
          <>
            {BANDS.map((b) => (
              <Readout
                key={b.value}
                label={b.label}
                value={`${fmt(r.coverage[b.value].coverage)} · width ${fmt(r.coverage[b.value].meanSize)}`}
              />
            ))}
          </>
        ),
        quantiles: (
          <>
            <Readout label="split q̂" value={fmt(r.qSplit)} />
            <Readout label="CQR q̂" value={fmt(r.qCqr)} />
          </>
        ),
      }}
      caption="The noise grows to the right, and the model's interval is too narrow by the width factor (and, as 'ignores the growth' rises, blind to the growth), so it under-covers. Split conformal on |y − ŷ| restores the coverage with a constant width: too wide on the left, too narrow on the right, right on average. CQR shifts both quantiles by q̂ (negative when the model's interval is too wide) and keeps their adaptive shape, so its misses spread evenly along x instead of piling up on the right. Red points fall outside the band shown."
    >
      <Plot x={xAxis} y={yAxis}>
        <Area name="interval" x={r.tx} y={shown.upper} base={shown.lower} slot={0} opacity={0.2} />
        <Curve name="prediction ŷ" x={r.tx} y={r.tf} emphasis />
        <Points name="covered" x={inX} y={inY} muted size={4} />
        <Points name="not covered" x={outX} y={outY} tone="destructive" size={5} />
      </Plot>
    </Figure>
  )
}

// ── 3 · Coverage over calibration draws ──────────────────────────────────────────────────────────────────────────────

export function CoverageLawSpecimen() {
  const state = useFigureState({
    conformal: row('1 · calibration', {
      calibration: int(50, { ge: 5, le: 2000, suggestions: [10, 50, 200, 1000], label: 'calibration cases n' }),
      alpha: float(0.1, { gt: 0, lt: 1, suggestions: [0.05, 0.1, 0.2], label: 'miscoverage α' }),
      repeats: int(300, { ge: 20, le: 2000, suggestions: [100, 300, 1000], label: 'calibration draws' }),
    }),
  })
  const { calibration: n, alpha, repeats } = state.conformal
  const nTest = 2000
  const sim = useComputed(
    () => {
      const coverage: number[] = []
      for (let r = 0; r < repeats; r++) {
        const d = quantileModelOutputs(child(stream('conformal/law'), 'draw', r), {
          n: n + nTest,
          alpha,
          homoscedastic: 1,
        })
        const y = flat(d.y!)
        const f = flat(d.prediction)
        const out = splitConformalRegression({ targets: y.slice(0, n), predictions: f.slice(0, n) }, f.slice(n), alpha)
        coverage.push(intervalCoverage(out.lower, out.upper, y.slice(n)).coverage)
      }
      return coverage
    },
    [n, alpha, repeats],
    { mode: 'release' },
  )
  const l = Math.floor((n + 1) * alpha)
  const law = l >= 1 ? Beta(n + 1 - l, l) : null
  const coverage = sim.value
  const meanCoverage = coverage.reduce((a, b) => a + b, 0) / coverage.length
  const covAxis = useAxis({ label: 'coverage on new cases', range: [Math.max(0, 1 - 4 * alpha - 0.15), 1] })
  const densAxis = useAxis({ label: 'density', hold: 'union', key: `${n}/${alpha}` })
  return (
    <Figure
      title="The coverage of one calibration is a Beta random variable"
      purpose="Conditional on the calibration set, the coverage of split conformal is random: with distinct scores it follows Beta(n + 1 − l, l), l = ⌊(n + 1)α⌋, whose mean is at least 1 − α and whose spread shrinks as n grows."
      state={state}
      readouts={
        <>
          <Readout label="mean coverage" value={fmt(meanCoverage)} />
          <Readout label="target 1 − α" value={fmt(1 - alpha)} />
          <Readout label="Beta mean" value={law ? fmt((n + 1 - l) / (n + 1)) : 'every answer included'} />
          <Readout
            label="draws below 1 − α"
            value={fmt(coverage.filter((c) => c < 1 - alpha).length / coverage.length)}
          />
        </>
      }
      caption={`Each draw calibrates split conformal on n fresh cases and measures its coverage on ${nTest} more. The histogram of those coverages follows the Beta law (ink curve), slightly widened by the finite test set. With small n a single calibration can under-cover noticeably; the guarantee is on average over calibration sets. When (n + 1)α < 1 the conformal quantile is infinite and every interval is the whole line. Recomputed on release.`}
    >
      <Plot x={covAxis} y={densAxis}>
        <Histogram name="coverage per calibration draw" values={coverage} bins={30} slot={0} stale={sim.stale} />
        {law && <Density name={`Beta(${n + 1 - l}, ${l})`} dist={law} emphasis />}
        <Annotation x={1 - alpha} text="1 − α" dashed />
      </Plot>
    </Figure>
  )
}

// ── 4 · Classification sets ──────────────────────────────────────────────────────────────────────────────────────────

const SCORES: { value: ClassificationScore; label: string }[] = [
  { value: 'lac', label: 'LAC (1 − p_y)' },
  { value: 'aps', label: 'APS' },
  { value: 'raps', label: 'RAPS' },
]

export function ConformalSetsSpecimen() {
  const state = useFigureState({
    model: row('1 · model', {
      classes: int(6, { ge: 2, le: 20, suggestions: [3, 6, 10], label: 'classes K' }),
      temperature: float(2.5, {
        gt: 0,
        scale: 'log10',
        suggestions: [0.5, 1, 2.5, 5],
        label: 'overconfidence (logits ×)',
      }),
      separation: slider(0.2, 4, 1.5, { step: 0.1, label: 'class separation' }),
      seed: slider(0, 20, 0, { step: 1, label: 'seed' }),
    }),
    conformal: row('2 · conformal', {
      alpha: float(0.1, { gt: 0, lt: 1, suggestions: [0.05, 0.1, 0.2], label: 'miscoverage α' }),
      lambda: float(0.05, { ge: 0, suggestions: [0, 0.01, 0.05, 0.2], label: 'RAPS penalty λ' }),
    }),
  })
  const { classes: K, temperature, separation, seed } = state.model
  const { alpha, lambda } = state.conformal
  const n = 4000
  const r = useMemo(() => {
    const d = classifierOutputs(stream(`conformal/sets/${seed}`), { n, classes: K, temperature, separation })
    const half = n / 2
    const y = flat(d.y!)
    const P = softmax(d.x)
    const rows = toRows(P) as number[][]
    const cal = { probabilities: rows.slice(0, half), labels: y.slice(0, half) }
    const test = rows.slice(half)
    const ty = y.slice(half)
    return SCORES.map(({ value }) => {
      const out = conformalClassification(cal, test, alpha, {
        score: value,
        lambda,
        kReg: 1,
        stream: child(stream(`conformal/sets/u/${seed}`), value),
      })
      const sizes = flat(out.sizes)
      const hist = Array.from({ length: K + 1 }, (_, k) => sizes.filter((s) => s === k).length / sizes.length)
      return { score: value, ...setCoverage(out.sets, ty), hist, quantile: out.quantile }
    })
  }, [K, temperature, separation, seed, alpha, lambda])
  const sizeAxis = useAxis({ label: 'set size', range: [-0.6, K + 0.6], integer: true })
  const shareAxis = useAxis({ label: 'share of test cases', range: [0, 1] })
  return (
    <Figure
      title="Conformal sets: LAC, APS and RAPS"
      purpose="All three scores give sets that contain the true class for at least 1 − α of new cases; they differ in how the set size adapts to how hard a case is."
      state={state}
      readouts={Object.fromEntries(
        r.map((s, k) => [
          SCORES[k].label,
          <>
            <Readout label="coverage" value={fmt(s.coverage)} />
            <Readout label="mean size" value={fmt(s.meanSize)} />
            <Readout label="q̂" value={fmt(s.quantile)} />
          </>,
        ]),
      )}
      caption="Sets are calibrated on 2000 cases and judged on 2000 more. LAC keeps every class with probability at least 1 − q̂, which gives the smallest sets on average. APS adds classes in order of probability until their mass reaches q̂, randomised at the boundary, so a hard case gets a larger set. RAPS adds a penalty λ per class beyond the first, which trims APS's long tail of unlikely classes. Raise the overconfidence to see APS sets grow."
    >
      <Plot x={sizeAxis} y={shareAxis}>
        {r.map((s, k) => (
          <Bars
            key={s.score}
            name={SCORES[k].label}
            x={s.hist.map((_, i) => i + (k - 1) * 0.27)}
            y={s.hist}
            width={0.25}
            slot={k}
          />
        ))}
      </Plot>
    </Figure>
  )
}
