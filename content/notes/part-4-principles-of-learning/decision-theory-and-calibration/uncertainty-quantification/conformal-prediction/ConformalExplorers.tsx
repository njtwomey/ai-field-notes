import { useMemo, useState } from 'react'
import { classifierOutputs, quantileModelOutputs } from 'aifn-applied/data/synthetic'
import { child, stream } from 'aifn/foundation/random'
import { toFlat, toRows, type Tensor } from 'aifn/foundation/tensor'
import {
  conformalClassification,
  conformalisedQuantileRegression,
  intervalCoverage,
  setCoverage,
  splitConformalRegression,
  type ClassificationScore,
} from 'aifn/learning/conformal'
import { softmax } from 'aifn/numerics/special'
import { Beta } from 'aifn/probability/distributions'
import {
  Annotation,
  Area,
  Bars,
  ControlRow,
  Curve,
  Density,
  Figure,
  Histogram,
  NumberSelector,
  Plot,
  Points,
  Readout,
  Select,
  formatNumber,
  useAxis,
} from 'aifn-render'

const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(digits))) : '—')
const flat = (t: Tensor) => Array.from(toFlat(t))

// ── 1 · Conformal regression explorer ─────────────────────────────────────────────────────────────────────────────

const BANDS = [
  { value: 'model', label: 'model raw interval' },
  { value: 'split', label: 'split conformal (ŷ ± q̂)' },
  { value: 'cqr', label: 'conformalised quantile regression (CQR)' },
] as const
type BandType = (typeof BANDS)[number]['value']

export function ConformalRegressionExplorer() {
  const [band, setBand] = useState<BandType>('cqr')
  const [alpha, setAlpha] = useState(0.1)
  const [nCal, setNCal] = useState(200)
  const [heteroscedastic, setHeteroscedastic] = useState(3)

  const nTest = 600
  const scale = 0.6
  const homoscedastic = 0
  const seed = 0

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

  const shown = r.bands[band]
  const covered = flat(r.coverage[band].covered)
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
      controls={
        <>
          <ControlRow label="Method & guarantee">
            <Select
              label="Interval method"
              value={band}
              onChange={(v) => setBand(v as BandType)}
              options={BANDS.map((b) => ({ value: b.value, label: b.label }))}
            />
            <NumberSelector
              label="Miscoverage rate α"
              value={alpha}
              onChange={setAlpha}
              min={0.05}
              max={0.3}
              step={0.05}
              suggestions={[0.05, 0.1, 0.2]}
            />
          </ControlRow>
          <ControlRow label="Data & calibration">
            <NumberSelector
              label="Calibration size n"
              value={nCal}
              onChange={setNCal}
              min={20}
              max={1000}
              step={50}
              suggestions={[50, 100, 200, 500]}
            />
            <NumberSelector
              label="Noise heteroscedasticity"
              value={heteroscedastic}
              onChange={setHeteroscedastic}
              min={0}
              max={5}
              step={0.5}
              suggestions={[0, 1.5, 3, 5]}
            />
          </ControlRow>
        </>
      }
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
            <Readout label="Split q̂" value={fmt(r.qSplit)} />
            <Readout label="CQR q̂" value={fmt(r.qCqr)} />
          </>
        ),
      }}
      caption="The noise grows to the right, and the model's uncalibrated interval is too narrow, so it under-covers. Split conformal on |y − ŷ| restores coverage with a constant width (too wide on the left, too narrow on the right, right on average). CQR shifts both quantiles by q̂ and preserves adaptivity, miss-distributing errors evenly along x. Red points fall outside the band."
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

// ── 2 · Coverage law explorer ─────────────────────────────────────────────────────────────────────────────────────

export function CoverageLawExplorer() {
  const [n, setN] = useState(50)
  const [alpha, setAlpha] = useState(0.1)
  const repeats = 250
  const nTest = 1500

  const sim = useMemo(() => {
    const cov: number[] = []
    for (let r = 0; r < repeats; r++) {
      const d = quantileModelOutputs(child(stream('conformal/law'), 'draw', r), {
        n: n + nTest,
        alpha,
        homoscedastic: 1,
      })
      const y = flat(d.y!)
      const f = flat(d.prediction)
      const out = splitConformalRegression({ targets: y.slice(0, n), predictions: f.slice(0, n) }, f.slice(n), alpha)
      cov.push(intervalCoverage(out.lower, out.upper, y.slice(n)).coverage)
    }
    return cov
  }, [n, alpha, repeats, nTest])

  const l = Math.floor((n + 1) * alpha)
  const law = l >= 1 ? Beta(n + 1 - l, l) : null
  const meanCoverage = sim.reduce((a, b) => a + b, 0) / sim.length

  const covAxis = useAxis({ label: 'coverage on new cases', range: [Math.max(0, 1 - 4 * alpha - 0.15), 1] })
  const densAxis = useAxis({ label: 'density', hold: 'union', key: `${n}/${alpha}` })

  return (
    <Figure
      title="The coverage of one calibration is a Beta random variable"
      purpose="Conditional on the calibration set, the coverage of split conformal is random: with distinct scores it follows Beta(n + 1 − l, l), l = ⌊(n + 1)α⌋, whose mean is at least 1 − α and whose spread shrinks as n grows."
      controls={
        <ControlRow label="Calibration parameters">
          <NumberSelector
            label="Calibration size n"
            value={n}
            onChange={setN}
            min={10}
            max={500}
            step={25}
            suggestions={[20, 50, 100, 200]}
          />
          <NumberSelector
            label="Miscoverage α"
            value={alpha}
            onChange={setAlpha}
            min={0.05}
            max={0.25}
            step={0.05}
            suggestions={[0.05, 0.1, 0.2]}
          />
        </ControlRow>
      }
      readouts={{
        diagnostics: (
          <>
            <Readout label="Mean empirical coverage" value={fmt(meanCoverage)} />
            <Readout label="Target 1 − α" value={fmt(1 - alpha)} />
            <Readout label="Beta theoretical mean" value={law ? fmt((n + 1 - l) / (n + 1)) : 'every answer included'} />
            <Readout
              label="Draws below 1 − α"
              value={fmt(sim.filter((c) => c < 1 - alpha).length / sim.length)}
            />
          </>
        ),
      }}
      caption="Each draw calibrates split conformal on n fresh cases and measures empirical coverage on fresh test cases. The histogram follows the theoretical Beta law (solid line). With small n, a single calibration draw can under-cover or over-cover noticeably; the coverage guarantee holds on average across calibration sets."
    >
      <Plot x={covAxis} y={densAxis}>
        <Histogram name="coverage per calibration draw" values={sim} bins={30} slot={0} />
        {law && <Density name={`Beta(${n + 1 - l}, ${l})`} dist={law} emphasis />}
        <Annotation x={1 - alpha} text="1 − α" dashed />
      </Plot>
    </Figure>
  )
}

// ── 3 · Classification sets explorer ─────────────────────────────────────────────────────────────────────────────

const SCORES: { value: ClassificationScore; label: string }[] = [
  { value: 'lac', label: 'LAC (1 − p_y)' },
  { value: 'aps', label: 'APS (adaptive)' },
  { value: 'raps', label: 'RAPS (regularized)' },
]

export function ConformalSetsExplorer() {
  const [K, setK] = useState(6)
  const [alpha, setAlpha] = useState(0.1)
  const [lambda, setLambda] = useState(0.05)

  const temperature = 2.5
  const separation = 1.5
  const seed = 0
  const n = 3000

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
  }, [K, temperature, separation, seed, alpha, lambda, n])

  const sizeAxis = useAxis({ label: 'set size', range: [-0.6, K + 0.6], integer: true })
  const shareAxis = useAxis({ label: 'share of test cases', range: [0, 1] })

  return (
    <Figure
      title="Conformal classification sets: LAC, APS and RAPS"
      purpose="All three nonconformity scores produce prediction sets containing the true class for at least 1 − α of new cases; they differ in how set size adapts to input ambiguity."
      controls={
        <ControlRow label="Conformal configuration">
          <NumberSelector
            label="Classes K"
            value={K}
            onChange={setK}
            min={3}
            max={10}
            step={1}
            suggestions={[3, 6, 10]}
          />
          <NumberSelector
            label="Miscoverage α"
            value={alpha}
            onChange={setAlpha}
            min={0.05}
            max={0.25}
            step={0.05}
            suggestions={[0.05, 0.1, 0.2]}
          />
          <NumberSelector
            label="RAPS penalty λ"
            value={lambda}
            onChange={setLambda}
            min={0}
            max={0.2}
            step={0.01}
            suggestions={[0, 0.02, 0.05, 0.1]}
          />
        </ControlRow>
      }
      readouts={Object.fromEntries(
        r.map((s, k) => [
          SCORES[k].label,
          <>
            <Readout label="Coverage" value={fmt(s.coverage)} />
            <Readout label="Mean size" value={fmt(s.meanSize)} />
            <Readout label="Cutoff q̂" value={fmt(s.quantile)} />
          </>,
        ]),
      )}
      caption="Sets are calibrated on 1500 cases and judged on 1500 test cases. LAC keeps every class with estimated probability at least 1 − q̂. APS sorts classes by probability and accumulates until reaching q̂, adapting size to hard cases. RAPS adds penalty λ per extra class, preventing long tails of unlikely labels."
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
