import { arRegimes, meanShifts, poissonShifts, varianceShifts } from 'aifn-applied/data/synthetic'
import { coalMining } from 'aifn-applied/data/real'
import type { Dataset } from 'aifn-applied/data'
import {
  bocpd,
  bocpdForecast,
  bocpdPredictiveDensity,
  constantHazard,
  laggedObservations,
  mapChangepoints,
  normalGamma,
  normalKnownVariance,
  poissonGamma,
  regressionNormalGamma,
  runLengthMass,
  runLengthRow,
  type BocpdState,
  type ConjugatePredictive,
  type Regressed,
  type RunStats,
} from 'aifn/inference/filtering'
import { stream } from 'aifn/foundation/random'
import { toFlat } from 'aifn/foundation/tensor'
import { extend, seek, trace, type Trace } from 'aifn/foundation/trace'
import { useEffect, useMemo, useRef, useState } from 'react'
import { Player } from '@lab/controls'
import { Dashboard, DashboardCell, DashboardRow, Figure } from '@lab/layout'
import { slider, useFigureState, variants } from '@lab/state'
import { Bars, Curve, Handle, Plot, Plots, Raster, Readout, useAxis } from '@lab/viz'
import { formatValue } from '@lab/views'

const f3 = (v: number) => formatValue(Number(v.toPrecision(3)))

// ── Row 1: the data ─────────────────────────────────────────────────────────────────────────────────────────────────

const n = slider(100, 400, 250, { step: 10, label: 'length n' })
const meanGap = slider(20, 150, 60, { step: 5, label: 'mean gap between changes' })
const seed = slider(1, 40, 1, { step: 1, label: 'seed' })

/** Each dataset's parameters (the row), and how it is made (DATASETS). */
const DATA = variants(
  {
    mean: {
      label: 'mean shifts',
      params: {
        n,
        meanGap,
        jump: slider(0.5, 5, 3, { label: 'spread of means' }),
        sd: slider(0.2, 2, 1, { label: 'noise sd' }),
        seed,
      },
    },
    variance: {
      label: 'variance shifts',
      params: { n, meanGap, minRatio: slider(1.2, 6, 2.5, { label: 'least sd ratio' }), seed },
    },
    poisson: {
      label: 'Poisson shifts (counts)',
      params: { n, meanGap, minRatio: slider(1.2, 6, 2, { label: 'least rate ratio' }), seed },
    },
    ar: {
      label: 'AR regimes (a = 0.9, −0.7)',
      params: { n, meanGap, sd: slider(0.2, 2, 1, { label: 'innovation sd' }), seed },
    },
    coal: { label: 'coal-mining disasters (1851–1962)', params: {} },
  },
  { label: '1 · data', choiceLabel: 'dataset' },
)
type Values = Readonly<Record<string, number>>
const DATASETS: Record<DataKey, (p: Values) => Dataset> = {
  mean: (p) =>
    meanShifts(stream(`bocpd-${p.seed}`), { n: p.n, meanGap: p.meanGap, jump: p.jump, sd: p.sd, minJump: p.sd }),
  variance: (p) => varianceShifts(stream(`bocpd-${p.seed}`), { n: p.n, meanGap: p.meanGap, minRatio: p.minRatio }),
  poisson: (p) => poissonShifts(stream(`bocpd-${p.seed}`), { n: p.n, meanGap: p.meanGap, minRatio: p.minRatio }),
  ar: (p) => arRegimes(stream(`bocpd-${p.seed}`), { n: p.n, meanGap: p.meanGap, sd: p.sd }),
  coal: () => coalMining(),
}
type DataKey = 'mean' | 'variance' | 'poisson' | 'ar' | 'coal'
const COUNTS: readonly DataKey[] = ['poisson', 'coal']
const NATURAL: Record<DataKey, ModelKey> = {
  mean: 'normal',
  variance: 'normalGamma',
  poisson: 'poisson',
  ar: 'ar',
  coal: 'poisson',
}

// ── Row 2: the segment model and the hazard ─────────────────────────────────────────────────────────────────────────

/** Every model sees observations as `Regressed` records, so one trace type serves all (scalar models read `y`). */
type Obs = Regressed
function scalar<S extends RunStats>(m: ConjugatePredictive<number, S>): ConjugatePredictive<Obs, S> {
  return {
    name: m.name,
    prior: () => m.prior(),
    predictive: (s) => m.predictive(s),
    update: (s, x) => m.update(s, x.y),
    value: (x) => x.y,
  }
}

/** Each segment model's prior (the row), and the conjugate predictive it makes (SEGMENT_MODELS). */
const MODELS = variants(
  {
    normal: {
      label: 'normal, known sd',
      params: {
        mean: slider(-5, 5, 0, { label: 'prior mean μ₀' }),
        priorSd: slider(0.5, 10, 3, { label: 'prior sd of the mean σ₀' }),
        sd: slider(0.1, 3, 1, { label: 'noise sd σ' }),
      },
    },
    normalGamma: {
      label: 'normal–gamma',
      params: {
        mean: slider(-5, 5, 0, { label: 'prior mean μ₀' }),
        kappa: slider(0.01, 5, 0.1, { label: 'prior strength κ₀' }),
        alpha: slider(0.5, 10, 1, { label: 'precision shape α₀' }),
        beta: slider(0.1, 10, 1, { label: 'precision rate β₀' }),
      },
    },
    poisson: {
      label: 'Poisson–gamma (counts only)',
      params: {
        shape: slider(0.2, 10, 1, { label: 'rate prior shape' }),
        rate: slider(0.05, 5, 0.5, { label: 'rate prior rate' }),
      },
    },
    ar: {
      label: 'autoregression',
      params: {
        order: slider(1, 3, 1, { step: 1, label: 'order p' }),
        priorScale: slider(0.1, 3, 1, { label: 'coefficient prior sd' }),
        alpha: slider(0.5, 10, 1, { label: 'noise shape α' }),
        beta: slider(0.1, 10, 1, { label: 'noise scale β' }),
      },
    },
  },
  {
    label: '2 · segment model and hazard',
    choiceLabel: 'predictive model',
    shared: { lambda: slider(10, 400, 100, { step: 5, label: 'hazard: mean gap λ (H = 1/λ)' }) },
  },
)
const SEGMENT_MODELS: Record<ModelKey, (p: Values) => ConjugatePredictive<Obs>> = {
  normal: (p) => scalar(normalKnownVariance({ mean: p.mean, priorSd: p.priorSd, sd: p.sd })),
  normalGamma: (p) => scalar(normalGamma({ mean: p.mean, kappa: p.kappa, alpha: p.alpha, beta: p.beta })),
  poisson: (p) => scalar(poissonGamma({ shape: p.shape, rate: p.rate })),
  ar: (p) => regressionNormalGamma({ dimension: p.order + 1, priorScale: p.priorScale, alpha: p.alpha, beta: p.beta }),
}
type ModelKey = 'normal' | 'normalGamma' | 'poisson' | 'ar'

/** The run-length heatmap's floor: probabilities below 10⁻⁶ share the lightest colour. */
const FLOOR = -6

export function BocpdShowcase() {
  const state = useFigureState({ data: DATA, model: MODELS })
  const data = state.data
  const model = state.model
  const counts = COUNTS.includes(data.key)
  // A change of dataset brings its natural model; the reader can then pick any model that fits the data's type (the
  // Poisson model only for counts).
  const lastData = useRef(data.key)
  const { set } = state
  useEffect(() => {
    if (lastData.current !== data.key) {
      lastData.current = data.key
      set('model', NATURAL[data.key])
    } else if (model.key === 'poisson' && !counts) set('model', NATURAL[data.key])
  }, [data.key, model.key, counts, set])

  const dataKey = `${data.key}:${JSON.stringify(data.values)}`
  const ds = useMemo(() => DATASETS[data.key](data.values as unknown as Values), [dataKey]) // eslint-disable-line react-hooks/exhaustive-deps
  const values = useMemo(() => Array.from(toFlat(ds.x)), [ds])
  const times = useMemo(() => (ds.t ? Array.from(toFlat(ds.t)) : values.map((_, i) => i)), [ds, values])
  const truth = useMemo(() => (ds.meta.truth?.task === 'changepoint' ? ds.meta.truth.changepoints : []), [ds])

  const order = model.key === 'ar' ? (model.values as unknown as Values).order : 0
  // Observation i is x[i + order]: an autoregression needs p lags before its first observation.
  const obs = useMemo<Obs[]>(
    () => (order > 0 ? laggedObservations(values, order) : values.map((y) => ({ y, z: [] as number[] }))) as Obs[],
    [values, order],
  )
  const modelKey = `${model.key}:${JSON.stringify(model.values)}`
  const segmentModel = useMemo(
    () => SEGMENT_MODELS[model.key](model.values as unknown as Values),
    [modelKey.replace(/"lambda":[^,}]*/, '')], // eslint-disable-line react-hooks/exhaustive-deps
  )
  const lambda = model.values.lambda
  const alg = useMemo(() => bocpd(segmentModel, obs, { hazard: constantHazard(lambda) }), [segmentModel, obs, lambda])
  const N = obs.length

  // ── Playback: extend the trace as the player advances; a new model or hazard retraces to the current position ──
  const [pos, setPos] = useState(0)
  const at = Math.min(pos, N)
  const cache = useRef<{ alg: typeof alg; tr: Trace<BocpdState> } | null>(null)
  const tr = useMemo(() => {
    let c = cache.current
    if (!c || c.alg !== alg) c = { alg, tr: trace(alg, undefined, at, { keep: 'all' }) }
    else if (c.tr.meta.steps < at) c = { alg, tr: extend(c.tr, alg, at - c.tr.meta.steps) }
    cache.current = c
    return c.tr
  }, [alg, at])
  const bstate = useMemo(() => seek(alg, undefined, at, { checkpoints: tr }), [alg, at, tr])
  const states = tr.steps.slice(1, at + 1)

  // ── The run-length posterior: rows appended as the trace grows, held on the full n × n grid ──
  const rowsCache = useRef<{ alg: typeof alg; rows: number[][] } | null>(null)
  const rows = useMemo(() => {
    if (rowsCache.current?.alg !== alg) rowsCache.current = { alg, rows: [] }
    const rows = rowsCache.current.rows
    for (let t = rows.length; t < tr.steps.length - 1; t++)
      rows.push(
        Array.from(runLengthRow(tr.steps[t + 1], N + 1), (p) => (p > 0 ? Math.max(FLOOR, Math.log10(p)) : FLOOR)),
      )
    return rows
  }, [alg, tr, N])
  const obsTimes = useMemo(() => obs.map((_, i) => times[i + order]), [obs, times, order])
  // Held run-length axis: 1.6 × the longest true segment (room for a missed change), or every run length without truth.
  const rMax = useMemo(() => {
    if (!truth.length) return N
    const cuts = [0, ...truth, values.length]
    const longest = Math.max(...cuts.slice(1).map((c, i) => c - cuts[i]))
    return Math.min(N, Math.ceil(1.6 * longest))
  }, [truth, values, N])
  const grid = useMemo(() => {
    const z: number[][] = Array.from({ length: rMax + 1 }, () => new Array<number>(N).fill(NaN))
    for (let t = 0; t < at; t++) for (let r = 0; r <= rMax; r++) z[r][t] = rows[t][r]
    return { x: obsTimes, y: Array.from({ length: rMax + 1 }, (_, r) => r), z }
  }, [rows, at, N, obsTimes, rMax])

  const column = useMemo(() => {
    const p = at > 0 ? runLengthRow(bstate, N + 1) : Float64Array.from({ length: N + 1 }, (_, r) => (r === 0 ? 1 : 0))
    return { x: Array.from(p), y: Array.from(p, (_, r) => r) }
  }, [bstate, at, N])
  const maps = states.map((s) => s.map)
  const detected = useMemo(() => mapChangepoints(maps), [maps.join(',')]) // eslint-disable-line react-hooks/exhaustive-deps
  const mapPath = useMemo(
    () => ({ x: obsTimes.slice(0, at), y: maps }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [obsTimes, at, maps.join(',')],
  )

  // ── The series so far, with the true (dashed) and detected (solid) changepoints ──
  const yRange = useMemo<[number, number]>(() => {
    const lo = Math.min(...values)
    const hi = Math.max(...values)
    const pad = 0.08 * (hi - lo || 1)
    return [counts ? 0 : lo - pad, hi + pad]
  }, [values, counts])
  const xRange: [number, number] = [times[0], times[times.length - 1]]
  const verticals = (at: readonly number[], [lo, hi]: readonly [number, number] = yRange) => ({
    x: at.flatMap((c) => [c, c, NaN]),
    y: at.flatMap(() => [lo, hi, NaN]),
  })
  const series = useMemo(() => {
    const seen = values.slice(0, at + order)
    const from = Math.max(seen.length - 1, 0)
    return {
      // The observations still to come, faint, so the reader sees what the filter is about to meet.
      ahead: { x: times.slice(from), y: values.slice(from) },
      seen: { x: times.slice(0, seen.length), y: seen },
      // Every true change from the start, so the reader can watch the filter approach each one.
      truth: verticals(truth.map((c) => times[c])),
      detected: verticals(detected.map((c) => obsTimes[c])),
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [values, times, at, order, truth, detected, obsTimes, yRange])

  // The same true changes on the run-length posterior: a detected change starts a new diagonal ridge from r = 0 there.
  const truthOnRuns = useMemo(
    () =>
      verticals(
        truth.map((c) => times[c]),
        [0, rMax],
      ),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [truth, times, rMax],
  )

  // ── The time cursor: the last observation fed in; dragging it on either time plot moves the player ──
  const cursor = at > 0 ? obsTimes[at - 1] : obsTimes[0]
  const seekTo = (x: number) => {
    // The number of observations at or before x: the cursor sits on the latest one fed.
    let k = 0
    while (k < N && obsTimes[k] <= x) k++
    setPos(k)
  }

  // ── The one-step predictive p(xₜ₊₁ | x₁:ₜ), drawn sideways on the series' value axis ──
  const next = at < N ? obs[at] : undefined
  const predictive = useMemo(() => {
    const [lo, hi] = yRange
    const levels = counts
      ? Array.from({ length: Math.ceil(hi) + 1 }, (_, k) => k)
      : Array.from({ length: 161 }, (_, k) => lo + ((hi - lo) * k) / 160)
    const dens = Array.from(bocpdPredictiveDensity(segmentModel, bstate, levels, next))
    const peak = Math.max(...dens)
    return { levels, dens, next: next ? { x: [0, peak * 1.05], y: [next.y, next.y] } : null }
  }, [segmentModel, bstate, next, yRange, counts])

  const forecast = at > 0 ? bocpdForecast(segmentModel, bstate, next) : null
  const prevMap = at > 1 ? states[at - 2].map : 0
  const grows = at > 0 ? runLengthMass(bstate, prevMap + 1, prevMap + 2) : 1
  const recent = at > 0 ? runLengthMass(bstate, 0, 6) : 1
  const time = useAxis({ label: data.key === 'coal' ? 'year' : 'time', range: xRange })
  const valueAxis = useAxis({ label: 'xₜ', range: yRange })
  const densityAxis = useAxis({
    label: 'density',
    hold: 'union',
    key: `${data.key}-${model.key}`,
    range: [0, undefined],
  })
  const runAxis = useAxis({ label: 'run length rₜ', range: [0, rMax] })
  const probAxis = useAxis({ label: 'p(rₜ | x₁:ₜ) now', range: [0, 1] })
  const t = at === 0 ? '0' : `${at} of ${N}${data.key === 'coal' ? ` (${obsTimes[at - 1]})` : ''}`

  return (
    <Figure
      title="Bayesian online changepoint detection, one observation at a time"
      purpose="Each new observation either extends the current run or starts a new one; the run-length posterior keeps every hypothesis, so a change shows as mass falling back to r = 0 and a new diagonal ridge."
      defaultSize="XL"
      state={state}
      controls={
        <Player
          label="3 · feed observations"
          value={at}
          onChange={setPos}
          count={N + 1}
          format={(p) => (p === 0 ? 'none yet' : `t = ${p}`)}
        />
      }
      readouts={{
        'at t': (
          <>
            <Readout label="observations t" value={t} />
            <Readout label="xₜ" value={at > 0 ? f3(obs[at - 1].y) : '—'} />
            <Readout label="MAP run length r̂ₜ" value={at > 0 ? bstate.map : 0} />
            <Readout label="growth: P(rₜ = r̂ₜ₋₁ + 1)" value={f3(grows)} />
            <Readout label="changepoint: P(rₜ ≤ 5)" value={f3(recent)} />
            <Readout
              label="forecast mean ± sd"
              value={forecast ? `${f3(forecast.mean)} ± ${f3(Math.sqrt(forecast.variance))}` : '—'}
            />
            <Readout label="log p(xₜ | x₁:ₜ₋₁)" value={at > 0 ? f3(bstate.logPredictive) : '—'} />
          </>
        ),
        changepoints: (
          <>
            <Readout
              label="changepoints detected / true"
              value={`${detected.length} / ${truth.filter((c) => c < at + order).length}`}
            />
          </>
        ),
      }}
      caption={`Press play to feed the observations one at a time; the axes hold the whole series, so the picture fills in. Drag the time cursor t on the series or on the run-length posterior (or press anywhere on either) to jump to that observation: both time plots, the predictive density and the current run-length column follow. Top left: the series so far (the rest faint), with every true changepoint dashed in ink and those read from the MAP run lengths (backtracked) solid. Top right: the one-step predictive density of the next observation, a mixture over run lengths weighted by their posterior, drawn on the series' value axis; the dashed line is the value that actually comes next. Bottom: the run-length posterior p(rₜ | x₁:ₜ) on a log₁₀ scale (darkest at 1, lightest at 10⁻⁶ and below), one column per observation, with the MAP run length and the true changepoints dashed; beside it the current column on a linear scale. The run-length axis is held at 1.6 times the longest true segment (all run lengths for real data); mass above it is not drawn. rₜ counts the observations in the current run, so r = 0 is a change just after xₜ, which with a constant hazard always has probability H = 1/λ; the readout "P(rₜ ≤ 5)" is the probability of a change in the last five steps. Changing the model or λ re-runs the filter up to the current step.`}
    >
      <Dashboard>
        <DashboardRow ratio={1} minHeight={240}>
          <DashboardCell>
            <Plots cols={2} widths={[4, 1]}>
              <Plot x={time} y={valueAxis}>
                <Curve name="to come" {...series.ahead} muted thin />
                <Curve name="series xₜ" {...series.seen} slot={0} showPoints={counts} />
                <Curve name="true changepoints" {...series.truth} dashed emphasis silent />
                <Curve name="detected changepoints" {...series.detected} slot={1} />
                <Handle kind="x" at={cursor} label="t" onDrag={seekTo} />
              </Plot>
              <Plot x={densityAxis} y={valueAxis}>
                {counts ? (
                  <Bars name="predictive" x={predictive.levels} y={predictive.dens} orient="y" width={0.6} slot={2} />
                ) : (
                  <Curve name="predictive" x={predictive.dens} y={predictive.levels} slot={2} />
                )}
                {predictive.next && <Curve name="next" {...predictive.next} emphasis dashed />}
              </Plot>
            </Plots>
          </DashboardCell>
        </DashboardRow>
        <DashboardRow ratio={1.3} minHeight={300}>
          <DashboardCell ratio={4}>
            <Plot x={time} y={runAxis}>
              <Raster {...grid} valueLabel="log₁₀ p(rₜ | x₁:ₜ)" range={[FLOOR, 0]} colorBar={false} />
              <Curve name="MAP run length" {...mapPath} slot={1} />
              <Curve name="true changepoints" {...truthOnRuns} dashed emphasis silent />
              <Handle kind="x" at={cursor} label="t" onDrag={seekTo} />
            </Plot>
          </DashboardCell>
          <DashboardCell ratio={1}>
            <Plot x={probAxis} y={runAxis} legend={false}>
              <Curve name="p(rₜ | x₁:ₜ)" {...column} slot={1} />
            </Plot>
          </DashboardCell>
        </DashboardRow>
      </Dashboard>
    </Figure>
  )
}
