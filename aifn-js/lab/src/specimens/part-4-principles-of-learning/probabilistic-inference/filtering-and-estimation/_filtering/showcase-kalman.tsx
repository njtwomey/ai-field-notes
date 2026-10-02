import { constantVelocityModel } from 'aifn-applied/timeseries'
import {
  kalmanFilter,
  kalmanFilterSteps,
  normalisedEstimationErrorSquared,
  normalisedInnovationSquared,
  rtsSmoother,
  rtsSmootherSteps,
  simulateStateSpace,
} from 'aifn/inference/filtering'
import { trace } from 'aifn/foundation/trace'
import { stream } from 'aifn/foundation/random'
import { toFlat, type Tensor } from 'aifn/foundation/tensor'
import { covarianceEllipse } from 'aifn/numerics/geometry'
import { ChiSquare } from 'aifn/probability/distributions'
import { useMemo, useState } from 'react'
import { Player } from '@lab/controls'
import { Dashboard, DashboardCell, DashboardRow, Figure } from '@lab/layout'
import { row, slider, toggle, useComputed, useFigureState } from '@lab/state'
import { Curve, Handle, Plot, Plots, Points, Readout, useAxis } from '@lab/viz'
import { formatValue } from '@lab/views'

const f3 = (v: number) => formatValue(Number(v.toPrecision(3)))
type P2 = [number, number]

const T = 40
const DROPOUT = { from: 16, to: 26 } // steps t = 16 … 25 (1-based) have no measurement when dropouts are on
const TRUE_Q = 0.05
const TRUE_SD = 1.5
const INITIAL = [0, 0, 1, 0.6]

/** The true track and its measurements, simulated once from the true model. */
const TRUTH = (() => {
  const model = constantVelocityModel({
    processNoise: TRUE_Q,
    measurementStd: TRUE_SD,
    initialMean: INITIAL,
    initialPositionStd: 0.01,
    initialVelocityStd: 0.01,
  })
  const sim = simulateStateSpace(stream('showcase-kalman'), model, T)
  const z = toFlat(sim.states)
  const y = toFlat(sim.observations)
  return {
    states: sim.states,
    position: Array.from({ length: T }, (_, t): P2 => [z[4 * t], z[4 * t + 1]]),
    measured: Array.from({ length: T }, (_, t): P2 => [y[2 * t], y[2 * t + 1]]),
  }
})()

/** Held axes: the track and its measurements with a margin for the ellipses. */
const BOX = (() => {
  const all = [...TRUTH.position, ...TRUTH.measured]
  const xs = all.map((p) => p[0])
  const ys = all.map((p) => p[1])
  const pad = 4
  return {
    x: [Math.floor(Math.min(...xs) - pad), Math.ceil(Math.max(...xs) + pad)] as [number, number],
    y: [Math.floor(Math.min(...ys) - pad), Math.ceil(Math.max(...ys) + pad)] as [number, number],
  }
})()

/** Position mean and 2×2 position covariance of a 4-vector mean and a 4 × 4 covariance. */
function position(mean: Tensor, cov: Tensor) {
  const m = toFlat(mean)
  const P = toFlat(cov)
  return {
    mean: [m[0], m[1]] as P2,
    cov: [
      [P[0], P[1]],
      [P[4], P[5]],
    ],
  }
}

function ellipse(mean: P2, cov: number[][]) {
  const e = covarianceEllipse(mean, cov, { mass: 0.95, points: 60 })
  const p = toFlat(e.points)
  return { x: p.filter((_, i) => i % 2 === 0), y: p.filter((_, i) => i % 2 === 1) }
}

const EMPTY = { x: [] as number[], y: [] as number[] }

const xy = (pts: readonly P2[]) => ({ x: pts.map((p) => p[0]), y: pts.map((p) => p[1]) })
const TIMES = Array.from({ length: T }, (_, i) => i + 1)

export function KalmanTracking() {
  const state = useFigureState({
    data: row('1 · data', { dropouts: toggle(true, 'dropouts (steps 16–25 missing)') }),
    model: row("2 · filter's model", {
      q: slider(0.001, 2, TRUE_Q, { label: 'process noise q', step: 0.001 }),
      sd: slider(0.1, 6, TRUE_SD, { label: 'measurement sd σ', step: 0.05 }),
    }),
  })
  const { dropouts } = state.data
  const { q, sd } = state.model
  const [moved, setMoved] = useState<Record<number, P2>>({})
  const [pos, setPos] = useState(0)

  // The measurements as drawn: a dragged one moves on the pointer event.
  const measured = useMemo(
    () =>
      TRUTH.measured.map((p, t): P2 =>
        dropouts && t + 1 >= DROPOUT.from && t + 1 < DROPOUT.to ? [NaN, NaN] : (moved[t] ?? p),
      ),
    [dropouts, moved],
  )
  // Frames: 0 is before any data; 1 … T run the filter forward to step t; T + 1 … 2T run the smoother back to step s.
  const smoothing = pos > T
  const model = useMemo(
    () =>
      constantVelocityModel({
        processNoise: q,
        measurementStd: sd,
        initialMean: INITIAL,
        initialPositionStd: 3,
        initialVelocityStd: 1,
      }),
    [q, sd],
  )
  const y = useMemo(() => measured.map((p) => [p[0], p[1]]), [measured])
  // The forward filter, its NIS and NEES. Dragging a measurement reruns it in the same render: 40 steps take about a
  // millisecond, less than the second render a scheduled (useComputed) run would add.
  const filtered = useMemo(() => {
    // The step-through filter: played frame t is its state after step t.
    const run = trace(kalmanFilterSteps(model, y), undefined, T)
    const steps = Array.from({ length: T }, (_, t) => {
      const st = run.steps[t + 1].step!
      return { predicted: position(st.predictedMean, st.predictedCov), filtered: position(st.mean, st.cov) }
    })
    const prior = position(run.steps[0].mean, run.steps[0].cov)
    // NIS and NEES over the whole run, from the packed result.
    const filter = kalmanFilter(model, y)
    const nis = toFlat(normalisedInnovationSquared(filter))
    const neesFilter = toFlat(normalisedEstimationErrorSquared(filter.mean, filter.cov, TRUTH.states))
    return { steps, prior, nis, neesFilter, logLikelihood: filter.logLikelihood }
  }, [model, y])
  // The backward smoother pass, only once the player reaches it, by the scheduler (a drag of q or σ while smoothing
  // reruns both passes).
  const smoothed = useComputed(() => {
    if (!smoothing) return null
    // The step-through smoother: its step k smooths index T − 1 − k.
    const run = trace(rtsSmootherSteps(model, y), undefined, T)
    const steps = Array.from({ length: T }, (_, t) => position(run.steps[T - 1 - t].mean, run.steps[T - 1 - t].cov))
    const smoother = rtsSmoother(model, y)
    const nees = toFlat(normalisedEstimationErrorSquared(smoother.mean, smoother.cov, TRUTH.states))
    return { steps, nees }
    // Inputs only while smoothing: a measurement dragged during the forward pass must not schedule a (null) rerun.
  }, [smoothing && model, smoothing && y])
  const result = useMemo(
    () => ({
      ...filtered,
      steps: filtered.steps.map((st, t) => ({ ...st, smoothed: smoothed.value?.steps[t] ?? st.filtered })),
      neesSmoother: smoothed.value?.nees ?? [],
    }),
    [filtered, smoothed.value],
  )
  const computed = { stale: smoothed.stale }

  const t = smoothing ? T - 1 : pos - 1 // the filter's last processed step (0-based), −1 before any
  const s = smoothing ? 2 * T - pos : T // the smoother's current step (0-based)
  const observed = (i: number) => !Number.isNaN(measured[i][0])

  const tracks = useMemo(
    () => ({
      truth: xy(TRUTH.position.slice(0, t + 1)),
      seen: xy(measured.slice(0, t + 1)),
      ahead: xy(measured.slice(t + 1)),
      filter: xy(result.steps.slice(0, t + 1).map((st) => st.filtered.mean)),
      smoother: smoothing ? xy(result.steps.slice(s).map((x) => x.smoothed.mean)) : null,
    }),
    [result, measured, t, s, smoothing],
  )
  // The ellipses at the played step: the prior before any data; predict and update while filtering; filter and
  // smoother while smoothing.
  const ellipses = useMemo(() => {
    if (t < 0) return { prior: ellipse(result.prior.mean, result.prior.cov) }
    if (!smoothing) {
      const st = result.steps[t]
      return {
        predict: ellipse(st.predicted.mean, st.predicted.cov),
        update: observed(t) ? ellipse(st.filtered.mean, st.filtered.cov) : undefined,
      }
    }
    const st = result.steps[s]
    return { filter: ellipse(st.filtered.mean, st.filtered.cov), smoother: ellipse(st.smoothed.mean, st.smoothed.cov) }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- observed reads measured
  }, [result, measured, t, s, smoothing])

  const sdOf = (c: number[][]) => Math.sqrt(c[0][0] + c[1][1])
  // The covariances do not depend on the measured values, only on the model and which steps are missing: dragging a
  // measurement leaves this panel untouched.
  const covKey = `${q}|${sd}|${dropouts}|${smoothed.value ? 1 : 0}`
  const spread = useMemo(() => {
    const filterSd = result.steps.map((st) => sdOf(st.filtered.cov))
    return {
      filter: { x: TIMES.slice(0, t + 1), y: filterSd.slice(0, t + 1) },
      smoother: smoothing ? { x: TIMES.slice(s), y: result.steps.slice(s).map((st) => sdOf(st.smoothed.cov)) } : null,
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed by covKey: the covariances only
  }, [covKey, t, s, smoothing])

  const nisBand = useMemo(() => {
    const X = ChiSquare(2)
    return [X.quantile(0.025) as number, X.quantile(0.975) as number]
  }, [])
  const nisNow = useMemo(() => ({ x: TIMES.slice(0, t + 1), y: result.nis.slice(0, t + 1) }), [result, t])

  // Average NIS over the observed steps so far, against the χ²_{2n}/n band; average NEES against χ²_{4n}/n.
  const seen = result.nis.slice(0, t + 1).filter((v) => Number.isFinite(v))
  const avgNis = seen.length ? seen.reduce((a, b) => a + b, 0) / seen.length : NaN
  const band = (df: number, n: number) => {
    if (n === 0) return '—'
    const X = ChiSquare(df * n)
    return `[${f3((X.quantile(0.025) as number) / n)}, ${f3((X.quantile(0.975) as number) / n)}]`
  }
  const avg = (v: readonly number[]) => v.reduce((a, b) => a + b, 0) / v.length
  const nees = smoothing ? avg(result.neesSmoother) : t >= 0 ? avg(result.neesFilter.slice(0, t + 1)) : NaN

  const current = !smoothing && t >= 0 && observed(t) ? measured[t] : null
  const px = useAxis({ label: 'x', range: BOX.x, zoom: false })
  const py = useAxis({ label: 'y', range: BOX.y, equal: px, zoom: false })
  const time = useAxis({ label: 't', range: [0, T + 1] })
  const sdAxis = useAxis({ label: 'position sd', range: [0, 10] })
  const nisAxis = useAxis({ label: 'NIS', range: [0, 15] })
  return (
    <Figure
      title="Kalman filter and RTS smoother tracking a target in the plane"
      purpose="Each step the filter predicts (the ellipse grows by the process noise) and then updates on the measurement (it shrinks); through a dropout it only predicts, and the backward smoother pass then pulls every estimate towards the later data and shrinks the ellipses most where the filter was least sure."
      defaultSize="L"
      state={state}
      controls={
        <Player
          label="3 · play"
          value={pos}
          onChange={setPos}
          count={2 * T + 1}
          format={(p) => (p === 0 ? 'prior' : p <= T ? `filter, t = ${p}` : `smoother, t = ${2 * T - p + 1}`)}
        />
      }
      readouts={{
        consistency: (
          <>
            <Readout label="average NIS (observed steps)" value={Number.isFinite(avgNis) ? f3(avgNis) : '—'} />
            <Readout label="its 95% band" value={band(2, seen.length)} />
            <Readout
              label={smoothing ? 'average NEES (smoother)' : 'average NEES (filter)'}
              value={Number.isFinite(nees) ? f3(nees) : '—'}
            />
            <Readout label="its 95% band" value={band(4, smoothing ? T : Math.max(t + 1, 0))} />
          </>
        ),
        fit: <Readout label="log-likelihood" value={f3(result.logLikelihood)} />,
      }}
      caption={`The target moves at nearly constant velocity (true q = ${TRUE_Q}, true σ = ${TRUE_SD}); the sliders set the filter's own model. Left: the true track (dashed), the measurements (faint until the filter reaches them), the prior position ellipse at the start, the filter mean, and at the played step the predicted (dashed) and updated (solid) 95% position ellipses. Drag the current measurement to see the update follow it. After step ${T} the player runs the RTS smoother backwards: the smoothed track and, at its current step, the filtered (dashed) and smoothed (solid) ellipses. Top right: the position uncertainty √(P_xx + P_yy) of the filter and the smoother; it grows through the dropout (steps 16–25). Bottom right: the normalised innovation squared at each step with the χ²₂ 95% band. With the true noise the average NIS and NEES sit in or near their bands (one simulated track); shrink q and they climb out of them (the filter is overconfident and lags), raise σ and they fall below (underconfident).`}
    >
      <Dashboard>
        <DashboardRow minHeight={420}>
          <DashboardCell ratio={1.3}>
            <Plot x={px} y={py} renderer="canvas">
              <Curve name="truth" x={tracks.truth.x} y={tracks.truth.y} muted dashed />
              <Points name="measurements" x={tracks.seen.x} y={tracks.seen.y} slot={0} live />
              <Curve
                name="filter"
                x={tracks.filter.x}
                y={tracks.filter.y}
                slot={1}
                thin={smoothing}

                live
              />
              {tracks.ahead.x.length > 0 && <Points name="to come" x={tracks.ahead.x} y={tracks.ahead.y} muted />}
              {ellipses.prior && <Curve name="prior 95%" {...ellipses.prior} slot={1} dashed />}
              {ellipses.predict && (
                <Curve name="predict 95%" {...ellipses.predict} slot={1} dashed stale={computed.stale} />
              )}
              {/* Named (so in the legend) only while there is an update to show. */}
              <Curve
                name={!smoothing && t >= 0 ? 'update 95%' : undefined}
                {...(ellipses.update ?? EMPTY)}
                slot={1}
                live
              />
              {tracks.smoother && <Curve name="smoother" {...tracks.smoother} slot={2} stale={computed.stale} />}
              {ellipses.filter && <Curve name="filter 95%" {...ellipses.filter} slot={1} dashed />}
              {ellipses.smoother && <Curve name="smoother 95%" {...ellipses.smoother} slot={2} />}
              {current && (
                <Handle
                  kind="point"
                  at={current}
                  label={`y_${t + 1}`}
                  onDrag={(p) => setMoved((m) => ({ ...m, [t]: [p[0], p[1]] }))}
                />
              )}
            </Plot>
          </DashboardCell>
          <DashboardCell>
            <Plots rows={2}>
              <Plot x={time} y={sdAxis} renderer="canvas">
                <Curve name="filter" {...spread.filter} slot={1} />
                {spread.smoother && <Curve name="smoother" {...spread.smoother} slot={2} />}
              </Plot>
              <Plot x={time} y={nisAxis} renderer="canvas">
                <Points name="NIS" x={nisNow.x} y={nisNow.y} slot={0} live />
                <Curve name="χ²₂ 97.5%" x={[1, T]} y={[nisBand[1], nisBand[1]]} muted dashed />
                <Curve name="χ²₂ 2.5%" x={[1, T]} y={[nisBand[0], nisBand[0]]} muted dashed />
              </Plot>
            </Plots>
          </DashboardCell>
        </DashboardRow>
      </Dashboard>
    </Figure>
  )
}
