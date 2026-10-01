import { covarianceEllipse } from 'aifn/numerics/geometry'
import { normals, stream } from 'aifn/foundation/random'
import { imagPart, realPart, toFlat, toRows, type Tensor } from 'aifn/foundation/tensor'
import {
  armaAutocorrelation,
  armaRoots,
  exponentialSmoothing,
  exponentialSmoothingFitSteps,
  garchProperties,
  isStationary,
  simulateArma,
  simulateGarch,
  stateSpaceEm,
  stl,
} from 'aifn-applied/timeseries'
import { kalmanFilter, rtsSmoother, simulateStateSpace } from 'aifn/inference/filtering'
import { sampleAcf, samplePacf } from 'aifn/probability/stats'
import { run, trace } from 'aifn/foundation/trace'
import { useMemo } from 'react'
import { Button, Player, usePlayhead } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { choice, number, row, slider, toggle, useFigureState } from '@lab/state'
import { Area, Bars, Curve, formatNumber, Handle, Plot, Plots, Points, Readout, useAxis } from '@lab/viz'
import { TracePanel } from '@lab/views'

const flat = (t: Tensor) => toFlat(t)
const steps = (n: number, start = 1) => Array.from({ length: n }, (_, i) => i + start)
const fmt = (v: number) => formatNumber(v)

/** The k-σ ellipse of a 2×2 covariance about a mean (aifn covarianceEllipse), as x and y arrays. */
function ellipse(mean: number[], cov: number[][], k = 2) {
  const rows = toRows(covarianceEllipse([mean[0], mean[1]], cov, { k }).points) as number[][]
  return { x: rows.map((r) => r[0]), y: rows.map((r) => r[1]) }
}

// ---------------------------------------------------------------------------------------------------------------------
// 1. Kalman filter and RTS smoother tracking a target in the plane.

/** Constant-velocity motion with unit time step: state (x, y, vx, vy), position observed. */
function constantVelocity(q: number, r: number) {
  return {
    A: [
      [1, 0, 1, 0],
      [0, 1, 0, 1],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ],
    C: [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
    ],
    // Discretised white-noise acceleration of intensity q.
    Q: [
      [q / 3, 0, q / 2, 0],
      [0, q / 3, 0, q / 2],
      [q / 2, 0, q, 0],
      [0, q / 2, 0, q],
    ],
    R: [
      [r, 0],
      [0, r],
    ],
    m0: [0, 0, 0.5, 0.2],
    P0: [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 0.1, 0],
      [0, 0, 0, 0.1],
    ],
  }
}

const KALMAN_T = 40

/** The 2×2 position block of the t-th n×n covariance in a flat [T, n, n] array. */
const positionCov = (covs: ArrayLike<number>, t: number, n = 4): number[][] => {
  const o = t * n * n
  return [
    [covs[o], covs[o + 1]],
    [covs[o + n], covs[o + n + 1]],
  ]
}

/** What the player shows at a position: the filter's predict and update at each t, then the smoother's pass back. */
function kalmanPhase(position: number, T: number): { t: number; phase: 'predict' | 'update' | 'smooth' } {
  if (position < 2 * T) return { t: Math.floor(position / 2), phase: position % 2 ? 'update' : 'predict' }
  return { t: T - 1 - (position - 2 * T), phase: 'smooth' }
}

const NONE = { x: [] as number[], y: [] as number[] }

export function KalmanTrackingSpecimen() {
  const state = useFigureState({
    model: row('1 · model', {
      logQ: slider(-4, 0, -2, { label: 'log₁₀ process noise q', step: 0.25 }),
      logR: slider(-1, 1.5, 0.5, { label: 'log₁₀ observation noise r', step: 0.25 }),
      seed: number(4, { min: 1, max: 20, step: 1, label: 'seed' }),
    }),
  })
  const { logQ, logR, seed } = state.model
  const T = KALMAN_T
  const model = useMemo(() => constantVelocity(10 ** logQ, 10 ** logR), [logQ, logR])
  const sim = useMemo(() => simulateStateSpace(stream(seed), model, KALMAN_T), [model, seed])
  const run = useMemo(() => {
    const filt = kalmanFilter(model, sim.observations)
    const sm = rtsSmoother(model, sim.observations)
    return {
      filt,
      sm,
      pred: toRows(filt.predictedMean),
      predCov: toFlat(filt.predictedCov),
      mean: toRows(filt.mean),
      cov: toFlat(filt.cov),
      smooth: toRows(sm.mean),
      smoothCov: toFlat(sm.cov),
      innovation: toRows(filt.innovation),
    }
  }, [model, sim])
  const truth = useMemo(() => toRows(sim.states), [sim])
  const obs = useMemo(() => toRows(sim.observations), [sim])
  const [position, setPosition] = usePlayhead(3 * T)
  const { t, phase } = kalmanPhase(position, T)

  const rmse = (rows: number[][]) =>
    Math.sqrt(rows.reduce((s, r, k) => s + (r[0] - truth[k][0]) ** 2 + (r[1] - truth[k][1]) ** 2, 0) / T)
  const path = (rows: number[][]) => ({ x: rows.map((r) => r[0]), y: rows.map((r) => r[1]) })
  const truePath = useMemo(() => path(truth), [truth])
  const observed = useMemo(() => path(obs), [obs])
  // The filter's path: the updated estimates so far, ending at the prediction while it waits for y_t.
  const filterRows =
    phase === 'smooth' ? run.mean : [...run.mean.slice(0, t), phase === 'predict' ? run.pred[t] : run.mean[t]]
  const filterPath = path(filterRows)
  const predicted = phase === 'smooth' ? NONE : ellipse(run.pred[t], positionCov(run.predCov, t))
  const updated = phase === 'update' ? ellipse(run.mean[t], positionCov(run.cov, t)) : NONE
  const smoothed = phase === 'smooth' ? path(run.smooth.slice(t)) : NONE
  const smoothedEllipse = phase === 'smooth' ? ellipse(run.smooth[t], positionCov(run.smoothCov, t)) : NONE
  const seen = phase === 'update' ? path([obs[t]]) : NONE
  const xa = useAxis({ label: 'x', hold: 'initial', key: seed })
  const ya = useAxis({ label: 'y', hold: 'initial', key: seed, equal: xa })
  const P = positionCov(phase === 'smooth' ? run.smoothCov : phase === 'update' ? run.cov : run.predCov, t)
  const nu = run.innovation[t]
  return (
    <Figure
      title="Kalman filter and RTS smoother"
      purpose="The filter alternates a prediction from the motion model with an update from each observation; the smoother then runs back from the end and also uses the later observations, so its ellipses are smaller."
      state={state}
      defaultSize="L"
      controls={
        <>
          <ControlRow label="2 · steps">
            <Player
              value={position}
              onChange={setPosition}
              count={3 * T}
              duration={8}
              label="filter step, then smoother pass"
              format={(p) => {
                const s = kalmanPhase(p, T)
                return `t = ${s.t + 1} · ${s.phase}`
              }}
            />
          </ControlRow>
        </>
      }
      readouts={{
        'this step': (
          <>
            <Readout label="step" value={`${phase} at t = ${t + 1}`} />
            <Readout label="position variance tr P" value={fmt(P[0][0] + P[1][1])} />
            <Readout
              label="innovation y_t − Cμ_{t|t−1}"
              value={phase === 'update' ? `(${fmt(nu[0])}, ${fmt(nu[1])})` : '–'}
            />
          </>
        ),
        'whole run': (
          <>
            <Readout label="filter RMSE" value={fmt(rmse(run.mean))} />
            <Readout label="smoother RMSE" value={fmt(rmse(run.smooth))} />
            <Readout label="raw observation RMSE" value={fmt(rmse(obs))} />
            <Readout label="log p(y)" value={fmt(run.filt.logLikelihood)} />
          </>
        ),
      }}
      caption="A target moving with nearly constant velocity, observed with noise of variance r in each coordinate. Play or step: at each t the filter predicts the position from the motion model (dashed 2σ ellipse, which grows), then sees y_t (ink) and updates (solid ellipse, which shrinks). After t = 40 the RTS smoother runs back from the end; its ellipses are smallest in the middle of the track, where there are observations on both sides. With small q the filter trusts its motion model and smooths hard; with large q it follows the observations."
    >
      <Plot x={xa} y={ya}>
        <Curve name="true path" x={truePath.x} y={truePath.y} slot={0} />
        <Points name="observations" x={observed.x} y={observed.y} muted />
        <Curve name="Kalman filter" x={filterPath.x} y={filterPath.y} slot={1} live />
        <Curve name="predicted 2σ" x={predicted.x} y={predicted.y} slot={1} dashed live />
        <Curve name="updated 2σ" x={updated.x} y={updated.y} slot={1} live />
        <Curve name="RTS smoother" x={smoothed.x} y={smoothed.y} slot={2} live />
        <Curve name="smoothed 2σ" x={smoothedEllipse.x} y={smoothedEllipse.y} slot={2} live />
        <Points name="y_t" x={seen.x} y={seen.y} emphasis live />
      </Plot>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 2. ARMA processes: stationarity triangle, series, ACF and PACF.

/** The fewest values the sample ACF is drawn from. */
const MIN_ACF = 30

const TRIANGLE = { x: [-2, 0, 2, -2], y: [-1, 1, -1, -1] }
const CIRCLE = {
  x: Array.from({ length: 121 }, (_, i) => Math.cos((2 * Math.PI * i) / 120)),
  y: Array.from({ length: 121 }, (_, i) => Math.sin((2 * Math.PI * i) / 120)),
}

export function ArmaSpecimen() {
  const state = useFigureState({
    model: row('1 · model', {
      phi1: slider(-2, 2, 0.5, { label: 'φ₁', step: 0.01 }),
      phi2: slider(-1, 1, 0.3, { label: 'φ₂', step: 0.01 }),
      theta: slider(-0.95, 0.95, 0, { label: 'θ₁ (MA)', step: 0.05 }),
    }),
    sample: row('2 · sample and reveal', {
      seed: number(2, { min: 1, max: 20, step: 1, label: 'seed' }),
      theory: toggle(true, 'theoretical ACF'),
    }),
  })
  const { phi1, phi2, theta } = state.model
  const { seed, theory } = state.sample
  const lags = 24
  const n = 300
  const spec = useMemo(() => ({ ar: [phi1, phi2], ma: [theta] }), [phi1, phi2, theta])
  const stationary = isStationary(spec.ar)
  const sim = useMemo(() => simulateArma(stream(seed), spec, n), [spec, seed])
  const xs = useMemo(() => flat(sim.x), [sim])
  // The sample ACF and PACF of the first t values, for every t the player can show (none below MIN_ACF).
  const frames = useMemo(() => {
    if (!stationary) return null
    return steps(n + 1, 0).map((t) => {
      if (t < MIN_ACF) return null
      const a = sampleAcf(xs.slice(0, t), lags)
      return { acf: flat(a.acf).slice(1), pacf: flat(samplePacf(xs.slice(0, t), lags)), band: a.band }
    })
  }, [xs, stationary])
  const rho = useMemo(() => (stationary ? flat(armaAutocorrelation(spec, lags)) : []), [spec, stationary])
  const [t, setT] = usePlayhead(n + 1, 1)
  const frame = frames?.[t] ?? null
  const roots = armaRoots(spec)
  const lagX = useMemo(() => steps(lags), [])
  const yRange = useMemo((): [number, number] => {
    const finite = xs.filter(Number.isFinite)
    const lo = Math.min(...finite)
    const hi = Math.max(...finite)
    const pad = 0.05 * (hi - lo || 1)
    return [lo - pad, hi + pad]
  }, [xs])
  const band = frame ? frame.band : 0
  const parabola = useMemo(() => {
    const u = steps(81, 0).map((i) => -2 + i / 20)
    return { x: u, y: u.map((v) => -(v * v) / 4) }
  }, [])
  const arRoots = { x: flat(realPart(roots.ar.roots)), y: flat(imagPart(roots.ar.roots)) }
  const maRoots = { x: flat(realPart(roots.ma.roots)), y: flat(imagPart(roots.ma.roots)) }
  const seen = { x: steps(t), y: xs.slice(0, t) }
  const last = t ? { x: [t], y: [xs[t - 1]] } : { x: [], y: [] }
  const bandX = [0.5, lags + 0.5]
  const phi1Axis = useAxis({ label: 'φ₁', range: [-2.2, 2.2] })
  const phi2Axis = useAxis({ label: 'φ₂', range: [-1.2, 1.2] })
  const re = useAxis({ label: 'Re z' })
  const im = useAxis({ label: 'Im z', equal: re })
  const time = useAxis({ label: 't', range: [0, n] })
  const xAxis = useAxis({ label: 'x_t', range: yRange })
  const lagAxis = useAxis({ label: 'lag', range: [0.5, lags + 0.5] })
  const acfAxis = useAxis({ label: 'autocorrelation', range: [-1, 1] })
  return (
    <Figure
      title="ARMA processes and their autocorrelations"
      purpose="Inside the triangle the AR(2) is stationary: both AR roots lie outside the unit circle, its ACF decays and its PACF cuts off after lag 2."
      state={state}
      defaultSize="XL"
      controls={
        <>
          <ControlRow label="3 · time">
            <Player
              value={t}
              onChange={setT}
              count={n + 1}
              duration={4}
              label="t"
              format={(k) => `t = ${k}`}
              startReason="The ACF and PACF are read from the whole sample, so the figure opens with all 300 values."
            />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout label="stationary" value={stationary ? 'yes' : 'no'} />
          <Readout label="smallest AR root modulus" value={fmt(roots.ar.minModulus)} />
          <Readout label="MA invertible" value={roots.ma.minModulus > 1 ? 'yes' : 'no'} />
          <Readout label="values in the sample ACF" value={frame ? t : `– (fewer than ${MIN_ACF})`} />
          {sim.diverged && <Readout label="series overflowed at" value={`t = ${sim.divergedAt}`} />}
        </>
      }
      caption="Drag the point in the (φ₁, φ₂) plane (the shaded triangle is the stationary region). The AR polynomial 1 − φ₁z − φ₂z² is stationary inside the triangle, where both its roots (right panel, circles) lie outside the unit circle; below the parabola φ₁² + 4φ₂ = 0 the roots are complex and the ACF oscillates. Outside the triangle the series is simulated without clipping and explodes, and no ACF is drawn. Play to watch the series unfold: the sample ACF and PACF (bottom right) are computed from the values so far, so they settle towards the theoretical ACF as t grows. Dashed lines are the ±1.96/√t white-noise band."
    >
      <Plots rows={2} cols={2}>
        <Plot x={phi1Axis} y={phi2Axis}>
          <Area name="stationary region" x={TRIANGLE.x} y={TRIANGLE.y} muted opacity={0.12} />
          <Curve name="complex roots below" x={parabola.x} y={parabola.y} muted dashed />
          <Handle {...state.handle(['model.phi1', 'model.phi2'], { label: '(φ₁, φ₂)' })} />
        </Plot>
        <Plot x={re} y={im}>
          <Curve name="unit circle" x={CIRCLE.x} y={CIRCLE.y} muted />
          <Points name="AR roots" x={arRoots.x} y={arRoots.y} slot={0} />
          <Points name="MA roots" x={maRoots.x} y={maRoots.y} slot={1} />
        </Plot>
        <Plot x={time} y={xAxis} legend={false}>
          <Curve name="x" x={seen.x} y={seen.y} slot={0} live />
          <Points name="x_t" x={last.x} y={last.y} slot={0} live />
        </Plot>
        <Plot x={lagAxis} y={acfAxis}>
          {theory && stationary && <Curve name="theoretical ACF" x={lagX} y={rho.slice(1)} slot={2} />}
          <Bars name="sample ACF" x={frame ? lagX : []} y={frame ? frame.acf : []} slot={0} width={0.5} live />
          <Points name="sample PACF" x={frame ? lagX : []} y={frame ? frame.pacf : []} slot={1} live />
          <Curve name="±1.96/√t" x={bandX} y={[band, band]} muted dashed live />
          <Curve name="±1.96/√t" x={bandX} y={[-band, -band]} muted dashed live />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 3. Holt–Winters forecasts.

const PERIOD = 12
const HW_N = 72
const HW_H = 24
const SEASON = Array.from({ length: PERIOD }, (_, j) => 1 + 0.25 * Math.sin((2 * Math.PI * j) / PERIOD))
const HW_DATA = (() => {
  const e = flat(normals(stream('holt-winters'), HW_N, 0, 1.5))
  // A trending level with a multiplicative season: the season's swing grows with the level.
  return steps(HW_N, 0).map((t) => (20 + 0.4 * t) * SEASON[t % PERIOD] + e[t])
})()

type Seasonality = 'additive' | 'multiplicative'

/** Forecast origins from two seasons of data to the end. */
const ORIGINS = steps(HW_N - 2 * PERIOD + 1, 2 * PERIOD)
const HW_T = steps(HW_N, 0)

export function HoltWintersSpecimen() {
  const state = useFigureState({
    structure: row('1 · structure', {
      seasonal: choice(['multiplicative', 'additive'] as Seasonality[], 'multiplicative', { label: 'season' }),
      trend: choice(['additive', 'damped'], 'additive', { label: 'trend' }),
    }),
    smoothing: row('2 · smoothing', {
      alpha: slider(0.01, 1, 0.3, { label: 'α (level)', step: 0.01 }),
      beta: slider(0, 1, 0.1, { label: 'β* (trend)', step: 0.01 }),
      gamma: slider(0, 1, 0.2, { label: 'γ (season)', step: 0.01 }),
      phi: slider(0.8, 1, 0.95, { label: 'φ (damping)', step: 0.005, when: (v) => v.trend === 'damped' }),
    }),
  })
  const seasonal = state.structure.seasonal as Seasonality
  const trend = state.structure.trend as 'additive' | 'damped'
  const { alpha, beta, gamma, phi } = state.smoothing
  const spec = useMemo(
    () => ({ trend, seasonal, period: PERIOD, alpha, beta, gamma, phi }),
    [trend, seasonal, alpha, beta, gamma, phi],
  )
  // One fit per forecast origin: the model sees data[0 … origin − 1] and forecasts HW_H steps on from there.
  const fits = useMemo(
    () => ORIGINS.map((origin) => exponentialSmoothing(HW_DATA.slice(0, origin), spec, { horizon: HW_H })),
    [spec],
  )
  const [k, setK] = usePlayhead(ORIGINS.length)
  const origin = ORIGINS[k]
  const fit = fits[k]
  const yRange = useMemo((): [number, number] => {
    const all = [...HW_DATA, ...fits.flatMap((f) => [...flat(f.upper), ...flat(f.lower)])].filter(Number.isFinite)
    return [Math.min(0, ...all), Math.max(...all)]
  }, [fits])
  const fitBySse = () => {
    const best = run(exponentialSmoothingFitSteps(HW_DATA, { trend, seasonal, period: PERIOD }), undefined, 2000).params
    state.set('smoothing.alpha', best.alpha)
    state.set('smoothing.beta', best.beta ?? beta)
    state.set('smoothing.gamma', best.gamma ?? gamma)
    if (best.phi !== undefined) state.set('smoothing.phi', best.phi)
  }
  const future = steps(HW_H, origin)
  const seenT = steps(origin, 0)
  const ta = useAxis({ label: 't', range: [0, HW_N + HW_H] })
  const ya = useAxis({ label: 'y', range: yRange })
  return (
    <Figure
      title="Holt–Winters forecasts"
      purpose="Level, trend and season are updated by exponential smoothing; the forecast extends them with an interval that widens with the horizon."
      state={state}
      defaultSize="L"
      controls={
        <>
          <Button variant="outline" size="sm" onClick={fitBySse}>
            Fit α, β*, γ by least squares
          </Button>
          <ControlRow label="3 · forecast origin">
            <Player
              value={k}
              onChange={setK}
              count={ORIGINS.length}
              duration={4}
              label="origin"
              format={(p) => `t = ${ORIGINS[p]}`}
            />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout label="forecast origin" value={`t = ${origin} (${fmt(origin / PERIOD)} seasons seen)`} />
          <Readout label="SSE" value={fmt(fit.sse)} />
          <Readout label="σ̂" value={fmt(Math.sqrt(fit.sigma2))} />
          <Readout label="α, β*, γ" value={`${fmt(alpha)}, ${fmt(beta)}, ${fmt(gamma)}`} />
          {fit.approximateIntervals && <Readout label="intervals" value="additive-error approximation" />}
        </>
      }
      caption="A trending series whose seasonal swing grows with its level. The multiplicative model captures the growing swing; the additive one leaves it in the residuals, so its SSE is larger. The 95% interval widens with the horizon as the level, trend and season uncertainties accumulate. Play to move the forecast origin: the model sees only the data before it (the rest is grey), and the forecast from two seasons is poorer than from six. 'Fit α, β*, γ by least squares' runs Nelder–Mead on the one-step SSE of the whole series."
    >
      <Plot x={ta} y={ya}>
        <Curve name="data" x={HW_T} y={HW_DATA} muted />
        <Curve name="data seen" x={seenT} y={HW_DATA.slice(0, origin)} showPoints slot={3} live />
        <Curve name="one-step forecasts" x={seenT} y={flat(fit.fitted)} slot={0} live />
        <Area
          name="95% interval"
          x={future}
          y={flat(fit.upper)}
          base={flat(fit.lower)}
          slot={1}
          opacity={0.18}
          line={false}
          live
        />
        <Curve name="forecast" x={future} y={flat(fit.forecast)} slot={1} live />
      </Plot>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 4. GARCH volatility clustering.

/** Steps between the ACFs the player shows, and the fewest returns they are drawn from. */
const GARCH_EVERY = 10
const GARCH_MIN = 40

export function GarchSpecimen() {
  const state = useFigureState({
    model: row('1 · model', {
      persistence: slider(0, 0.995, 0.97, { label: 'persistence α + β', step: 0.005 }),
      alpha: slider(0, 0.3, 0.1, { label: 'α (reaction)', step: 0.01 }),
      seed: number(3, { min: 1, max: 30, step: 1, label: 'seed' }),
    }),
  })
  const { persistence, alpha, seed } = state.model
  const a = Math.min(alpha, persistence)
  const spec = useMemo(() => ({ omega: 1 - persistence, alpha: a, beta: persistence - a }), [persistence, a])
  const n = 1000
  const sim = useMemo(() => simulateGarch(stream(seed), spec, n), [seed, spec])
  const r = useMemo(() => flat(sim.returns), [sim])
  const sd = useMemo(() => flat(sim.variance).map(Math.sqrt), [sim])
  // The ACFs of the returns seen so far, every GARCH_EVERY steps (none below GARCH_MIN values).
  const frames = useMemo(
    () =>
      steps(n / GARCH_EVERY + 1, 0).map((f) => {
        const t = f * GARCH_EVERY
        if (t < GARCH_MIN) return null
        const seen = r.slice(0, t)
        const a = sampleAcf(seen, 30)
        return {
          acf: flat(a.acf).slice(1),
          acf2: flat(
            sampleAcf(
              seen.map((v) => v * v),
              30,
            ).acf,
          ).slice(1),
          band: a.band,
        }
      }),
    [r],
  )
  const acfRange = useMemo((): [number, number] => {
    const all = frames.flatMap((f) => (f ? [...f.acf, ...f.acf2, f.band, -f.band] : []))
    return [Math.min(-0.1, ...all) - 0.05, Math.max(0.2, ...all) + 0.05]
  }, [frames])
  const rRange = useMemo((): [number, number] => {
    const m = Math.max(...r.map(Math.abs), ...sd.map((v) => 2 * v)) * 1.05
    return [-m, m]
  }, [r, sd])
  const [t, setT] = usePlayhead(n + 1, 1)
  const frame = frames[Math.floor(t / GARCH_EVERY)]
  const props = garchProperties(spec)
  const seen = r.slice(0, t)
  const m2 = seen.reduce((s, v) => s + v * v, 0) / Math.max(1, t)
  const m4 = seen.reduce((s, v) => s + v ** 4, 0) / Math.max(1, t)
  const lagX = steps(30)
  const x = steps(t)
  const upper = sd.slice(0, t).map((v) => 2 * v)
  const lower = upper.map((v) => -v)
  const ta = useAxis({ label: 't', range: [0, n] })
  const ra = useAxis({ label: 'r_t', range: rRange })
  const lagAxis = useAxis({ label: 'lag', range: [0.5, 30.5] })
  const acfAxis = useAxis({ label: 'ACF', range: acfRange })
  return (
    <Figure
      title="GARCH(1,1) volatility clustering"
      purpose="Returns are uncorrelated, but their squares are not: large moves cluster, and the tails are heavier than normal."
      state={state}
      defaultSize="L"
      controls={
        <>
          <ControlRow label="2 · time">
            <Player
              value={t}
              onChange={setT}
              count={n + 1}
              duration={5}
              label="t"
              format={(k) => `t = ${k}`}
              startReason="Volatility clustering and the ACF of r² need the whole series, so the figure opens with all 1000 returns."
            />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout label="β" value={fmt(spec.beta)} />
          <Readout label="sample kurtosis (so far)" value={t > 1 ? fmt(m4 / (m2 * m2)) : '–'} />
          <Readout label="theoretical kurtosis" value={Number.isFinite(props.kurtosis) ? fmt(props.kurtosis) : '∞'} />
        </>
      }
      caption="ω = 1 − (α + β) keeps the unconditional variance at 1. The top panel shows returns with ±2σ_t; the bottom compares the autocorrelations of r_t (near zero) and r_t² (slowly decaying, at a rate set by the persistence). With α = 0 the variance is constant and both ACFs vanish. Play to watch the series unfold: the ACFs are computed from the returns so far (from 40 values), so the slow decay of the ACF of r² emerges as calm and turbulent stretches accumulate."
    >
      <Plots rows={2} heights={[3, 2]}>
        <Plot x={ta} y={ra}>
          <Curve name="returns" x={x} y={seen} slot={0} thin live />
          <Curve name="±2σ_t" x={x} y={upper} slot={1} live />
          <Curve name="±2σ_t" x={x} y={lower} slot={1} live />
        </Plot>
        <Plot x={lagAxis} y={acfAxis}>
          <Bars name="ACF of r" x={frame ? lagX : []} y={frame ? frame.acf : []} slot={0} width={0.5} live />
          <Curve name="ACF of r²" x={frame ? lagX : []} y={frame ? frame.acf2 : []} slot={1} showPoints live />
          <Curve name="±1.96/√t" x={[0.5, 30.5]} y={frame ? [frame.band, frame.band] : []} muted dashed live />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 5. EM for a linear-Gaussian state-space model.

const EM_TRUTH = { A: 0.95, C: 1, Q: 0.2, R: 1, m0: 0, P0: 1 }
const EM_Y = simulateStateSpace(stream('em'), EM_TRUTH, 200).observations

const EM_YS = flat(EM_Y)
const EM_T = steps(EM_YS.length)

export function EmSpecimen() {
  const state = useFigureState({ a0: slider(-0.9, 0.99, 0.3, { label: 'starting A', step: 0.01 }) })
  const { a0 } = state
  const t = useMemo(
    () =>
      trace(stateSpaceEm(EM_Y, { A: a0, C: 1, Q: 1, R: 3, m0: 0, P0: 1 }, { estimate: { C: false } }), undefined, 150, {
        record: {
          'log p(y)': (s) => s.logLikelihood,
          A: (s) => flat(s.model.A)[0],
          Q: (s) => flat(s.model.Q)[0],
          R: (s) => flat(s.model.R)[0],
        },
      }),
    [a0],
  )
  const ta = useAxis({ label: 't' })
  const ya = useAxis({ label: 'y, E[z_t | y]', hold: 'initial' })
  return (
    <Figure
      title="EM for a state-space model"
      purpose="Each EM step smooths with the current model and re-estimates A, Q and R from the smoothed moments; the likelihood never falls."
      state={state}
      defaultSize="L"
      caption={`200 observations of an AR(1) state (A = ${EM_TRUTH.A}, Q = ${EM_TRUTH.Q}) seen through noise of variance R = ${EM_TRUTH.R}, with C fixed at 1 (otherwise the scale of the state is not identified). Scrub the steps: the smoothed state under the current model tightens around the data as A rises towards its maximum-likelihood value.`}
    >
      <TracePanel
        trace={t}
        show={['log p(y)', 'A', 'Q', 'R']}
        renderState={(s) => {
          const sm = rtsSmoother(s.current, EM_Y)
          return (
            <Plot x={ta} y={ya}>
              <Points name="observations" x={EM_T} y={EM_YS} muted thin />
              <Curve name="smoothed state" x={EM_T} y={flat(sm.mean)} slot={0} />
            </Plot>
          )
        }}
      />
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 6. Seasonal decomposition by STL.

const STL_N = 96
const STL_DATA = (() => {
  const e = flat(normals(stream('stl'), STL_N, 0, 0.6))
  return steps(STL_N, 0).map((t) => 5 + 3 * Math.sin(t / 15) + 2 * Math.sin((2 * Math.PI * t) / 12) + e[t])
})()

export function StlSpecimen() {
  const state = useFigureState({ span: slider(7, 51, 7, { label: 'seasonal span (odd)', step: 2 }) })
  const { span } = state
  const d = useMemo(() => stl(STL_DATA, 12, { seasonalSpan: span }), [span])
  const x = steps(STL_N, 0)
  const ta = useAxis({ label: 't' })
  const ya = useAxis({ label: 'y, trend' })
  const sa = useAxis({ label: 'seasonal', hold: 'union' })
  const ra = useAxis({ label: 'remainder', hold: 'union' })
  return (
    <Figure
      title="Seasonal-trend decomposition by loess"
      purpose="STL splits a series into a smooth trend, a seasonal pattern that may drift, and a remainder; the seasonal span sets how fast the pattern may drift."
      state={state}
      caption="A slow wave plus a period-12 season plus noise. A short seasonal span lets the seasonal pattern change from year to year and absorb some noise; a long span forces it towards one fixed pattern."
    >
      <Plots rows={3} hoverGroup>
        <Plot x={ta} y={ya}>
          <Curve name="y" x={x} y={STL_DATA} muted />
          <Curve name="trend" x={x} y={flat(d.trend)} slot={0} />
        </Plot>
        <Plot x={ta} y={sa}>
          <Curve name="seasonal" x={x} y={flat(d.seasonal)} slot={1} />
        </Plot>
        <Plot x={ta} y={ra}>
          <Bars name="remainder" x={x} y={flat(d.remainder)} slot={2} />
        </Plot>
      </Plots>
    </Figure>
  )
}
