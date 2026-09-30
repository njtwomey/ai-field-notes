import { eigh2 } from 'aifn/linalg'
import { normals, stream } from 'aifn/random'
import { toFlat, toRows, type Tensor } from 'aifn/tensor'
import {
  armaAutocorrelation,
  armaRoots,
  exponentialSmoothing,
  garchProperties,
  isStationary,
  kalmanFilter,
  rtsSmoother,
  sampleAcf,
  samplePacf,
  simulateArma,
  simulateGarch,
  simulateStateSpace,
  smoothingFitter,
  stateSpaceEm,
  stl,
} from 'aifn/timeseries'
import { run, trace } from 'aifn/trace'
import { useMemo, useState } from 'react'
import { Button, Player, Select, Slider, Switch, usePlayhead } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { formatNumber, Panel, Readout, Subplots, XYChart, type Handle, type XYSeries } from '@lab/viz'
import { TraceView } from '@lab/views'

const flat = (t: Tensor) => toFlat(t)
const steps = (n: number, start = 1) => Array.from({ length: n }, (_, i) => i + start)
const fmt = (v: number) => formatNumber(v)
const NO_SERIES: XYSeries[] = []

/** The 2σ ellipse of a 2×2 covariance about a mean, from its eigendecomposition. */
function ellipse(mean: number[], cov: number[][], k = 2, points = 40) {
  const { values, vectors } = eigh2([
    [cov[0][0], cov[0][1]],
    [cov[1][0], cov[1][1]],
  ])
  const a = k * Math.sqrt(Math.max(values[0], 0))
  const b = k * Math.sqrt(Math.max(values[1], 0))
  const x: number[] = []
  const y: number[] = []
  for (let i = 0; i <= points; i++) {
    const t = (2 * Math.PI * i) / points
    const u = a * Math.cos(t)
    const v = b * Math.sin(t)
    x.push(mean[0] + u * vectors[0][0] + v * vectors[1][0])
    y.push(mean[1] + u * vectors[0][1] + v * vectors[1][1])
  }
  return { x, y }
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
  const [logQ, setLogQ] = useState(-2)
  const [logR, setLogR] = useState(0.5)
  const [seed, setSeed] = useState(4)
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
  const series = useMemo(
    (): XYSeries[] => [
      { name: 'true path', type: 'line', x: truth.map((r) => r[0]), y: truth.map((r) => r[1]), slot: 0 },
      { name: 'observations', type: 'scatter', x: obs.map((r) => r[0]), y: obs.map((r) => r[1]), muted: true },
    ],
    [truth, obs],
  )
  const path = (rows: number[][]) => ({ x: rows.map((r) => r[0]), y: rows.map((r) => r[1]) })
  // The filter's path: the updated estimates so far, ending at the prediction while it waits for y_t.
  const filterRows =
    phase === 'smooth' ? run.mean : [...run.mean.slice(0, t), phase === 'predict' ? run.pred[t] : run.mean[t]]
  const predicted = phase === 'smooth' ? NONE : ellipse(run.pred[t], positionCov(run.predCov, t))
  const updated = phase === 'update' ? ellipse(run.mean[t], positionCov(run.cov, t)) : NONE
  const smoothed = phase === 'smooth' ? path(run.smooth.slice(t)) : NONE
  const smoothedEllipse = phase === 'smooth' ? ellipse(run.smooth[t], positionCov(run.smoothCov, t)) : NONE
  const live: XYSeries[] = [
    { name: 'Kalman filter', type: 'line', slot: 1, ...path(filterRows) },
    { name: 'Kalman filter', type: 'line', slot: 1, dashed: true, ...predicted },
    { name: 'Kalman filter', type: 'line', slot: 1, ...updated },
    { name: 'RTS smoother', type: 'line', slot: 2, ...smoothed },
    { name: 'RTS smoother', type: 'line', slot: 2, ...smoothedEllipse },
    { name: 'observations', type: 'scatter', emphasis: true, ...(phase === 'update' ? path([obs[t]]) : NONE) },
  ]
  const P = positionCov(phase === 'smooth' ? run.smoothCov : phase === 'update' ? run.cov : run.predCov, t)
  const nu = run.innovation[t]
  return (
    <Figure
      title="Kalman filter and RTS smoother"
      description="The filter alternates a prediction from the motion model with an update from each observation; the smoother then runs back from the end and also uses the later observations, so its ellipses are smaller."
      defaultSize="L"
      controls={
        <>
          <ControlRow label="1 · model">
            <Slider label="log₁₀ process noise q" value={logQ} min={-4} max={0} step={0.25} onChange={setLogQ} />
            <Slider label="log₁₀ observation noise r" value={logR} min={-1} max={1.5} step={0.25} onChange={setLogR} />
            <Slider label="seed" value={seed} min={1} max={20} step={1} onChange={setSeed} />
          </ControlRow>
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
      readouts={
        <>
          <Readout label="step" value={`${phase} at t = ${t + 1}`} />
          <Readout label="position variance tr P" value={fmt(P[0][0] + P[1][1])} />
          <Readout
            label="innovation y_t − Cμ_{t|t−1}"
            value={phase === 'update' ? `(${fmt(nu[0])}, ${fmt(nu[1])})` : '–'}
          />
          <Readout label="filter RMSE" value={fmt(rmse(run.mean))} />
          <Readout label="smoother RMSE" value={fmt(rmse(run.smooth))} />
          <Readout label="raw observation RMSE" value={fmt(rmse(obs))} />
          <Readout label="log p(y)" value={fmt(run.filt.logLikelihood)} />
        </>
      }
      caption="A target moving with nearly constant velocity, observed with noise of variance r in each coordinate. Play or step: at each t the filter predicts the position from the motion model (dashed 2σ ellipse, which grows), then sees y_t (ink) and updates (solid ellipse, which shrinks). After t = 40 the RTS smoother runs back from the end; its ellipses are smallest in the middle of the track, where there are observations on both sides. With small q the filter trusts its motion model and smooths hard; with large q it follows the observations."
    >
      <XYChart
        series={series}
        live={live}
        xLabel="x"
        yLabel="y"
        aspect="equal"
        axisKey={seed}
        rescaleOnChange={false}
      />
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
  const [phi1, setPhi1] = useState(0.5)
  const [phi2, setPhi2] = useState(0.3)
  const [theta, setTheta] = useState(0)
  const [seed, setSeed] = useState(2)
  const [theory, setTheory] = useState(true)
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
  const handles: Handle[] = [
    {
      kind: 'point',
      at: [phi1, phi2],
      label: '(φ₁, φ₂)',
      onDrag: ([a, b]) => {
        setPhi1(Math.max(-2, Math.min(2, Math.round(a * 100) / 100)))
        setPhi2(Math.max(-1, Math.min(1, Math.round(b * 100) / 100)))
      },
    },
  ]
  const band = frame ? frame.band : 0
  const theorySeries = useMemo(
    (): XYSeries[] =>
      theory && stationary ? [{ name: 'theoretical ACF', type: 'line', x: lagX, y: rho.slice(1), slot: 2 }] : [],
    [theory, stationary, rho, lagX],
  )
  const acfLive: XYSeries[] = [
    { name: 'sample ACF', type: 'bar', x: frame ? lagX : [], y: frame ? frame.acf : [], slot: 0, thin: true },
    { name: 'sample PACF', type: 'scatter', x: frame ? lagX : [], y: frame ? frame.pacf : [], slot: 1 },
    { name: '±1.96/√t', type: 'line', x: [0.5, lags + 0.5], y: [band, band], muted: true, dashed: true },
    { name: '±1.96/√t', type: 'line', x: [0.5, lags + 0.5], y: [-band, -band], muted: true, dashed: true },
  ]
  const seriesLive: XYSeries[] = [
    { name: 'x', type: 'line', x: steps(t), y: xs.slice(0, t), slot: 0 },
    { name: 'x', type: 'scatter', x: t ? [t] : [], y: t ? [xs[t - 1]] : [], slot: 0 },
  ]
  return (
    <Figure
      title="ARMA processes and their autocorrelations"
      description="Inside the triangle the AR(2) is stationary; its ACF decays and its PACF cuts off after lag 2."
      defaultSize="XL"
      controls={
        <>
          <ControlRow label="1 · model">
            <Slider label="φ₁" value={phi1} min={-2} max={2} step={0.01} onChange={setPhi1} />
            <Slider label="φ₂" value={phi2} min={-1} max={1} step={0.01} onChange={setPhi2} />
            <Slider label="θ₁ (MA)" value={theta} min={-0.95} max={0.95} step={0.05} onChange={setTheta} />
          </ControlRow>
          <ControlRow label="2 · sample and reveal">
            <Slider label="seed" value={seed} min={1} max={20} step={1} onChange={setSeed} />
            <Switch label="theoretical ACF" checked={theory} onChange={setTheory} />
          </ControlRow>
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
      caption="Drag the point in the (φ₁, φ₂) plane. The AR polynomial 1 − φ₁z − φ₂z² is stationary inside the triangle, where both its roots (right panel, circles) lie outside the unit circle; below the parabola φ₁² + 4φ₂ = 0 the roots are complex and the ACF oscillates. Outside the triangle the series is simulated without clipping and explodes, and no ACF is drawn. Play to watch the series unfold: the sample ACF and PACF (bottom right) are computed from the values so far, so they settle towards the theoretical ACF as t grows. Dashed lines are the ±1.96/√t white-noise band."
    >
      <Subplots rows={2} cols={2} heightRatios={[1, 1]}>
        <Panel>
          <XYChart
            xLabel="φ₁"
            yLabel="φ₂"
            xRange={[-2.2, 2.2]}
            yRange={[-1.2, 1.2]}
            handles={handles}
            series={[
              { name: 'stationary region', type: 'line', x: TRIANGLE.x, y: TRIANGLE.y, muted: true },
              {
                name: 'complex roots below',
                type: 'line',
                x: steps(81, 0).map((i) => -2 + i / 20),
                y: steps(81, 0).map((i) => -((-2 + i / 20) ** 2) / 4),
                muted: true,
                dashed: true,
              },
              { name: '(φ₁, φ₂)', type: 'scatter', x: [phi1], y: [phi2], emphasis: true },
            ]}
          />
        </Panel>
        <Panel>
          <XYChart
            aspect="equal"
            xLabel="Re z"
            yLabel="Im z"
            series={[
              { name: 'unit circle', type: 'line', x: CIRCLE.x, y: CIRCLE.y, muted: true },
              { name: 'AR roots', type: 'scatter', x: flat(roots.ar.real), y: flat(roots.ar.imag), slot: 0 },
              { name: 'MA roots', type: 'scatter', x: flat(roots.ma.real), y: flat(roots.ma.imag), slot: 1 },
            ]}
          />
        </Panel>
        <Panel>
          <XYChart
            xLabel="t"
            yLabel="x_t"
            xRange={[0, n]}
            yRange={yRange}
            series={NO_SERIES}
            live={seriesLive}
            legend={false}
          />
        </Panel>
        <Panel>
          <XYChart
            xLabel="lag"
            yLabel="autocorrelation"
            integerX
            xRange={[0.5, lags + 0.5]}
            yRange={[-1, 1]}
            series={theorySeries}
            live={acfLive}
            legend
          />
        </Panel>
      </Subplots>
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
const HW_BACKGROUND: XYSeries[] = [{ name: 'data', type: 'line', x: steps(HW_N, 0), y: HW_DATA, muted: true }]

export function HoltWintersSpecimen() {
  const [seasonal, setSeasonal] = useState<Seasonality>('multiplicative')
  const [trend, setTrend] = useState<'additive' | 'damped'>('additive')
  const [alpha, setAlpha] = useState(0.3)
  const [beta, setBeta] = useState(0.1)
  const [gamma, setGamma] = useState(0.2)
  const [phi, setPhi] = useState(0.95)
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
    const best = run(smoothingFitter(HW_DATA, { trend, seasonal, period: PERIOD }), {}, 2000).params
    setAlpha(best.alpha)
    setBeta(best.beta ?? beta)
    setGamma(best.gamma ?? gamma)
    if (best.phi !== undefined) setPhi(best.phi)
  }
  const future = steps(HW_H, origin)
  return (
    <Figure
      title="Holt–Winters forecasts"
      description="Level, trend and season are updated by exponential smoothing; the forecast extends them with a widening interval."
      defaultSize="L"
      controls={
        <>
          <ControlRow label="1 · structure">
            <Select label="season" value={seasonal} onChange={setSeasonal} options={['multiplicative', 'additive']} />
            <Select label="trend" value={trend} onChange={setTrend} options={['additive', 'damped']} />
          </ControlRow>
          <ControlRow label="2 · smoothing">
            <Slider label="α (level)" value={alpha} min={0.01} max={1} step={0.01} onChange={setAlpha} />
            <Slider label="β* (trend)" value={beta} min={0} max={1} step={0.01} onChange={setBeta} />
            <Slider label="γ (season)" value={gamma} min={0} max={1} step={0.01} onChange={setGamma} />
            {trend === 'damped' && (
              <Slider label="φ (damping)" value={phi} min={0.8} max={1} step={0.005} onChange={setPhi} />
            )}
            <Button variant="outline" size="sm" onClick={fitBySse}>
              Fit by least squares
            </Button>
          </ControlRow>
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
      caption="A trending series whose seasonal swing grows with its level. The multiplicative model captures the growing swing; the additive one leaves it in the residuals, so its SSE is larger. The 95% interval widens with the horizon as the level, trend and season uncertainties accumulate. Play to move the forecast origin: the model sees only the data before it (the rest is grey), and the forecast from two seasons is poorer than from six. 'Fit by least squares' runs Nelder–Mead on the one-step SSE of the whole series."
    >
      <XYChart
        xLabel="t"
        yLabel="y"
        xRange={[0, HW_N + HW_H]}
        yRange={yRange}
        axisKey={`${seasonal}-${trend}`}
        rescaleOnChange={false}
        series={HW_BACKGROUND}
        live={[
          {
            name: 'data seen',
            type: 'line',
            x: steps(origin, 0),
            y: HW_DATA.slice(0, origin),
            showPoints: true,
            slot: 3,
          },
          { name: 'one-step forecasts', type: 'line', x: steps(origin, 0), y: flat(fit.fitted), slot: 0 },
          { name: 'forecast', type: 'line', x: future, y: flat(fit.forecast), slot: 1 },
          { name: '95% interval', type: 'line', x: future, y: flat(fit.upper), slot: 1, dashed: true },
          { name: '95% interval', type: 'line', x: future, y: flat(fit.lower), slot: 1, dashed: true },
        ]}
      />
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 4. GARCH volatility clustering.

/** Steps between the ACFs the player shows, and the fewest returns they are drawn from. */
const GARCH_EVERY = 10
const GARCH_MIN = 40

export function GarchSpecimen() {
  const [persistence, setPersistence] = useState(0.97)
  const [alpha, setAlpha] = useState(0.1)
  const [seed, setSeed] = useState(3)
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
  const returnsLive: XYSeries[] = [
    { name: 'returns', type: 'line', x, y: seen, slot: 0, thin: true },
    { name: '±2σ_t', type: 'line', x, y: sd.slice(0, t).map((v) => 2 * v), slot: 1 },
    { name: '±2σ_t', type: 'line', x, y: sd.slice(0, t).map((v) => -2 * v), slot: 1 },
  ]
  const acfLive: XYSeries[] = [
    { name: 'ACF of r', type: 'bar', x: frame ? lagX : [], y: frame ? frame.acf : [], slot: 0, thin: true },
    { name: 'ACF of r²', type: 'line', x: frame ? lagX : [], y: frame ? frame.acf2 : [], slot: 1, showPoints: true },
    {
      name: '±1.96/√t',
      type: 'line',
      x: [0.5, 30.5],
      y: frame ? [frame.band, frame.band] : [],
      muted: true,
      dashed: true,
    },
  ]
  return (
    <Figure
      title="GARCH(1,1) volatility clustering"
      description="Returns are uncorrelated, but their squares are not: large moves cluster, and the tails are heavier than normal."
      defaultSize="L"
      controls={
        <>
          <ControlRow label="1 · model">
            <Slider
              label="persistence α + β"
              value={persistence}
              min={0}
              max={0.995}
              step={0.005}
              onChange={setPersistence}
            />
            <Slider label="α (reaction)" value={alpha} min={0} max={0.3} step={0.01} onChange={setAlpha} />
            <Slider label="seed" value={seed} min={1} max={30} step={1} onChange={setSeed} />
          </ControlRow>
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
      <Subplots rows={2} heightRatios={[3, 2]}>
        <Panel>
          <XYChart
            xLabel="t"
            yLabel="r_t"
            xRange={[0, n]}
            yRange={rRange}
            series={NO_SERIES}
            live={returnsLive}
            legend
          />
        </Panel>
        <Panel>
          <XYChart
            xLabel="lag"
            yLabel="ACF"
            integerX
            xRange={[0.5, 30.5]}
            yRange={acfRange}
            series={NO_SERIES}
            live={acfLive}
            legend
          />
        </Panel>
      </Subplots>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 5. EM for a linear-Gaussian state-space model.

const EM_TRUTH = { A: 0.95, C: 1, Q: 0.2, R: 1, m0: 0, P0: 1 }
const EM_Y = simulateStateSpace(stream('em'), EM_TRUTH, 200).observations

export function EmSpecimen() {
  const [a0, setA0] = useState(0.3)
  const t = useMemo(
    () =>
      trace(stateSpaceEm(EM_Y, { A: a0, C: 1, Q: 1, R: 3, m0: 0, P0: 1 }, { estimate: { C: false } }), {}, 150, {
        record: {
          'log p(y)': (s) => s.logLikelihood,
          A: (s) => flat(s.model.A)[0],
          Q: (s) => flat(s.model.Q)[0],
          R: (s) => flat(s.model.R)[0],
        },
      }),
    [a0],
  )
  const y = flat(EM_Y)
  return (
    <TraceView
      title="EM for a state-space model"
      description="Each EM step smooths with the current model and re-estimates A, Q and R from the smoothed moments; the likelihood never falls."
      trace={t}
      show={['log p(y)', 'A', 'Q', 'R']}
      startAtFirst
      defaultSize="L"
      controls={<Slider label="starting A" value={a0} min={-0.9} max={0.99} step={0.01} onChange={setA0} />}
      caption={`200 observations of an AR(1) state (A = ${EM_TRUTH.A}, Q = ${EM_TRUTH.Q}) seen through noise of variance R = ${EM_TRUTH.R}, with C fixed at 1 (otherwise the scale of the state is not identified). Scrub the steps: the smoothed state under the current model tightens around the data as A rises towards its maximum-likelihood value.`}
      renderState={(s) => {
        const sm = rtsSmoother(s.current, EM_Y)
        return (
          <XYChart
            xLabel="t"
            yLabel="y, E[z_t | y]"
            rescaleOnChange={false}
            series={[
              { name: 'observations', type: 'scatter', x: steps(y.length), y, muted: true },
              { name: 'smoothed state', type: 'line', x: steps(y.length), y: flat(sm.mean), slot: 0 },
            ]}
          />
        )
      }}
    />
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
  const [span, setSpan] = useState(7)
  const d = useMemo(() => stl(STL_DATA, 12, { seasonalSpan: span }), [span])
  const x = steps(STL_N, 0)
  return (
    <Figure
      title="Seasonal-trend decomposition by loess"
      description="STL splits a series into a smooth trend, a seasonal pattern that may drift, and a remainder."
      controls={<Slider label="seasonal span (odd)" value={span} min={7} max={51} step={2} onChange={setSpan} />}
      caption="A slow wave plus a period-12 season plus noise. A short seasonal span lets the seasonal pattern change from year to year and absorb some noise; a long span forces it towards one fixed pattern."
    >
      <Subplots rows={3} sharex hoverGroup>
        <Panel>
          <XYChart
            xLabel="t"
            yLabel="y, trend"
            series={[
              { name: 'y', type: 'line', x, y: STL_DATA, muted: true },
              { name: 'trend', type: 'line', x, y: flat(d.trend), slot: 0 },
            ]}
          />
        </Panel>
        <Panel>
          <XYChart
            xLabel="t"
            yLabel="seasonal"
            series={[{ name: 'seasonal', type: 'line', x, y: flat(d.seasonal), slot: 1 }]}
          />
        </Panel>
        <Panel>
          <XYChart
            xLabel="t"
            yLabel="remainder"
            series={[{ name: 'remainder', type: 'bar', x, y: flat(d.remainder), slot: 2 }]}
          />
        </Panel>
      </Subplots>
    </Figure>
  )
}
