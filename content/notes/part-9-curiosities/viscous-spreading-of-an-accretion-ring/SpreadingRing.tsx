import { useMemo } from 'react'
import {
  Curve,
  Dashboard,
  DashboardCell,
  DashboardRow,
  Figure,
  Handle,
  Player,
  Plot,
  Readout,
  slider,
  useAxis,
  useFigureState,
  usePlayhead,
} from 'aifn-render'
import { angularMomentumDensity, budget, massDensity, surfaceDensity } from './ring'

// The Player scrubs τ = 12νt/R₀² on a log scale: 20 frames per decade from 10⁻³ to 10³.
const FRAMES = 120
const LOG_TAU_MIN = -3
const LOG_TAU_MAX = 3
const tauAt = (k: number) => 10 ** (LOG_TAU_MIN + ((LOG_TAU_MAX - LOG_TAU_MIN) * k) / FRAMES)
const TAUS = Array.from({ length: FRAMES + 1 }, (_, k) => tauAt(k))
const frameOf = (tau: number) =>
  Math.min(FRAMES, Math.max(0, Math.round(((Math.log10(tau) - LOG_TAU_MIN) / (LOG_TAU_MAX - LOG_TAU_MIN)) * FRAMES)))

// Radius grids: linear for Σ near the ring, logarithmic for the mass and angular momentum per unit ln r. The log grid
// is refined on [0.8, 1.2], where the ring starts with a width of √(τ/2) ≈ 0.02 R₀.
const LINEAR = Array.from({ length: 401 }, (_, i) => (2.5 * i) / 400)
const LOGARITHMIC = [
  ...Array.from({ length: 401 }, (_, i) => 10 ** (-2 + (5 * i) / 400)).filter((x) => x < 0.8 || x > 1.2),
  ...Array.from({ length: 201 }, (_, i) => 0.8 + (0.4 * i) / 200),
].sort((a, b) => a - b)

const pct = (v: number) => `${(100 * v).toFixed(1)}%`
const fix = (v: number, d = 4) => v.toFixed(d)

export function SpreadingRing() {
  const state = useFigureState({
    split: slider(0.2, 3, 1, { label: 'split radius r_c / R₀', step: 0.01, onChart: true }),
  })
  const split = state.split
  const [frame, setFrame] = usePlayhead(FRAMES + 1)
  const tau = TAUS[frame]

  // The whole history of the budget, by quadrature at every frame (about 50 ms).
  const series = useMemo(() => TAUS.map((t) => budget(t, split)), [split])
  const now = series[frame]

  const profile = useMemo(() => LINEAR.map((x) => surfaceDensity(x, tau)), [tau])
  const perLog = useMemo(
    () => ({
      mass: LOGARITHMIC.map((x) => x * massDensity(x, tau)),
      angular: LOGARITHMIC.map((x) => x * angularMomentumDensity(x, tau)),
    }),
    [tau],
  )
  const history = useMemo(
    () => ({
      inside: series.map((b) => b.massInside),
      outside: series.map((b) => b.massOutside),
      accreted: series.map((b) => b.accreted),
      angularOutside: series.map((b) => b.angularOutside),
    }),
    [series],
  )

  const disc = now.massInside + now.massOutside
  const angular = now.angularInside + now.angularOutside

  const rAxis = useAxis({ label: 'radius r / R₀', range: [0, 2.5], nice: false })
  const sAxis = useAxis({ label: 'surface density Σ / (m / πR₀²)', range: [0, undefined] })
  const logR = useAxis({ label: 'radius r / R₀', log: true, range: [0.01, 1000] })
  const perAxis = useAxis({ label: 'per unit ln r (units m, m√(GMR₀))', range: [0, undefined] })
  const tAxis = useAxis({
    label: 'viscous time τ = 12νt / R₀²',
    log: true,
    range: [10 ** LOG_TAU_MIN, 10 ** LOG_TAU_MAX],
  })
  const fAxis = useAxis({ label: 'fraction of the total', range: [0, 1] })

  return (
    <Figure
      title="A spreading ring"
      purpose="A ring of gas in a Keplerian disc spreads under constant viscosity: the mass flows onto the centre while the angular momentum moves to ever larger radii in a shrinking share of the mass."
      state={state}
      defaultSize="XL"
      controls={
        <Player
          value={frame}
          onChange={setFrame}
          count={FRAMES + 1}
          label="viscous time τ"
          format={(k) => `τ = ${tauAt(k).toPrecision(3)}`}
        />
      }
      readouts={{
        time: (
          <>
            <Readout label="τ = 12νt / R₀²" value={tau.toPrecision(3)} />
            <Readout label="t in viscous times R₀²/ν" value={(tau / 12).toPrecision(3)} />
          </>
        ),
        'mass (units m)': (
          <>
            <Readout label="in the disc, ∫2πRΣ dR" value={fix(disc)} />
            <Readout label="in the disc, closed form P(¼, 1/τ)" value={fix(now.discClosedForm)} />
            <Readout label="onto the centre, ∫Ṁ dt" value={fix(now.accreted)} />
            <Readout label="total" value={fix(disc + now.accreted, 6)} />
          </>
        ),
        [`beyond r_c = ${split.toFixed(2)} R₀`]: (
          <>
            <Readout label="share of the mass" value={pct(now.massOutside)} />
            <Readout label="share of the angular momentum" value={pct(now.angularOutside)} />
            <Readout label="total angular momentum (units m√(GMR₀))" value={fix(angular, 6)} />
          </>
        ),
      }}
      caption="Top left: the surface density Σ(r) at the current τ, from the closed form. Top right: the mass and the angular momentum per unit ln r on a log radius axis; the area under each curve against ln r is the disc's total. Bottom: the history of the budget split at r_c, every value an integral of the closed form by adaptive Gauss–Kronrod quadrature. Drag the vertical r_c line in either radius chart to move the split, and drag the τ line on the bottom chart to scrub time. The total mass stays 1 and the total angular momentum stays 1 at every τ."
    >
      <Dashboard>
        <DashboardRow minHeight={240}>
          <DashboardCell>
            <Plot x={rAxis} y={sAxis} legend={false}>
              <Curve name="Σ(r)" x={LINEAR} y={profile} slot={0} live />
              <Curve name="initial radius" x={[1, 1]} y={[0, Math.max(...profile)]} muted dashed />
              <Handle {...state.handle('split', { label: 'r_c' })} />
            </Plot>
          </DashboardCell>
          <DashboardCell>
            <Plot x={logR} y={perAxis}>
              <Curve name="mass, dM/d ln r" x={LOGARITHMIC} y={perLog.mass} slot={0} live />
              <Curve name="angular momentum, dJ/d ln r" x={LOGARITHMIC} y={perLog.angular} slot={1} live />
              <Handle {...state.handle('split', { label: 'r_c' })} />
            </Plot>
          </DashboardCell>
        </DashboardRow>
        <DashboardRow minHeight={220}>
          <DashboardCell>
            <Plot x={tAxis} y={fAxis}>
              <Curve name="mass inside r_c" x={TAUS} y={history.inside} slot={0} />
              <Curve name="mass outside r_c" x={TAUS} y={history.outside} slot={0} dashed />
              <Curve name="mass onto the centre" x={TAUS} y={history.accreted} slot={2} />
              <Curve name="angular momentum outside r_c" x={TAUS} y={history.angularOutside} slot={1} />
              <Handle kind="x" at={tau} onDrag={(t) => setFrame(frameOf(t))} label="τ" />
            </Plot>
          </DashboardCell>
        </DashboardRow>
      </Dashboard>
    </Figure>
  )
}
