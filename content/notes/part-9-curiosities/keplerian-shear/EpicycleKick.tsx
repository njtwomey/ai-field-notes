import { useMemo, useState } from 'react'
import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Player,
  Plot,
  Plots,
  Points,
  Readout,
  useAxis,
  useComputed,
  useFigureState,
  Vectors,
} from 'aifn-render'
import { epicycle, epicyclicFrequency, SHEAR_EXPONENT, type Potential } from './shear'

/** The kick arrow is drawn this many times its length in velocity units. */
const ARROW = 2
const KICK = 0.3
/** Kept steps per player position. */
const STRIDE = 3

const clamp = (v: number) => Math.min(KICK, Math.max(-KICK, v))
const circle = Array.from({ length: 181 }, (_, i) => (2 * Math.PI * i) / 180)

/**
 * A particle kicked off a circular orbit at r = 1, in the inertial frame and in the frame rotating with the circular
 * orbit, where it traces an epicycle about a guiding centre. The integration is RK4; Hill's linear solution is drawn
 * beside it.
 */
export function EpicycleKick() {
  const state = useFigureState({
    potential: choice(
      [
        { value: 'kepler', label: 'point mass, Φ = −1/r (Kepler)' },
        { value: 'flat', label: 'flat rotation curve, Φ = ln r' },
      ],
      'kepler',
      { label: 'potential' },
    ),
    u0: float(0.1, { min: -KICK, max: KICK, step: 0.01, label: 'radial kick u₀', onChart: true }),
    v0: float(0, { min: -KICK, max: KICK, step: 0.01, label: 'azimuthal kick v₀', onChart: true }),
    orbits: int(3, { min: 1, max: 20, label: 'orbits', suggestions: [1, 3, 10] }),
    stepsPerOrbit: int(400, { min: 20, max: 4000, label: 'RK4 steps per orbit', suggestions: [50, 200, 400, 1000] }),
  })
  const potential = state.potential as Potential
  const q = SHEAR_EXPONENT[potential]
  const kappa = epicyclicFrequency(1, q)

  const computed = useComputed(
    () => epicycle(potential, state.u0, state.v0, state.orbits, state.stepsPerOrbit),
    [potential, state.u0, state.v0, state.orbits, state.stepsPerOrbit],
  )
  const r = computed.value
  const count = Math.floor((r.t.length - 1) / STRIDE) + 1
  const [position, setPosition] = useState({ r, k: 0 })
  const k = Math.min(position.r === r ? position.k : 0, count - 1)
  const i = k * STRIDE
  const upTo = <T,>(a: T[]) => a.slice(0, i + 1)

  const peri = r.pericentres
  const swept = peri.length >= 2 ? ((peri[1].phi - peri[0].phi) * 180) / Math.PI : NaN
  const radialPeriod = peri.length >= 2 ? peri[1].time - peri[0].time : NaN

  const range = useMemo(() => {
    let lo = 0
    let hi = 0
    for (const v of [...r.eta, ...r.hillEta]) {
      lo = Math.min(lo, v)
      hi = Math.max(hi, v)
    }
    let xlo = 0
    let xhi = 0
    for (const v of [...r.xi, ...r.hillXi]) {
      xlo = Math.min(xlo, v)
      xhi = Math.max(xhi, v)
    }
    const pad = 0.1 * Math.max(hi - lo, xhi - xlo, 0.2)
    return { eta: [lo - pad, hi + pad] as [number, number], xi: [xlo - pad, xhi + pad] as [number, number] }
  }, [r])

  const xa = useAxis({ label: 'x', range: [-1.7, 1.7] })
  const ya = useAxis({ label: 'y', range: [-1.7, 1.7], equal: xa })
  const ea = useAxis({ label: 'η, azimuthal offset (behind ← → ahead)', range: range.eta })
  const xia = useAxis({ label: 'ξ = r − 1', range: range.xi, equal: ea })

  return (
    <Figure
      title="An epicycle in the rotating frame"
      defaultSize="L"
      state={state}
      controls={
        <Player
          value={k}
          onChange={(v) => setPosition({ r, k: v })}
          count={count}
          label="time"
          format={(p) => `t = ${r.t[Math.min(p * STRIDE, r.t.length - 1)].toFixed(2)}`}
        />
      }
      readouts={{
        frequencies: (
          <>
            <Readout label="κ / Ω, theory" value={formatNumber(kappa)} />
            <Readout
              label="κ / Ω, measured as 360° / azimuth swept"
              value={Number.isFinite(swept) ? formatNumber(360 / swept) : '—'}
            />
            <Readout
              label="radial period, between pericentres"
              value={Number.isFinite(radialPeriod) ? formatNumber(radialPeriod) : '—'}
            />
            <Readout
              label="azimuth swept between pericentres"
              value={Number.isFinite(swept) ? `${swept.toFixed(2)}°` : '—'}
            />
          </>
        ),
        [`at t = ${r.t[i].toFixed(2)}`]: (
          <>
            <Readout label="ξ (radial offset)" value={formatNumber(r.xi[i])} />
            <Readout label="η (azimuthal offset)" value={formatNumber(r.eta[i])} />
            <Readout label="energy error E − E₀" value={formatNumber(r.energy[i] - r.energy[0])} />
          </>
        ),
      }}
      caption={
        <>
          Units G = M = 1. The particle starts at (1, 0) with the circular velocity plus a kick of u₀ outwards and v₀
          along the orbit. Drag the arrow tip in the left panel to set the kick (drawn {ARROW} times its length). Left:
          the orbit in the inertial frame, with the circle r = 1. Right: the same motion in the frame rotating with the
          circular orbit, with ξ = r − 1 and η = φ − t (arc length along r = 1), on equal axes; the dashed curve is
          Hill&apos;s linear solution. For a point mass κ = Ω, so the radial oscillation takes exactly one orbit, the
          azimuth swept between pericentres is 360° and the orbit closes into an ellipse. In the rotating frame the
          particle runs backwards round an ellipse twice as long as it is wide. An azimuthal kick moves the guiding
          centre outwards, where the shear carries it backwards. For the flat rotation curve κ = √2 Ω and the orbit does
          not close.
        </>
      }
    >
      <Plots cols={2} widths={[1, 1.3]}>
        <Plot x={xa} y={ya}>
          <Curve name="r = 1" x={circle.map(Math.cos)} y={circle.map(Math.sin)} muted />
          <Curve name="orbit (whole run)" x={r.x} y={r.y} slot={0} width={1} muted />
          <Curve name="orbit" x={upTo(r.x)} y={upTo(r.y)} slot={0} stale={computed.stale} />
          <Points name="particle" x={[r.x[i]]} y={[r.y[i]]} emphasis live />
          <Vectors vectors={[{ from: [1, 0], to: [1 + ARROW * state.u0, ARROW * state.v0], label: 'kick', slot: 1 }]} />
          <Handle
            kind="point"
            at={[1 + ARROW * state.u0, ARROW * state.v0]}
            label="kick"
            onDrag={([x, y]) => {
              state.set('u0', clamp((x - 1) / ARROW))
              state.set('v0', clamp(y / ARROW))
            }}
          />
        </Plot>
        <Plot x={ea} y={xia}>
          <Curve name="Hill's linear solution" x={r.hillEta} y={r.hillXi} emphasis dashed width={1} />
          <Curve name="epicycle (whole run)" x={r.eta} y={r.xi} slot={0} width={1} muted />
          <Curve name="epicycle" x={upTo(r.eta)} y={upTo(r.xi)} slot={0} stale={computed.stale} />
          <Points name="particle" x={[r.eta[i]]} y={[r.xi[i]]} emphasis live />
        </Plot>
      </Plots>
    </Figure>
  )
}
