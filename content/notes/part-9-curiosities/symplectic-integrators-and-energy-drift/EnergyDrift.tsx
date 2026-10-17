import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Player,
  Plot,
  Plots,
  Points,
  Readout,
  setting,
  slider,
  useAxis,
  useComputed,
  useFigureState,
  usePlayhead,
} from 'aifn-render'
import type { SymplecticMethod } from 'aifn-compute/dynamics/ode'
import { envelope, exactEllipse, keplerRuns, MAX_STEPS, type KeplerRun } from './physics'

const fmt = (v: number) => formatNumber(v)
const TAU = 2 * Math.PI
/** Errors below this are drawn at it on the log axis. */
const FLOOR = 1e-16

const METHOD_LABEL: Record<SymplecticMethod, string> = {
  leapfrog: 'leapfrog',
  'velocity-verlet': 'velocity Verlet',
  'symplectic-euler': 'symplectic Euler',
}

type Drawn = {
  name: string
  slot: number
  run: KeplerRun
  energy: { t: number[]; y: number[] }
}

export function EnergyDrift() {
  const state = useFigureState({
    ecc: slider(0, 0.9, 0.6, { label: 'eccentricity e', step: 0.01 }),
    h: float(0.05, { gt: 0, le: 0.5, label: 'step size h', suggestions: [0.01, 0.02, 0.05, 0.1, 0.2] }),
    periods: int(300, { ge: 1, le: 1000, label: 'periods', suggestions: [10, 100, 200, 500, 1000] }),
    method: choice<SymplecticMethod>(
      [
        { value: 'leapfrog', label: 'leapfrog' },
        { value: 'velocity-verlet', label: 'velocity Verlet' },
      ],
      'leapfrog',
      { label: 'symplectic method' },
    ),
    logScale: setting(false, 'log |H − H₀|'),
  })
  const { ecc, h, periods, method, logScale } = state

  const computed = useComputed(() => {
    const { steps, per, rk4, sympl } = keplerRuns(ecc, h, periods, method)
    const make = (name: string, slot: number, run: KeplerRun): Drawn => ({
      name,
      slot,
      run,
      energy: envelope(run.dE, h),
    })
    return {
      steps,
      per,
      runs: [make('RK4', 0, rk4), make(METHOD_LABEL[method], 1, sympl)],
    }
  }, [ecc, h, periods, method])
  const { steps, per, runs } = computed.value
  const done = (steps * h) / TAU

  // Frame k shows period k: the orbit over it, the pericentres so far and the energy error up to time kT.
  const count = Math.max(2, Math.floor(done) + 1)
  const [k, setK] = usePlayhead(count)
  const lastStep = Math.min(steps, Math.round((k * TAU) / h))
  const firstStep = Math.max(0, lastStep - per)

  const ellipse = useMemo(() => exactEllipse(ecc), [ecc])
  const drawn = runs.map((r) => {
    const upto = upToTime(r.energy.t, k * TAU)
    const e = r.energy.y.slice(0, upto + 1)
    return {
      ...r,
      x: r.run.x.slice(firstStep, lastStep + 1),
      y: r.run.y.slice(firstStep, lastStep + 1),
      px: r.run.peri.x.slice(0, Math.floor(lastStep / per)),
      py: r.run.peri.y.slice(0, Math.floor(lastStep / per)),
      et: r.energy.t.slice(0, upto + 1),
      ey: logScale ? e.map((v) => Math.max(FLOOR, Math.abs(v))) : e,
      dE: r.run.dE[lastStep] ?? NaN,
      dL: r.run.dL[lastStep] ?? NaN,
      omega: r.run.omega[lastStep] ?? NaN,
    }
  })
  const allE = runs.flatMap((r) => r.energy.y)
  const eRange: [number, number] = logScale
    ? [1e-12, Math.max(1e-10, ...allE.map(Math.abs)) * 2]
    : [Math.min(0, ...allE) * 1.05, Math.max(0, ...allE) * 1.05 || 1e-6]

  const ox = useAxis({ label: 'x', range: [-2, 2] })
  const oy = useAxis({ label: 'y', range: [-2, 2], equal: ox })
  const tx = useAxis({ label: 'time t (periods)', range: [0, Math.max(1, done)], format: (v) => fmt(v) })
  const ty = useAxis({ label: logScale ? '|H − H₀|' : 'H − H₀', range: eRange, log: logScale, key: logScale })

  return (
    <Figure
      title="RK4 against a symplectic method on a Kepler orbit"
      state={state}
      defaultSize="XL"
      controls={
        <Player
          value={k}
          onChange={setK}
          count={count}
          label="period"
          format={(i) => `period ${i} of ${Math.floor(done)}`}
        />
      }
      readouts={{
        [`after ${k} periods`]: drawn.map((d) => (
          <Readout
            key={d.name}
            label={`${d.name}: H − H₀, h − h₀, ω`}
            value={`${fmt(d.dE)}, ${fmt(d.dL)}, ${fmt(d.omega)}°`}
          />
        )),
        cost: (
          <>
            <Readout
              label="steps per integrator"
              value={steps < Math.round((periods * TAU) / h) ? `${steps} (capped at ${MAX_STEPS})` : steps}
            />
            <Readout label="force evaluations per step (RK4, symplectic)" value="4, 1" />
          </>
        ),
      }}
      caption="The Kepler orbit with semi-major axis 1 and eccentricity e (units with G M = 1, period 2π), started at pericentre and integrated with the same step h by RK4 and by a symplectic method. Left: the orbit during the current period, the exact ellipse (dashed) and the pericentre at the end of every period so far (dots), from the eccentricity vector; ω is its angle in degrees. Right: the energy error H − H₀ against time, thinned to its envelope. Play to advance one period per frame. The symplectic method's error stays in a fixed band; RK4's error grows in proportion to time, so its orbit shrinks. The symplectic orbit keeps its size and shape but precesses steadily. The symplectic method also keeps the angular momentum h exactly, to rounding; RK4 loses it. Type a step size or a number of periods, or pick a suggestion; runs are cut at 200 000 steps."
    >
      <Plots cols={2} widths={[1, 1.4]}>
        <Plot x={ox} y={oy}>
          <Curve name="exact orbit" x={ellipse.x} y={ellipse.y} muted dashed />
          {drawn.map((d) => (
            <Curve key={d.name} name={d.name} x={d.x} y={d.y} slot={d.slot} stale={computed.stale} />
          ))}
          {drawn.map((d) =>
            d.px.length ? (
              <Points key={d.name} name={`${d.name} pericentres`} x={d.px} y={d.py} slot={d.slot} thin />
            ) : null,
          )}
          <Points name="mass" x={[0]} y={[0]} emphasis size={9} />
        </Plot>
        <Plot x={tx} y={ty}>
          {drawn.map((d) => (
            <Curve
              key={d.name}
              name={d.name}
              x={d.et.map((t) => t / TAU)}
              y={d.ey}
              slot={d.slot}
              width={1}
              stale={computed.stale}
            />
          ))}
        </Plot>
      </Plots>
    </Figure>
  )
}

/** The index of the last entry of an ascending `ts` at or before `t` (0 if none). */
function upToTime(ts: readonly number[], t: number): number {
  let lo = 0
  let hi = ts.length - 1
  if (hi < 0 || ts[0] > t) return 0
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1
    if (ts[mid] <= t + 1e-9) lo = mid
    else hi = mid - 1
  }
  return lo
}
