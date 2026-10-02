/**
 * "LQG: estimate, then control": the sampled double integrator with process noise, of which only the position is
 * measured, with noise. Core `lqg` designs the LQR gain K and the Kalman gain L separately (the separation principle);
 * `lqgSimulation` runs the certainty-equivalent loop u = −Kx̂ and, on the same noise, the full-state LQR u = −Kx.
 */
import { lqg, lqgSimulation, type LqgSimulationState } from 'aifn/dynamics/control'
import { stream } from 'aifn/foundation/random'
import { toComplexFlat, toFlat } from 'aifn/foundation/tensor'
import { trace } from 'aifn/foundation/trace'
import { useMemo } from 'react'
import { Player, usePlayhead } from '@lab/controls'
import { Dashboard, DashboardCell, DashboardRow, Figure } from '@lab/layout'
import { int, row, slider, useFigureState } from '@lab/state'
import { Curve, Plot, Plots, Points, Readout, formatNumber, useAxis } from '@lab/viz'

const fmt = (v: number) => formatNumber(v)
const DT = 0.1
const A = [
  [1, DT],
  [0, 1],
]
const B = [[0.5 * DT * DT], [DT]]
const C = [[1, 0]]
const Q = [
  [1, 0],
  [0, 0.1],
]
const X0 = [3, 0]
const CIRCLE = (() => {
  const t = Array.from({ length: 121 }, (_, k) => (2 * Math.PI * k) / 120)
  return { x: t.map(Math.cos), y: t.map(Math.sin) }
})()

type Loop = {
  t: number[]
  p: number[]
  v: number[]
  pHat: number[]
  vHat: number[]
  y: number[]
  u: number[]
  cost: number[]
}

/** One closed loop recorded at every step: true state, estimate, measurement, input and average cost so far. */
function simulate(
  feedback: 'estimate' | 'state',
  weights: Parameters<typeof lqgSimulation>[1],
  design: Parameters<typeof lqgSimulation>[2],
  steps: number,
  seed: number,
): Loop {
  const alg = lqgSimulation({ A, B, C }, weights, design, { feedback })
  // The same root stream for both loops: the noise is drawn per step from it, so both see the same w and v.
  const tr = trace(alg, { x0: X0 }, steps, { keep: 'all', stream: stream(`lqg-page-${seed}`) })
  const out: Loop = { t: [], p: [], v: [], pHat: [], vHat: [], y: [], u: [], cost: [] }
  for (const s of tr.steps as readonly LqgSimulationState[]) {
    const x = toFlat(s.x)
    const xh = toFlat(s.xHat)
    out.t.push(s.t * DT)
    out.p.push(x[0])
    out.v.push(x[1])
    out.pHat.push(xh[0])
    out.vHat.push(xh[1])
    out.y.push(toFlat(s.y)[0])
    out.u.push(toFlat(s.u)[0])
    out.cost.push(s.t > 0 ? s.cost / s.t : 0)
  }
  return out
}

export function LqgSpecimen() {
  const state = useFigureState({
    noise: row('1 · noise', {
      logW: slider(-4, 0, -2, { label: 'log₁₀ process noise variance (velocity)', step: 0.1 }),
      logV: slider(-4, 1, -1, { label: 'log₁₀ measurement noise variance (position)', step: 0.1 }),
      seed: int(0, { ge: 0, le: 9999, label: 'noise seed' }),
    }),
    design: row('2 · design', {
      logR: slider(-3, 1, -1, { label: 'log₁₀ force weight R', step: 0.1 }),
      logMismatch: slider(-3, 3, 0, { label: 'log₁₀ (filter’s assumed V / true V)', step: 0.1 }),
    }),
    steps: int(300, { ge: 10, le: 5000, suggestions: [100, 300, 1000], label: 'steps' }),
  })
  const { logW, logV, seed } = state.noise
  const { logR, logMismatch } = state.design
  const steps = Number(state.steps)
  const result = useMemo(() => {
    // Process noise enters the velocity (a random force) plus a little on the position, so W is positive definite.
    const w = 10 ** logW
    const W = [
      [1e-3 * w, 0],
      [0, w],
    ]
    const V = [[10 ** logV]]
    const weights = { Q, R: [[10 ** logR]], W, V }
    // The filter may be tuned for the wrong measurement noise; the regulator never depends on the noise.
    const design = lqg({ A, B, C }, { ...weights, V: [[10 ** (logV + logMismatch)]] }, { discrete: true })
    const tuned = lqg({ A, B, C }, weights, { discrete: true })
    return {
      design,
      tunedL: toFlat(tuned.L),
      out: simulate('estimate', weights, design, steps, seed),
      full: simulate('state', weights, design, steps, seed),
    }
  }, [logW, logV, seed, logR, logMismatch, steps])
  const { design, out, full } = result
  const [i, setI] = usePlayhead(out.t.length)
  const slice = (xs: number[]) => xs.slice(0, i + 1)
  const regulator = toComplexFlat(design.regulatorPoles)
  const estimator = toComplexFlat(design.estimatorPoles)
  const rms = (a: number[], b: number[]) => Math.sqrt(a.reduce((s, v, k) => s + (v - b[k]) ** 2, 0) / a.length)
  const runKey = `${logW},${logV},${seed},${logR},${logMismatch},${steps}`

  const time = useAxis({ label: 't (s)', range: [0, steps * DT] })
  const pos = useAxis({ label: 'position p', hold: 'union', key: runKey })
  const vel = useAxis({ label: 'velocity v (not measured)', hold: 'union', key: runKey })
  const force = useAxis({ label: 'force u', hold: 'union', key: runKey })
  const avg = useAxis({ label: 'mean cost / step', log: true, hold: 'union', key: runKey })
  const re = useAxis({ label: 'Re z', range: [-1.15, 1.15] })
  const im = useAxis({ label: 'Im z', range: [-1.15, 1.15], equal: re })
  return (
    <Figure
      title="LQG: estimate, then control"
      purpose="With noisy measurements of part of the state, the optimal controller is the LQR gain applied to the Kalman filter's estimate. The two gains are designed separately: K depends only on the cost, L only on the noise, and the closed loop's poles are those of the regulator and of the estimator together (the separation principle)."
      defaultSize="XL"
      state={state}
      controls={
        <Player
          value={i}
          onChange={setI}
          count={out.t.length}
          format={(k) => `t = ${(k * DT).toFixed(1)} s`}
          label="3 · time t"
        />
      }
      readouts={{
        gains: (
          <>
            <Readout label="LQR gain K (cost only)" value={toFlat(design.K).map(fmt).join(', ')} />
            <Readout label="Kalman gain L (noise only)" value={toFlat(design.L).map(fmt).join(', ')} />
            <Readout label="L tuned for the true noise" value={result.tunedL.map(fmt).join(', ')} />
            <Readout
              label="|poles| regulator, estimator"
              value={`${regulator.map((z) => fmt(Math.hypot(z.re, z.im))).join(', ')}; ${estimator
                .map((z) => fmt(Math.hypot(z.re, z.im)))
                .join(', ')}`}
            />
          </>
        ),
        'whole run': (
          <>
            <Readout
              label="average cost per step (LQG, full-state LQR)"
              value={`${fmt(out.cost.at(-1)!)}, ${fmt(full.cost.at(-1)!)}`}
            />
            <Readout
              label="RMS estimation error p, v"
              value={`${fmt(rms(out.p, out.pHat))}, ${fmt(rms(out.v, out.vHat))}`}
            />
            <Readout label="RMS measurement error" value={fmt(rms(out.p, out.y))} />
          </>
        ),
      }}
      caption="The sampled double integrator (Δt = 0.1 s) starts at p = 3 and is regulated to the origin with Q = diag(1, 0.1) and force weight R. A random force (process noise) pushes the velocity; only the position is measured, with noise (dots). The LQG controller (first colour) feeds the Kalman filter's estimate (dashed) to the LQR gain: u = −Kx̂. On the same noise, the full-state LQR (third colour) uses the true state, which no real controller has; it is the lower bound on the cost. The velocity is never measured: the filter infers it from how the position moves. Raise the measurement noise and the filter trusts the model more (smaller L, slower estimator poles); raise the process noise and it trusts the measurements more. K does not change with either: the certainty-equivalent controller uses the same gain as if the estimate were the state. Tune the filter for the wrong measurement noise (the mismatch slider) and the cost rises although the regulator is unchanged. The pole plot shows the regulator poles (eigenvalues of A − BK) and the estimator poles (A − LC) inside the unit circle. Grey shows both whole runs; play to walk the loop. The bottom panel is the average cost per step so far."
    >
      <Dashboard>
        <DashboardRow minHeight={560}>
          <DashboardCell ratio={1.4}>
            <Plots rows={4} hoverGroup>
              <Plot x={time} y={pos}>
                <Curve name="whole runs" x={[...out.t, NaN, ...full.t]} y={[...out.p, NaN, ...full.p]} muted />
                <Points name="measurement y" x={slice(out.t)} y={slice(out.y)} muted size={3} />
                <Curve name="LQG: true p" x={slice(out.t)} y={slice(out.p)} slot={0} />
                <Curve name="estimate p̂" x={slice(out.t)} y={slice(out.pHat)} slot={0} dashed />
                <Curve name="full-state LQR" x={slice(full.t)} y={slice(full.p)} slot={2} />
              </Plot>
              <Plot x={time} y={vel} legend={false}>
                <Curve name="whole runs" x={[...out.t, NaN, ...full.t]} y={[...out.v, NaN, ...full.v]} muted />
                <Curve name="LQG: true v" x={slice(out.t)} y={slice(out.v)} slot={0} />
                <Curve name="estimate v̂" x={slice(out.t)} y={slice(out.vHat)} slot={0} dashed />
                <Curve name="full-state LQR" x={slice(full.t)} y={slice(full.v)} slot={2} />
              </Plot>
              <Plot x={time} y={force} legend={false}>
                <Curve name="whole runs" x={[...out.t, NaN, ...full.t]} y={[...out.u, NaN, ...full.u]} muted />
                <Curve name="LQG" x={slice(out.t)} y={slice(out.u)} slot={0} />
                <Curve name="full-state LQR" x={slice(full.t)} y={slice(full.u)} slot={2} />
              </Plot>
              <Plot x={time} y={avg} legend={false}>
                <Curve
                  name="whole runs"
                  x={[...out.t.slice(1), NaN, ...full.t.slice(1)]}
                  y={[...out.cost.slice(1), NaN, ...full.cost.slice(1)]}
                  muted
                />
                <Curve name="LQG" x={slice(out.t).slice(1)} y={slice(out.cost).slice(1)} slot={0} />
                <Curve name="full-state LQR" x={slice(full.t).slice(1)} y={slice(full.cost).slice(1)} slot={2} />
              </Plot>
            </Plots>
          </DashboardCell>
          <DashboardCell>
            <Plots rows={1}>
              <Plot x={re} y={im} title="closed-loop poles">
                <Curve name="unit circle" x={CIRCLE.x} y={CIRCLE.y} muted />
                <Points
                  name="regulator A − BK"
                  x={regulator.map((z) => z.re)}
                  y={regulator.map((z) => z.im)}
                  slot={0}
                />
                <Points
                  name="estimator A − LC"
                  x={estimator.map((z) => z.re)}
                  y={estimator.map((z) => z.im)}
                  slot={1}
                />
              </Plot>
            </Plots>
          </DashboardCell>
        </DashboardRow>
      </Dashboard>
    </Figure>
  )
}
