import { ackermann, lqr } from 'aifn-compute/dynamics/control'
import { poles, respond, stateSpace } from 'aifn-compute/systems'
import { imagPart, realPart, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { useMemo } from 'react'
import { Player, usePlayhead } from 'aifn-render/controls'
import { Dashboard, DashboardCell, DashboardRow, Figure } from 'aifn-render/layout'
import { row, slider, useFigureState } from 'aifn-render/state'
import { Annotation, Curve, Handle, Plot, Plots, Points, Readout, formatNumber, useAxis } from 'aifn-render/viz'

const flat = (t: Tensor) => toFlat(t)
const column = (m: Tensor, j: number) => toRows(m).map((r) => r[j])
const fmt = (v: number) => formatNumber(v)

/** Several traces as one NaN-separated line: the whole run drawn faintly behind the part played so far. */
function ghost(t: readonly number[], ys: readonly (readonly number[])[]): { x: number[]; y: number[] } {
  const x: number[] = []
  const y: number[] = []
  for (const v of ys) {
    x.push(...t, NaN)
    y.push(...v, NaN)
  }
  return { x, y }
}

const timeLabel = (t: readonly number[]) => (i: number) => `t = ${(t[i] ?? 0).toFixed(2)} s`

// ── The linearised cart-pole ─────────────────────────────────────────────────────────────────────────────────────────
// States (x, ẋ, θ, θ̇) about the upright position; input: the force on the cart (per unit cart mass).

const CART_POLE = stateSpace({
  A: [
    [0, 1, 0, 0],
    [0, 0, -0.98, 0],
    [0, 0, 0, 1],
    [0, 0, 21.56, 0],
  ],
  B: [0, 1, 0, -2],
  C: [
    [1, 0, 0, 0],
    [0, 0, 1, 0],
  ],
})

const cartPoleQ = (angleWeight: number) => [
  [1, 0, 0, 0],
  [0, 0.1, 0, 0],
  [0, 0, angleWeight, 0],
  [0, 0, 0, 0.1],
]

/** Half-width and height of the drawn cart, and the drawn pole's length (m). */
const CART_W = 0.2
const CART_H = 0.12
const POLE = 0.6

export function LqrCartPoleSpecimen() {
  const state = useFigureState({
    cost: row('1 · cost', {
      logQ: slider(-1, 3, 1, { label: 'log₁₀ angle weight q_θ', step: 0.1 }),
      logR: slider(-2, 2, -0.3, { label: 'log₁₀ force weight R', step: 0.1 }),
    }),
    start: row('2 · start', { theta0: slider(-0.4, 0.4, 0.2, { label: 'initial angle θ₀ (rad)', step: 0.01 }) }),
  })
  const { logQ, logR } = state.cost
  const { theta0 } = state.start
  const result = useMemo(() => lqr(CART_POLE.repr, cartPoleQ(10 ** logQ), [[10 ** logR]]), [logQ, logR])
  const run = useMemo(() => {
    const K = flat(result.K)
    const r = respond(CART_POLE, (_t, x) => -flat(x).reduce((s, v, i) => s + K[i] * v, 0), {
      dt: 0.02,
      tEnd: 6,
      x0: [0, 0, theta0, 0],
    })
    const t = flat(r.t)
    const cart = column(r.y, 0)
    const angle = column(r.y, 1)
    const force = flat(r.u)
    // The drawing's range: every cart position with the pole either side of it, held for the whole run.
    const lo = Math.min(...cart) - POLE - CART_W
    const hi = Math.max(...cart) + POLE + CART_W
    return { t, cart, angle, force, xRange: [lo, hi] as [number, number] }
  }, [result, theta0])
  const [i, setI] = usePlayhead(run.t.length)
  const { t, cart, angle, force } = run

  const states = useMemo(() => ghost(t, [cart, angle]), [t, cart, angle])
  const effort = useMemo(() => ghost(t, [force]), [t, force])
  const c = cart[i]
  const th = angle[i]
  const tip: [number, number] = [c + POLE * Math.sin(th), CART_H + POLE * Math.cos(th)]
  const K = flat(result.K)
  const slowest = Math.max(...flat(realPart(result.closedLoop)))
  const dx = useAxis({ label: 'x (m)', range: run.xRange, zoom: false })
  const dy = useAxis({ label: '', range: [-0.1, CART_H + POLE + 0.1], equal: dx, zoom: false })
  const time = useAxis({ label: 't (s)' })
  const sy = useAxis({ label: 'x (m), θ (rad)', hold: 'initial' })
  const uy = useAxis({ label: 'u', hold: 'initial' })
  return (
    <Figure
      title="LQR balances a cart-pole"
      purpose="The optimal gain K = R⁻¹BᵀP trades the angle error against control effort: a heavier angle weight or a cheaper force catches the pole faster with a larger push."
      defaultSize="L"
      state={state}
      controls={<Player value={i} onChange={setI} count={t.length} format={timeLabel(t)} label="3 · time t" />}
      readouts={{
        'at t': (
          <>
            <Readout label="t" value={`${fmt(t[i])} s`} />
            <Readout label="θ(t)" value={`${fmt(angle[i])} rad`} />
            <Readout label="u(t)" value={fmt(force[i])} />
          </>
        ),
        design: (
          <>
            <Readout label="K" value={K.map(fmt).join(', ')} />
            <Readout label="peak |u|" value={fmt(Math.max(...force.map(Math.abs)))} />
            <Readout label="slowest closed-loop Re λ" value={fmt(slowest)} />
            <Readout
              label="Riccati"
              value={result.converged ? `converged in ${result.steps} Kleinman steps` : String(result.failure)}
            />
          </>
        ),
      }}
      caption="The linearised cart-pole (upright at θ = 0) under u = −Kx from rest at angle θ₀. Play to watch the loop catch the pole: left, the cart and pole at time t; right, the states and the force up to t over the whole run in grey. Drag the horizontal line on the states panel to change θ₀. The cart runs in the direction the pole leans, to bring its base back under the pole, then returns to the origin. Axes are held while the weights change."
    >
      <Dashboard>
        <DashboardRow minHeight={320}>
          <DashboardCell>
            <Plot x={dx} y={dy} legend={false} toolbar={false}>
              <Curve name="track" x={run.xRange} y={[0, 0]} muted />
              <Curve
                name="cart"
                x={[c - CART_W, c + CART_W, c + CART_W, c - CART_W, c - CART_W]}
                y={[0, 0, CART_H, CART_H, 0]}
                slot={0}
                live
              />
              <Curve name="pole" x={[c, tip[0]]} y={[CART_H, tip[1]]} slot={1} live />
              <Points name="bob" x={[tip[0]]} y={[tip[1]]} slot={1} live />
            </Plot>
          </DashboardCell>
          <DashboardCell ratio={1.4}>
            <Plots rows={2} heights={[3, 2]} hoverGroup>
              <Plot x={time} y={sy}>
                <Curve name="whole run" x={states.x} y={states.y} muted />
                <Curve name="cart position x" x={t.slice(0, i + 1)} y={cart.slice(0, i + 1)} slot={0} />
                <Curve name="pole angle θ" x={t.slice(0, i + 1)} y={angle.slice(0, i + 1)} slot={1} />
                <Points name="cart position x" x={[t[i]]} y={[cart[i]]} slot={0} live />
                <Points name="pole angle θ" x={[t[i]]} y={[angle[i]]} slot={1} live />
                <Handle {...state.handle('start.theta0', { axis: 'y', label: 'θ₀' })} />
              </Plot>
              <Plot x={time} y={uy}>
                <Curve name="whole run" x={effort.x} y={effort.y} muted />
                <Curve name="force u = −Kx" x={t.slice(0, i + 1)} y={force.slice(0, i + 1)} slot={2} />
                <Points name="force u = −Kx" x={[t[i]]} y={[force[i]]} slot={2} live />
              </Plot>
            </Plots>
          </DashboardCell>
        </DashboardRow>
      </Dashboard>
    </Figure>
  )
}

// From expensive control (R = 10³) to cheap (R = 10⁻³), the order the player sweeps.
const R_SWEEP = Array.from({ length: 61 }, (_, i) => 3 - (6 * i) / 60)

const formatPole = (re: number, im: number) =>
  Math.abs(im) < 1e-9 ? fmt(re) : `${fmt(re)} ${im < 0 ? '−' : '+'} ${fmt(Math.abs(im))}i`

export function LqrPolesSpecimen() {
  const state = useFigureState({
    cost: row('1 · cost', { qTheta: slider(1, 100, 10, { label: 'angle weight q_θ', step: 1 }) }),
  })
  const { qTheta } = state.cost
  // Poles are a complex128 vector: its real and imaginary parts are float64 views.
  const open = useMemo(() => {
    const p = poles(CART_POLE)
    return { real: flat(realPart(p)), imag: flat(imagPart(p)) }
  }, [])
  // Every R of the sweep solved once; the player walks through them.
  const sweep = useMemo(
    () =>
      R_SWEEP.map((lr) => {
        const r = lqr(CART_POLE.repr, cartPoleQ(qTheta), [[10 ** lr]])
        return { real: flat(realPart(r.closedLoop)), imag: flat(imagPart(r.closedLoop)) }
      }),
    [qTheta],
  )
  const [k, setK] = usePlayhead(R_SWEEP.length)
  const locus = useMemo(() => ({ x: sweep.flatMap((p) => p.real), y: sweep.flatMap((p) => p.imag) }), [sweep])
  const current = sweep[k]
  const re = useAxis({ label: 'Re λ', hold: 'initial', key: qTheta })
  const im = useAxis({ label: 'Im λ', equal: re, hold: 'initial', key: qTheta })
  return (
    <Figure
      title="LQR poles as control gets cheaper"
      purpose="As the force weight R falls the closed-loop poles move from the stable mirror images of the open-loop poles out to infinity along asymptotes: the symmetric root locus."
      state={state}
      controls={
        <Player
          value={k}
          onChange={setK}
          count={R_SWEEP.length}
          format={(j) => `log₁₀ R = ${R_SWEEP[j].toFixed(1)}`}
          label="2 · sweep log₁₀ R"
        />
      }
      readouts={{
        'at this R': (
          <>
            <Readout label="R" value={fmt(10 ** R_SWEEP[k])} />
            <Readout
              label="closed-loop poles"
              value={current.real.map((r, j) => formatPole(r, current.imag[j])).join(', ')}
            />
          </>
        ),
      }}
      caption="Play to sweep the force weight R from 10³ down to 10⁻³ and watch the closed-loop poles (ink) travel along the locus (grey, every R of the sweep). Crosses: the open-loop poles of the cart-pole, two at 0 and ±4.64. Expensive control (large R) only reflects the unstable pole into the left half-plane; cheap control drives every pole far left."
    >
      <Plot x={re} y={im}>
        <Points name="locus over R" x={locus.x} y={locus.y} muted thin />
        <Points name="open-loop poles" x={open.real} y={open.imag} slot={1} shape={4} />
        <Points name="closed-loop poles" x={current.real} y={current.imag} emphasis live />
      </Plot>
    </Figure>
  )
}

// ── Pole placement on the double integrator ──────────────────────────────────────────────────────────────────────────

const DOUBLE_INTEGRATOR = stateSpace({
  A: [
    [0, 1],
    [0, 0],
  ],
  B: [0, 1],
  C: [1, 0],
})

/** The closed-loop response of the double integrator from x = 1 at rest under u = −Kx: times, x, ẋ and u. */
function doubleIntegratorRun(K: readonly number[]) {
  const r = respond(DOUBLE_INTEGRATOR, (_t, x) => -flat(x).reduce((acc, v, i) => acc + K[i] * v, 0), {
    dt: 0.02,
    tEnd: 8,
    x0: [1, 0],
  })
  return { t: flat(r.t), x: column(r.x, 0), v: column(r.x, 1), u: flat(r.u) }
}

export function PolePlacementSpecimen() {
  const state = useFigureState({
    designs: row('1 · designs', { logR: slider(-2, 2, 0, { label: 'LQR log₁₀ R', step: 0.1 }) }),
    re: slider(-4, 0.8, -1, { onChart: true, label: 'Re of the placed pole' }),
    im: slider(0, 3, 1.5, { onChart: true, label: 'Im of the placed pole' }),
  })
  const { logR } = state.designs
  const pole: [number, number] = [state.re, state.im]
  const place = useMemo(
    () =>
      ackermann(DOUBLE_INTEGRATOR.repr, [
        { re: state.re, im: state.im },
        { re: state.re, im: -state.im },
      ]),
    [state.re, state.im],
  )
  const optimal = useMemo(
    () =>
      lqr(
        DOUBLE_INTEGRATOR.repr,
        [
          [1, 0],
          [0, 1],
        ],
        [[10 ** logR]],
      ),
    [logR],
  )
  const placed = useMemo(() => (place.K ? doubleIntegratorRun(flat(place.K)) : null), [place])
  const lq = useMemo(() => doubleIntegratorRun(flat(optimal.K)), [optimal])
  const t = lq.t
  const [i, setI] = usePlayhead(t.length)

  const phase = useMemo(() => {
    const runs = placed ? [placed, lq] : [lq]
    return { x: runs.flatMap((r) => [...r.x, NaN]), y: runs.flatMap((r) => [...r.v, NaN]) }
  }, [placed, lq])
  const response = useMemo(
    () =>
      ghost(
        lq.t,
        (placed ? [placed, lq] : [lq]).flatMap((r) => [r.x, r.u]),
      ),
    [placed, lq],
  )
  const named = [
    { run: placed, name: 'pole placement', slot: 0 },
    { run: lq, name: 'LQR', slot: 2 },
  ]
  const K = place.K ? flat(place.K) : [NaN, NaN]
  const Kl = flat(optimal.K)
  const lqPoles = { x: flat(realPart(optimal.closedLoop)), y: flat(imagPart(optimal.closedLoop)) }
  const cost = (r: { x: number[]; v: number[]; u: number[] } | null) =>
    r ? r.x.reduce((acc, x, k) => acc + 0.02 * (x * x + r.v[k] ** 2 + 10 ** logR * r.u[k] ** 2), 0) : NaN
  const re = useAxis({ label: 'Re λ', range: [-4, 1] })
  const im = useAxis({ label: 'Im λ', range: [-3.2, 3.2] })
  const px = useAxis({ label: 'x', hold: 'union' })
  const pv = useAxis({ label: 'ẋ', hold: 'union' })
  const time = useAxis({ label: 't (s)' })
  const ry = useAxis({ label: 'x, u', hold: 'union' })
  return (
    <Figure
      title="Pole placement against LQR"
      purpose="Choosing the closed-loop poles fixes the gain K by Ackermann's formula; LQR chooses them for you by minimising ∫ xᵀx + R u² dt. Real parts set the decay rate and imaginary parts the oscillation."
      defaultSize="XL"
      state={state}
      controls={<Player value={i} onChange={setI} count={t.length} format={timeLabel(t)} label="2 · time t" />}
      readouts={{
        gains: (
          <>
            <Readout label="pole placement K" value={K.map(fmt).join(', ')} />
            <Readout label="LQR K" value={Kl.map(fmt).join(', ')} />
            <Readout label="LQR poles" value={lqPoles.x.map((r, k) => formatPole(r, lqPoles.y[k])).join(', ')} />
          </>
        ),
        cost: <Readout label="∫ x² + ẋ² + Ru² (placement, LQR)" value={`${fmt(cost(placed))}, ${fmt(cost(lq))}`} />,
      }}
      caption="The double integrator ẍ = u starts at x = 1 at rest under u = −Kx. Drag the upper pole of the pole-placement design in the complex plane (its conjugate follows); LQR's poles (Q = I, weight R on the force) are the second pair. Play to watch both designs bring the state to the origin: the phase plane (x, ẋ) and the responses x(t) (solid) and u(t) (dashed), with the whole runs in grey. LQR never costs more than pole placement by its own measure; poles in the right half-plane give a growing response."
    >
      <Plots cols={3} widths={[1, 1, 1.3]}>
        <Plot x={re} y={im}>
          <Annotation x={0} />
          <Points name="pole placement" x={[pole[0], pole[0]]} y={[pole[1], -pole[1]]} slot={0} />
          <Points name="LQR" x={lqPoles.x} y={lqPoles.y} slot={2} />
          <Handle {...state.handle(['re', 'im'], { label: 'pole' })} />
        </Plot>
        <Plot x={px} y={pv} legend={false}>
          <Curve name="whole run" x={phase.x} y={phase.y} muted />
          {named.map(({ run, name, slot }) =>
            run ? (
              <Curve key={name} name={name} x={run.x.slice(0, i + 1)} y={run.v.slice(0, i + 1)} slot={slot} />
            ) : null,
          )}
          {named.map(({ run, name, slot }) =>
            run ? <Points key={name} name={name} x={[run.x[i]]} y={[run.v[i]]} slot={slot} live /> : null,
          )}
        </Plot>
        <Plot x={time} y={ry}>
          <Curve name="whole run" x={response.x} y={response.y} muted />
          {named.map(({ run, name, slot }) =>
            run ? (
              <Curve key={`${name}x`} name={name} x={t.slice(0, i + 1)} y={run.x.slice(0, i + 1)} slot={slot} />
            ) : null,
          )}
          {named.map(({ run, name, slot }) =>
            run ? (
              <Curve
                key={`${name}u`}
                name={`${name} u`}
                x={t.slice(0, i + 1)}
                y={run.u.slice(0, i + 1)}
                slot={slot}
                dashed
              />
            ) : null,
          )}
        </Plot>
      </Plots>
    </Figure>
  )
}
