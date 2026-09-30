import { ackermann, lqr } from 'aifn/dynamics/control'
import { poles, respond, stateSpace } from 'aifn/systems'
import { toFlat, toRows, type Tensor } from 'aifn/foundation/tensor'
import { useMemo, useState } from 'react'
import { Player, Slider, usePlayhead } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { Panel, Readout, Subplots, XYChart, formatNumber, type Handle, type XYSeries } from '@lab/viz'

const flat = (t: Tensor) => toFlat(t)
const column = (m: Tensor, j: number) => toRows(m).map((r) => r[j])
const fmt = (v: number) => formatNumber(v)

/** Several traces as one NaN-separated line: the whole run drawn faintly behind the part played so far. */
function ghost(name: string, t: readonly number[], ys: readonly (readonly number[])[]): XYSeries {
  const x: number[] = []
  const y: number[] = []
  for (const v of ys) {
    x.push(...t, NaN)
    y.push(...v, NaN)
  }
  return { name, type: 'line', muted: true, x, y }
}

/** A trace up to position i (inclusive), and its point at i. */
const upTo = (t: readonly number[], y: readonly number[], i: number) => ({ x: t.slice(0, i + 1), y: y.slice(0, i + 1) })
const at = (t: readonly number[], y: readonly number[], i: number) => ({ x: [t[i]], y: [y[i]] })
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
  const [logQ, setLogQ] = useState(1)
  const [logR, setLogR] = useState(-0.3)
  const [theta0, setTheta0] = useState(0.2)
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

  const states = useMemo((): XYSeries[] => [ghost('whole run', t, [cart, angle])], [t, cart, angle])
  const effort = useMemo((): XYSeries[] => [ghost('whole run', t, [force])], [t, force])
  const ground = useMemo(
    (): XYSeries[] => [{ name: 'track', type: 'line', x: run.xRange, y: [0, 0], muted: true }],
    [run.xRange],
  )
  const live = {
    states: [
      { name: 'cart position x', type: 'line', slot: 0, ...upTo(t, cart, i) },
      { name: 'pole angle θ', type: 'line', slot: 1, ...upTo(t, angle, i) },
      { name: 'cart position x', type: 'scatter', slot: 0, ...at(t, cart, i) },
      { name: 'pole angle θ', type: 'scatter', slot: 1, ...at(t, angle, i) },
    ] as XYSeries[],
    effort: [
      { name: 'force u = −Kx', type: 'line', slot: 2, ...upTo(t, force, i) },
      { name: 'force u = −Kx', type: 'scatter', slot: 2, ...at(t, force, i) },
    ] as XYSeries[],
    drawing: (() => {
      const c = cart[i]
      const th = angle[i]
      const base: [number, number] = [c, CART_H]
      const tip: [number, number] = [c + POLE * Math.sin(th), CART_H + POLE * Math.cos(th)]
      return [
        {
          name: 'cart',
          type: 'line',
          slot: 0,
          x: [c - CART_W, c + CART_W, c + CART_W, c - CART_W, c - CART_W],
          y: [0, 0, CART_H, CART_H, 0],
        },
        { name: 'pole', type: 'line', slot: 1, x: [base[0], tip[0]], y: [base[1], tip[1]] },
        { name: 'bob', type: 'scatter', slot: 1, x: [tip[0]], y: [tip[1]] },
      ] as XYSeries[]
    })(),
  }
  const handles: Handle[] = [
    { kind: 'y', at: theta0, label: 'θ₀', onDrag: (v) => setTheta0(Math.max(-0.4, Math.min(0.4, v))) },
  ]
  const K = flat(result.K)
  const slowest = Math.max(...flat(result.closedLoop.real))
  return (
    <Figure
      title="LQR balances a cart-pole"
      description="The optimal gain K = R⁻¹BᵀP trades the angle error against control effort: a heavier angle weight or a cheaper force catches the pole faster with a larger push."
      defaultSize="L"
      controls={
        <>
          <ControlRow label="1 · cost">
            <Slider label="log₁₀ angle weight q_θ" value={logQ} min={-1} max={3} step={0.1} onChange={setLogQ} />
            <Slider label="log₁₀ force weight R" value={logR} min={-2} max={2} step={0.1} onChange={setLogR} />
          </ControlRow>
          <ControlRow label="2 · start">
            <Slider
              label="initial angle θ₀ (rad)"
              value={theta0}
              min={-0.4}
              max={0.4}
              step={0.01}
              onChange={setTheta0}
            />
          </ControlRow>
          <ControlRow label="3 · time">
            <Player value={i} onChange={setI} count={t.length} duration={4} format={timeLabel(t)} label="t" />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout label="t" value={`${fmt(t[i])} s`} />
          <Readout label="θ(t)" value={`${fmt(angle[i])} rad`} />
          <Readout label="u(t)" value={fmt(force[i])} />
          <Readout label="K" value={K.map(fmt).join(', ')} />
          <Readout label="peak |u|" value={fmt(Math.max(...force.map(Math.abs)))} />
          <Readout label="slowest closed-loop Re λ" value={fmt(slowest)} />
          <Readout
            label="Riccati"
            value={result.converged ? `converged in ${result.steps} Kleinman steps` : String(result.failure)}
          />
        </>
      }
      caption="The linearised cart-pole (upright at θ = 0) under u = −Kx from rest at angle θ₀. Play to watch the loop catch the pole: left, the cart and pole at time t; right, the states and the force up to t over the whole run in grey. Drag the horizontal line on the states panel to change θ₀. The cart runs in the direction the pole leans, to bring its base back under the pole, then returns to the origin. Axes are held while the weights change."
    >
      <Subplots cols={2} widthRatios={[1, 1.4]}>
        <Panel>
          <XYChart
            series={ground}
            live={live.drawing}
            aspect="equal"
            xRange={run.xRange}
            yRange={[-0.1, CART_H + POLE + 0.1]}
            xLabel="x (m)"
            legend={false}
            zoom={false}
          />
        </Panel>
        <Panel>
          <Subplots rows={2} sharex heightRatios={[3, 2]} hoverGroup rescaleOnChange={false}>
            <Panel>
              <XYChart series={states} live={live.states} xLabel="t (s)" yLabel="x (m), θ (rad)" handles={handles} />
            </Panel>
            <Panel>
              <XYChart series={effort} live={live.effort} xLabel="t (s)" yLabel="u" />
            </Panel>
          </Subplots>
        </Panel>
      </Subplots>
    </Figure>
  )
}

// From expensive control (R = 10³) to cheap (R = 10⁻³), the order the player sweeps.
const R_SWEEP = Array.from({ length: 61 }, (_, i) => 3 - (6 * i) / 60)

const formatPole = (re: number, im: number) =>
  Math.abs(im) < 1e-9 ? fmt(re) : `${fmt(re)} ${im < 0 ? '−' : '+'} ${fmt(Math.abs(im))}i`

export function LqrPolesSpecimen() {
  const [qTheta, setQTheta] = useState(10)
  // Poles are a [k, 2] complex tensor: (re, im) per row.
  const open = useMemo(() => {
    const p = toRows(poles(CART_POLE))
    return { real: p.map((z) => z[0]), imag: p.map((z) => z[1]) }
  }, [])
  // Every R of the sweep solved once; the player walks through them.
  const sweep = useMemo(
    () =>
      R_SWEEP.map((lr) => {
        const r = lqr(CART_POLE.repr, cartPoleQ(qTheta), [[10 ** lr]])
        return { real: flat(r.closedLoop.real), imag: flat(r.closedLoop.imag) }
      }),
    [qTheta],
  )
  const [k, setK] = usePlayhead(R_SWEEP.length)
  const series = useMemo((): XYSeries[] => {
    const x = sweep.flatMap((p) => p.real)
    const y = sweep.flatMap((p) => p.imag)
    return [
      { name: 'locus over R', type: 'scatter', x, y, muted: true },
      { name: 'open-loop poles', type: 'scatter', x: open.real, y: open.imag, slot: 1 },
    ]
  }, [sweep, open])
  const current = sweep[k]
  const live: XYSeries[] = [
    { name: 'closed-loop poles', type: 'scatter', x: current.real, y: current.imag, emphasis: true },
  ]
  return (
    <Figure
      title="LQR poles as control gets cheaper"
      description="As the force weight R falls the closed-loop poles move from the stable mirror images of the open-loop poles out to infinity along asymptotes: the symmetric root locus."
      controls={
        <>
          <ControlRow label="1 · cost">
            <Slider label="angle weight q_θ" value={qTheta} min={1} max={100} step={1} onChange={setQTheta} />
          </ControlRow>
          <ControlRow label="2 · sweep R">
            <Player
              value={k}
              onChange={setK}
              count={R_SWEEP.length}
              duration={4}
              format={(j) => `log₁₀ R = ${R_SWEEP[j].toFixed(1)}`}
              label="log₁₀ R"
            />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout label="R" value={fmt(10 ** R_SWEEP[k])} />
          <Readout
            label="closed-loop poles"
            value={current.real.map((re, j) => formatPole(re, current.imag[j])).join(', ')}
          />
        </>
      }
      caption="Play to sweep the force weight R from 10³ down to 10⁻³ and watch the closed-loop poles (ink) travel along the locus (grey, every R of the sweep). Crosses: the open-loop poles of the cart-pole, two at 0 and ±4.64. Expensive control (large R) only reflects the unstable pole into the left half-plane; cheap control drives every pole far left."
    >
      <XYChart aspect="equal" xLabel="Re λ" yLabel="Im λ" series={series} live={live} axisKey={qTheta} />
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
  const [pole, setPole] = useState<[number, number]>([-1, 1.5])
  const [logR, setLogR] = useState(0)
  const place = useMemo(
    () => ackermann(DOUBLE_INTEGRATOR.repr, { real: [pole[0], pole[0]], imag: [pole[1], -pole[1]] }),
    [pole],
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

  const handles: Handle[] = [
    {
      kind: 'point',
      at: pole,
      label: 'pole',
      onDrag: ([re, im]) => setPole([Math.max(-4, Math.min(0.8, re)), Math.max(0, Math.min(3, Math.abs(im)))]),
    },
  ]
  const phase = useMemo((): XYSeries[] => {
    const runs = placed ? [placed, lq] : [lq]
    return [
      {
        name: 'whole run',
        type: 'line',
        muted: true,
        x: runs.flatMap((r) => [...r.x, NaN]),
        y: runs.flatMap((r) => [...r.v, NaN]),
      },
    ]
  }, [placed, lq])
  const response = useMemo(
    (): XYSeries[] => [
      ghost(
        'whole run',
        lq.t,
        (placed ? [placed, lq] : [lq]).flatMap((r) => [r.x, r.u]),
      ),
    ],
    [placed, lq],
  )
  const named = [
    { run: placed, name: 'pole placement', slot: 0 },
    { run: lq, name: 'LQR', slot: 2 },
  ]
  const livePhase: XYSeries[] = named.flatMap(({ run, name, slot }) => [
    { name, type: 'line', slot, ...(run ? upTo(run.x, run.v, i) : { x: [], y: [] }) },
    { name, type: 'scatter', slot, ...(run ? at(run.x, run.v, i) : { x: [], y: [] }) },
  ])
  const liveResponse: XYSeries[] = named.flatMap(({ run, name, slot }) => [
    { name, type: 'line', slot, ...(run ? upTo(t, run.x, i) : { x: [], y: [] }) },
    { name, type: 'line', slot, dashed: true, ...(run ? upTo(t, run.u, i) : { x: [], y: [] }) },
    { name, type: 'scatter', slot, ...(run ? at(t, run.x, i) : { x: [], y: [] }) },
  ])
  const K = place.K ? flat(place.K) : [NaN, NaN]
  const Kl = flat(optimal.K)
  const lqPoles = { x: flat(optimal.closedLoop.real), y: flat(optimal.closedLoop.imag) }
  const cost = (r: { x: number[]; v: number[]; u: number[] } | null) =>
    r ? r.x.reduce((acc, x, k) => acc + 0.02 * (x * x + r.v[k] ** 2 + 10 ** logR * r.u[k] ** 2), 0) : NaN
  return (
    <Figure
      title="Pole placement against LQR"
      description="Choosing the closed-loop poles fixes the gain K by Ackermann's formula; LQR chooses them for you by minimising ∫ xᵀx + R u² dt. Real parts set the decay rate and imaginary parts the oscillation."
      defaultSize="XL"
      controls={
        <>
          <ControlRow label="1 · designs">
            <Slider label="LQR log₁₀ R" value={logR} min={-2} max={2} step={0.1} onChange={setLogR} />
          </ControlRow>
          <ControlRow label="2 · time">
            <Player value={i} onChange={setI} count={t.length} duration={4} format={timeLabel(t)} label="t" />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout label="pole placement K" value={K.map(fmt).join(', ')} />
          <Readout label="LQR K" value={Kl.map(fmt).join(', ')} />
          <Readout label="LQR poles" value={lqPoles.x.map((re, k) => formatPole(re, lqPoles.y[k])).join(', ')} />
          <Readout label="cost ∫ x² + ẋ² + Ru² (placement, LQR)" value={`${fmt(cost(placed))}, ${fmt(cost(lq))}`} />
        </>
      }
      caption="The double integrator ẍ = u starts at x = 1 at rest under u = −Kx. Drag the upper pole of the pole-placement design in the complex plane (its conjugate follows); LQR's poles (Q = I, weight R on the force) are the second pair. Play to watch both designs bring the state to the origin: the phase plane (x, ẋ) and the responses x(t) (solid) and u(t) (dashed), with the whole runs in grey. LQR never costs more than pole placement by its own measure; poles in the right half-plane give a growing response."
    >
      <Subplots cols={3} widthRatios={[1, 1, 1.3]}>
        <Panel>
          <XYChart
            xLabel="Re λ"
            yLabel="Im λ"
            xRange={[-4, 1]}
            yRange={[-3.2, 3.2]}
            handles={handles}
            series={[
              { name: 'imaginary axis', type: 'line', x: [0, 0], y: [-3.2, 3.2], muted: true },
              {
                name: 'pole placement',
                type: 'scatter',
                x: [pole[0], pole[0]],
                y: [pole[1], -pole[1]],
                slot: 0,
              },
              { name: 'LQR', type: 'scatter', ...lqPoles, slot: 2 },
            ]}
          />
        </Panel>
        <Panel>
          <XYChart series={phase} live={livePhase} xLabel="x" yLabel="ẋ" rescaleOnChange={false} legend={false} />
        </Panel>
        <Panel>
          <XYChart series={response} live={liveResponse} xLabel="t (s)" yLabel="x, u" rescaleOnChange={false} />
        </Panel>
      </Subplots>
    </Figure>
  )
}
