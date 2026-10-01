/**
 * UI-kit figures for figure state (DESIGN.md §4), probes (§6), the scheduler (§8a) and equations (§7). Each is a
 * review page for that building block, computed by aifn where there is maths to do.
 */
import { stream } from 'aifn/foundation/random'
import { toFlat, type Tensor } from 'aifn/foundation/tensor'
import { Laplace, Normal, StudentT, type Univariate } from 'aifn/probability/distributions'
import { useMemo, useState } from 'react'
import { Equation, EquationSteps, Figure, live, tex } from '@lab/layout'
import {
  choice,
  ProbeReadout,
  row,
  setting,
  slider,
  toEntries,
  toggle,
  useComputed,
  useFigureState,
  useProbe,
  variants,
} from '@lab/state'
import {
  Contours,
  Curve,
  Density,
  formatNumber,
  Handle,
  Histogram,
  Plot,
  Plots,
  Points,
  Probe,
  Raster,
  Readout,
  useAxis,
} from '@lab/viz'
import { BUMPS, bump, density2, grid } from '../plot/data'

const num = (v: unknown) => {
  const flat = typeof v === 'number' ? [v] : toFlat(v as Tensor)
  return flat[0]
}

// ── Figure state ─────────────────────────────────────────────────────────────────────────────────────────────────────

const FAMILY = variants(
  {
    normal: {
      label: 'Normal',
      params: { mu: slider(-2, 2, 0, { label: 'mean μ' }), sd: slider(0.3, 2.5, 1, { label: 'sd σ' }) },
    },
    laplace: {
      label: 'Laplace',
      params: { mu: slider(-2, 2, 0, { label: 'location μ' }), b: slider(0.3, 2, 0.8, { label: 'scale b' }) },
    },
    student: {
      label: 'Student t',
      params: {
        df: slider(1, 30, 3, { label: 'degrees of freedom ν', step: 1 }),
        mu: slider(-2, 2, 0, { label: 'location μ' }),
      },
    },
  },
  { label: '1 · distribution', choiceLabel: 'family' },
)

/** One declaration: variant rows, a reveal row, a setting, and a value placed on the chart; URL round trip and reset. */
export function FigureStateFigure() {
  const state = useFigureState({
    family: FAMILY,
    reveal: row('2 · reveal', {
      draws: toggle(false, 'draws'),
      n: choice([200, 1000, 5000], 1000, { label: 'how many' }),
    }),
    x0: slider(-5, 5, 1, { onChart: true, label: 'x₀' }),
    logY: setting(false, 'log density'),
  })
  const { family } = state
  const dist: Univariate = useMemo(() => {
    switch (family.key) {
      case 'normal':
        return Normal(family.values.mu, family.values.sd)
      case 'laplace':
        return Laplace(family.values.mu, family.values.b)
      case 'student':
        return StudentT(family.values.df, family.values.mu, 1)
    }
  }, [family])
  const draws = useMemo(
    () => (state.reveal.draws ? toFlat(dist.sample(stream('kit-state'), { shape: [state.reveal.n] }) as Tensor) : null),
    [dist, state.reveal.draws, state.reveal.n],
  )
  const x = useAxis({ label: 'x', range: [-5, 5] })
  const y = useAxis({ label: 'p(x)', hold: 'union', key: `${family.key}${state.logY}`, log: state.logY })
  const p0 = num(dist.prob(state.x0))
  const url = toEntries(state.schema, state.json)
  return (
    <Figure
      title="Figure state: one declaration, every binding"
      purpose="One schema gives the rows, typed values, a handle on the chart, reset and the URL."
      state={state}
      readouts={{
        'at x₀': (
          <>
            <Readout label="x₀" value={formatNumber(state.x0)} />
            <Readout label="p(x₀)" value={formatNumber(p0)} />
          </>
        ),
        'in the URL': (
          <Readout
            label="non-default values"
            value={url.length ? url.map(([k, v]) => `${k}=${v}`).join(' & ') : 'none'}
          />
        ),
      }}
      caption="Drag the vertical line to move x₀ (it has no slider: it is placed on the chart). Switch families and back: each keeps its values. Change anything and reload: the URL brings it back; the reset button clears it."
    >
      <Plot x={x} y={y}>
        {draws && <Histogram name="draws" values={draws} bins={50} range={[-5, 5]} muted />}
        <Density dist={dist} name={family.label} range={[-5, 5]} emphasis />
        <Handle {...state.handle('x0', { label: 'x₀' })} />
      </Plot>
    </Figure>
  )
}

// ── Probes ───────────────────────────────────────────────────────────────────────────────────────────────────────────

/** One probe shared by a curve and its derivative; a second, two-dimensional probe on a raster with its slice. */
export function SharedProbeFigure() {
  const state = useFigureState({
    omega: slider(0.5, 3, 1.5, { label: 'frequency ω' }),
    x0: slider(-4, 4, 1, { onChart: true }),
  })
  const probe = useProbe({ x: state.bind('x0'), label: 'x₀' })
  const xs = useMemo(() => grid(-4, 4, 401), [])
  const f = (v: number) => Math.sin(state.omega * v) * Math.exp(-0.1 * v * v)
  const df = (v: number) => (f(v + 1e-5) - f(v - 1e-5)) / 2e-5
  const fy = useMemo(() => xs.map(f), [xs, state.omega]) // eslint-disable-line react-hooks/exhaustive-deps -- f is a function of ω
  const dfy = useMemo(() => xs.map(df), [xs, state.omega]) // eslint-disable-line react-hooks/exhaustive-deps -- as above
  const x = useAxis({ label: 'x' })
  const yf = useAxis({ label: 'f(x)' })
  const yd = useAxis({ label: 'f′(x)', hold: 'union' })
  const px = probe.x ?? 0
  return (
    <Figure
      title="Probe: one point on every plot that shows its coordinate"
      purpose="A probe is figure state with a position: dragging it on either plot moves it on both."
      state={state}
      readouts={<ProbeReadout probe={probe} values={{ 'f(x₀)': f(px), 'f′(x₀)': df(px) }} />}
      caption="Drag the dashed line on the top or the bottom plot; the dots mark f and f′ at x₀."
    >
      <Plots rows={2} hoverGroup>
        <Plot x={x} y={yf}>
          <Curve name="f" x={xs} y={fy} />
          <Probe probe={probe} at={f(px)} />
        </Plot>
        <Plot x={x} y={yd}>
          <Curve name="f′" x={xs} y={dfy} slot={1} />
          <Probe probe={probe} at={df(px)} slot={1} />
        </Plot>
      </Plots>
    </Figure>
  )
}

/** Click-to-set on a raster: a point probe, the conditional slice through it, and its values. */
export function RasterProbeFigure() {
  const state = useFigureState({
    qx: slider(-3.5, 3.5, 0.4, { onChart: true }),
    qy: slider(-3.5, 3.5, -0.6, { onChart: true }),
  })
  const probe = useProbe({ x: state.bind('qx'), y: state.bind('qy'), label: 'x₁', yLabel: 'x₂' })
  const xs = useMemo(() => grid(-3.5, 3.5, 71), [])
  const z = useMemo(() => xs.map((yv) => xs.map((xv) => density2(xv, yv))), [xs])
  const slice = useMemo(() => xs.map((yv) => density2(probe.x ?? 0, yv)), [xs, probe.x])
  const x = useAxis({ label: 'x₁' })
  const y = useAxis({ label: 'x₂' })
  const s = useAxis({ label: 'p(x₁, x₂) at x₁', hold: 'union' })
  return (
    <Figure
      title="Probe on a raster: press anywhere to query"
      purpose="A point probe on a heatmap is the conditioning query: the slice at x₁ is drawn beside it, sharing x₂."
      state={state}
      readouts={<ProbeReadout probe={probe} values={{ 'p(x₁, x₂)': density2(probe.x ?? 0, probe.y ?? 0) }} />}
      caption="Press or drag anywhere on the heatmap: the probe jumps there. The slice on the right is the density along x₂ at the probe's x₁; its dot is the probed value."
    >
      <Plots cols={2} widths={[3, 1.2]}>
        <Plot x={x} y={y}>
          <Raster x={xs} y={xs} z={z} valueLabel="density" />
          <Contours x={xs} y={xs} z={z} levels={[0.2, 0.4, 0.6]} />
          <Probe probe={probe} />
        </Plot>
        <Plot x={s} y={y}>
          <Curve name="slice" x={slice} y={xs} slot={1} />
          {/* The same probe, by its x₂ only: a horizontal guide that drags x₂, and a dot at the probed value. */}
          <Probe probe={{ ...probe, x: undefined }} />
          <Points name="p at the probe" x={[density2(probe.x ?? 0, probe.y ?? 0)]} y={[probe.y ?? 0]} slot={1} live />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── The scheduler ────────────────────────────────────────────────────────────────────────────────────────────────────

const gradient = (x: number, y: number): [number, number] =>
  BUMPS.reduce<[number, number]>(
    ([gx, gy], b) => {
      const v = bump(b, x, y) / b.width ** 2
      return [gx - v * (x - b.at[0]), gy - v * (y - b.at[1])]
    },
    [0, 0],
  )

/**
 * Gradient ascent from `start` in `steps` small steps, keeping 200 points: deliberately expensive (about 3 µs a step),
 * to stand for a derived layer that costs more than a frame.
 */
function ascent(start: [number, number], steps: number) {
  const keep = Math.max(1, Math.floor(steps / 200))
  const eta = 60 / steps
  let [x, y] = start
  const px = [x]
  const py = [y]
  for (let i = 1; i <= steps; i++) {
    const [gx, gy] = gradient(x, y)
    x += eta * gx * 20
    y += eta * gy * 20
    if (i % keep === 0) {
      px.push(x)
      py.push(y)
    }
  }
  return { x: px, y: py }
}

const MODES = ['frame', 'release', 'memo'] as const

/** An expensive derived path under a draggable start: the scheduler's modes against a plain memo, measured. */
export function SchedulerFigure() {
  const state = useFigureState({
    mode: choice(
      [
        { value: 'frame', label: 'useComputed: frame' },
        { value: 'release', label: 'useComputed: release' },
        { value: 'memo', label: 'useMemo, no scheduler' },
      ],
      'frame',
      { label: 'scheduling' },
    ),
    steps: choice([2000, 20000, 100000, 400000], 100000, { label: 'steps per path (cost)' }),
    sx: slider(-3.5, 3.5, 2.2, { onChart: true }),
    sy: slider(-3.5, 3.5, 2.4, { onChart: true }),
  })
  const mode = state.mode as (typeof MODES)[number]
  const memo = useMemo(
    () => (mode === 'memo' ? ascent([state.sx, state.sy], state.steps) : null),
    [mode, state.sx, state.sy, state.steps],
  )
  const computed = useComputed(
    () => (mode === 'memo' ? null : ascent([state.sx, state.sy], state.steps)),
    [mode, state.sx, state.sy, state.steps],
    { mode: mode === 'memo' ? 'frame' : mode },
  )
  const path = memo ?? computed.value ?? { x: [], y: [] }
  const xs = useMemo(() => grid(-3.5, 3.5, 71), [])
  const z = useMemo(() => xs.map((yv) => xs.map((xv) => density2(xv, yv))), [xs])
  const x = useAxis({ label: 'x₁' })
  const y = useAxis({ label: 'x₂' })
  return (
    <Figure
      title="useComputed: an expensive layer during a drag"
      purpose="The handle moves on every pointer event; the derived path follows as fast as it can be computed, dimmed while it lags."
      state={state}
      readouts={{
        scheduler: (
          <>
            <Readout label="last run" value={mode === 'memo' ? '— (in render)' : `${computed.ms.toFixed(1)} ms`} />
            <Readout label="runs" value={mode === 'memo' ? '—' : String(computed.runs)} />
            <Readout label="slow (over 8 ms)" value={mode === 'memo' ? '—' : computed.slow ? 'yes' : 'no'} />
            <Readout label="stale now" value={computed.stale ? 'yes' : 'no'} />
          </>
        ),
      }}
      caption="Drag the start point. With 100 000 steps a path takes longer than a frame: under the scheduler the start point keeps up and the path catches up (faded while it lags); under useMemo every pointer move waits for the path. 'release' computes only when you let go."
    >
      <Plot x={x} y={y}>
        <Raster x={xs} y={xs} z={z} valueLabel="density" fillOpacity={0.85} />
        <Curve name="ascent path" x={path.x} y={path.y} slot={1} width={2.5} stale={computed.stale} live />
        <Handle {...state.handle(['sx', 'sy'], { label: 'start' })} />
      </Plot>
    </Figure>
  )
}

// ── Equations ────────────────────────────────────────────────────────────────────────────────────────────────────────

/** The normal density at a point, with the parameters and the result substituted live. */
export function EquationFigure() {
  const state = useFigureState({
    mu: slider(-2, 2, 0, { label: 'mean μ' }),
    sigma: slider(0.3, 2.5, 1, { label: 'sd σ' }),
    x0: slider(-4, 4, 1, { onChart: true }),
  })
  const dist = useMemo(() => Normal(state.mu, state.sigma), [state.mu, state.sigma])
  const p = num(dist.prob(state.x0))
  const zScore = (state.x0 - state.mu) / state.sigma
  const x = useAxis({ label: 'x', range: [-5, 5] })
  const y = useAxis({ label: 'p(x)', hold: 'union' })
  return (
    <Figure
      title="Equation: the density at x₀ with live values"
      purpose="The equation band shows the formula with the current numbers in it, so each control's effect reads as arithmetic."
      state={state}
      equation={
        <Equation>
          {tex`p(${live(state.x0, { digits: 3 })}) = \frac{1}{${live(state.sigma, { digits: 3 })} \cdot \sqrt{2\pi}} \exp\!\Big(-\tfrac12 \big(${live(zScore, { digits: 3 })}\big)^2\Big) = ${live(p, { digits: 4, strong: true })}`}
        </Equation>
      }
      caption="Drag x₀ or move μ and σ: the highlighted numbers in the band are the live values; z = (x₀ − μ)/σ."
    >
      <Plot x={x} y={y}>
        <Density dist={dist} name="N(μ, σ²)" range={[-5, 5]} />
        <Handle {...state.handle('x0', { label: 'x₀' })} />
      </Plot>
    </Figure>
  )
}

/** The chain rule for f(x) = sin(x²), one assignment per step, played. */
export function EquationStepsFigure() {
  const state = useFigureState({ x: slider(-2, 2, 0.8, { label: 'input x' }) })
  const [step, setStep] = useState(0)
  const u = state.x ** 2
  const v = Math.sin(u)
  const du = 2 * state.x
  const dv = Math.cos(u)
  const steps = [
    {
      tex: tex`u = x^2 = ${live(state.x, { digits: 3 })}^2 = ${live(u, { digits: 4 })}`,
      note: 'Forward: square the input.',
    },
    { tex: tex`f = \sin u = \sin ${live(u, { digits: 4 })} = ${live(v, { digits: 4 })}`, note: 'Forward: the output.' },
    {
      tex: tex`\frac{\partial f}{\partial u} = \cos u = ${live(dv, { digits: 4 })}`,
      note: 'Backward: the local derivative of sin.',
    },
    {
      tex: tex`\frac{df}{dx} = \frac{\partial f}{\partial u}\,\frac{du}{dx} = ${live(dv, { digits: 4 })} \cdot ${live(du, { digits: 4 })} = ${live(dv * du, { digits: 4, strong: true })}`,
      note: 'Backward: the chain rule multiplies the local derivatives.',
    },
  ]
  return (
    <Figure
      title="EquationSteps: the chain rule, one assignment at a time"
      purpose="A walkthrough is a sequence of equations driven by a Player; the earlier steps stay visible, faded."
      state={state}
      hoverReadout={false}
      caption="Play or step through the forward and backward passes; move x and every step updates."
    >
      <EquationSteps steps={steps} step={step} onStep={setStep} />
    </Figure>
  )
}
