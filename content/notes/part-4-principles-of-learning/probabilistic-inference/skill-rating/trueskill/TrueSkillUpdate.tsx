import { useMemo } from 'react'
import {
  Area,
  choice,
  Curve,
  Diagram,
  factor,
  Figure,
  float,
  formatNumber,
  int,
  link,
  MathText,
  Plot,
  Readout,
  type Segment,
  Segments,
  seriesLayers,
  type SeriesSpec,
  slider,
  useAxis,
  useFigureState,
  variable,
} from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'
import { TS_DEFAULTS, drawMargin, gauss, trueSkill1v1, type Outcome } from '../_shared/skill'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'

const SKILL_RANGE: [number, number] = [0, 50]
const SKILL_X = toFlat(linspace(SKILL_RANGE[0], SKILL_RANGE[1], 301))

const STEPS = [
  'Priors. Each skill is a Gaussian belief $\\Gauss(\\mu_i, \\sigma_i^2 + \\tau^2)$.',
  'Down to performances. Each performance message is the skill belief widened by $\\beta^2$.',
  'Down to the difference and back. $d = p_A - p_B$ is $\\Gauss(t, c^2)$; the outcome truncates it, and EP replaces the truncated density by the Gaussian with the same mean and variance.',
  'Up to the skills. The moment-matched message returns through the linear factors and updates both skills.',
]

/** Which nodes of the 1v1 factor graph each step is about. */
const ACTIVE: string[][] = [
  ['sA', 'sB'],
  ['pA', 'pB'],
  ['d', 'out'],
  ['sA', 'sB'],
]

function graph(step: number): DiagramSpec {
  const on = new Set(ACTIVE[step])
  const v = (id: string, x: number, y: number, label: string) => variable(id, x, y, label, { highlight: on.has(id) })
  const e = (a: string, b: string) => link(a, b, false)
  return {
    unit: 38,
    nodes: [
      factor('fA', 0, 0, ''),
      factor('fB', 2, 0, ''),
      v('sA', 0, 1.1, '$s_A$'),
      v('sB', 2, 1.1, '$s_B$'),
      factor('gA', 0, 2.2, ''),
      factor('gB', 2, 2.2, ''),
      v('pA', 0, 3.3, '$p_A$'),
      v('pB', 2, 3.3, '$p_B$'),
      factor('h', 1, 4.4, ''),
      v('d', 1, 5.5, '$d$'),
      factor('out', 1, 6.6, '', 's'),
    ],
    edges: [
      e('fA', 'sA'),
      e('fB', 'sB'),
      e('sA', 'gA'),
      e('sB', 'gB'),
      e('gA', 'pA'),
      e('gB', 'pB'),
      e('pA', 'h'),
      e('pB', 'h'),
      e('h', 'd'),
      e('d', 'out'),
    ],
  }
}

export function TrueSkillUpdate() {
  const state = useFigureState({
    step: int(3, { min: 0, max: 3, step: 1, label: 'message step', format: (v) => String(v) }),
    outcome: choice<Outcome>(
      [
        { value: 'win', label: 'A wins' },
        { value: 'draw', label: 'draw' },
        { value: 'loss', label: 'B wins' },
      ],
      'loss',
      { label: 'outcome' },
    ),
    pDraw: slider(0, 0.6, 0.1, { step: 0.01, label: 'draw probability (sets ε)' }),
    muA: float(30, { min: 5, max: 45, step: 0.5, label: 'μ of A' }),
    sA: float(4, { min: 0.5, max: 12, step: 0.1, label: 'σ of A' }),
    muB: float(22, { min: 5, max: 45, step: 0.5, label: 'μ of B' }),
    sB: float(6, { min: 0.5, max: 12, step: 0.1, label: 'σ of B' }),
    beta: float(TS_DEFAULTS.beta, { min: 0.5, max: 12, step: 0.05, label: 'performance noise β' }),
  })

  const r = useMemo(() => {
    const eps = drawMargin(state.pDraw, state.beta)
    const u = trueSkill1v1({ mu: state.muA, sigma: state.sA }, { mu: state.muB, sigma: state.sB }, state.outcome, {
      beta: state.beta,
      tau: TS_DEFAULTS.tau,
      eps,
    })
    return { u, eps }
  }, [state.muA, state.sA, state.muB, state.sB, state.beta, state.pDraw, state.outcome])

  const { u, eps } = r
  const k = state.step
  const sdA = Math.sqrt(u.var1)
  const sdB = Math.sqrt(u.var2)

  const skillSeries = useMemo<SeriesSpec[]>(() => {
    const curve = (name: string, m: number, s: number, slot: number, dashed = false): SeriesSpec => ({
      name,
      type: 'line',
      x: SKILL_X,
      y: SKILL_X.map((x) => gauss(x, m, s)),
      slot,
      dashed,
    })
    const out: SeriesSpec[] = []
    const prior = k === 3
    out.push(curve(prior ? 'A prior' : 'A skill', state.muA, sdA, 0, prior))
    out.push(curve(prior ? 'B prior' : 'B skill', state.muB, sdB, 1, prior))
    if (k === 1 || k === 2) {
      out.push(curve('A performance', state.muA, Math.sqrt(u.var1 + state.beta ** 2), 0, true))
      out.push(curve('B performance', state.muB, Math.sqrt(u.var2 + state.beta ** 2), 1, true))
    }
    if (k === 3) {
      out.push(curve('A posterior', u.p1.mu, u.p1.sigma, 0))
      out.push(curve('B posterior', u.p2.mu, u.p2.sigma, 1))
    }
    return out
  }, [k, state.muA, state.muB, sdA, sdB, u, state.beta])

  const diff = useMemo(() => {
    const lo = u.t - 4 * u.c
    const hi = u.t + 4 * u.c
    const xs = toFlat(linspace(lo, hi, 401))
    const inside = (d: number) =>
      state.outcome === 'win' ? d > eps : state.outcome === 'loss' ? d < -eps : Math.abs(d) <= eps
    const series = [
      { name: 'prior of d', x: xs, y: xs.map((d) => gauss(d, u.t, u.c)), muted: true, dashed: true },
      {
        name: 'truncated by the outcome',
        x: xs,
        y: xs.map((d) => (inside(d) ? gauss(d, u.t, u.c) / u.pOutcome : 0)),
        slot: 2,
      },
      {
        name: 'moment-matched Gaussian',
        x: xs,
        y: xs.map((d) => gauss(d, u.dMean, Math.sqrt(u.dVar))),
        emphasis: true,
      },
    ] as const
    const top = Math.max(...series[1].y, ...series[2].y)
    const segments: Segment[] =
      eps > 0
        ? [
            { from: [-eps, 0], to: [-eps, top] },
            { from: [eps, 0], to: [eps, top] },
          ]
        : []
    segments.push({ from: [0, 0], to: [0, top] })
    return { series, segments, range: [lo, hi] as [number, number] }
  }, [u, eps, state.outcome])

  const spec = useMemo(() => graph(k), [k])

  const xAxis = useAxis({ label: 'skill', range: SKILL_RANGE })
  const yAxis = useAxis({ label: 'density', hold: 'union' })
  const xAxis2 = useAxis({ label: 'performance difference d', range: diff.range })
  const yAxis2 = useAxis({ label: 'density', hold: 'union' })
  return (
    <Figure
      title="One TrueSkill update, message by message"
      state={state}
      caption="Set the two skill beliefs and the outcome, then step through the messages. The lower chart shows the performance difference d = p_A − p_B: its Gaussian prior (dashed), the part the outcome allows (shaded, between the grey lines at ±ε for a draw), and the Gaussian with the same mean and variance (dark). An unexpected result gives a large v, so large shifts in both means; the player with the larger σ moves further."

      readouts={
        <>
          <Readout label="P(outcome)" value={formatNumber(u.pOutcome)} />
          <Readout label="t" value={formatNumber(u.t)} />
          <Readout label="c" value={formatNumber(u.c)} />
          <Readout label="ε" value={formatNumber(eps)} />
          <Readout label="v" value={formatNumber(u.v)} />
          <Readout label="w" value={formatNumber(u.w)} />
          <Readout label="A: μ, σ" value={`${formatNumber(u.p1.mu)}, ${formatNumber(u.p1.sigma)}`} />
          <Readout label="B: μ, σ" value={`${formatNumber(u.p2.mu)}, ${formatNumber(u.p2.sigma)}`} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-[120px_1fr]">
        <div className="mx-auto w-28 md:w-full">
          <Diagram spec={spec} ariaLabel="Two-player TrueSkill factor graph with the current step highlighted" />
        </div>
        <div className="space-y-2">
          <p className="min-h-10 text-xs text-muted-foreground">
            <MathText text={STEPS[k]} />
          </p>
          <Plot x={xAxis} y={yAxis} height={240}>
            {seriesLayers(skillSeries)}
          </Plot>
          {k >= 2 && (
            <Plot x={xAxis2} y={yAxis2} height={200}>
              <Curve {...diff.series[0]} />
              <Area {...diff.series[1]} />
              <Curve {...diff.series[2]} />
              <Segments segments={diff.segments} />
            </Plot>
          )}
        </div>
      </div>
    </Figure>
  )
}
