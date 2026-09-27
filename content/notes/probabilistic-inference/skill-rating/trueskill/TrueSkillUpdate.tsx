import { useMemo, useState } from 'react'
import { MathText } from '@/components/content/MathText'
import { Diagram } from '@/components/diagram/Diagram'
import { factor, link, variable } from '@/components/diagram/components'
import type { DiagramSpec } from '@/components/diagram/types'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Segment,
  type XYSeries,
} from '@/components/viz'
import { linspace } from '@/lib/math'
import { TS_DEFAULTS, drawMargin, gauss, trueSkill1v1, type Outcome } from '../_shared/skill'

const SKILL_RANGE: [number, number] = [0, 50]
const SKILL_X = linspace(SKILL_RANGE[0], SKILL_RANGE[1], 301)

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
  const muA = useParam(30, { min: 5, max: 45, step: 0.5 })
  const sA = useParam(4, { min: 0.5, max: 12, step: 0.1 })
  const muB = useParam(22, { min: 5, max: 45, step: 0.5 })
  const sB = useParam(6, { min: 0.5, max: 12, step: 0.1 })
  const beta = useParam(TS_DEFAULTS.beta, { min: 0.5, max: 12, step: 0.05 })
  const pDraw = useParam(0.1, { min: 0, max: 0.6, step: 0.01 })
  const step = useParam(3, { min: 0, max: 3, step: 1 })
  const [outcome, setOutcome] = useState<Outcome>('loss')

  const r = useMemo(() => {
    const eps = drawMargin(pDraw.value, beta.value)
    const u = trueSkill1v1({ mu: muA.value, sigma: sA.value }, { mu: muB.value, sigma: sB.value }, outcome, {
      beta: beta.value,
      tau: TS_DEFAULTS.tau,
      eps,
    })
    return { u, eps }
  }, [muA.value, sA.value, muB.value, sB.value, beta.value, pDraw.value, outcome])

  const { u, eps } = r
  const k = step.value
  const sdA = Math.sqrt(u.var1)
  const sdB = Math.sqrt(u.var2)

  const skillSeries = useMemo<XYSeries[]>(() => {
    const curve = (name: string, m: number, s: number, slot: number, dashed = false): XYSeries => ({
      name,
      type: 'line',
      x: SKILL_X,
      y: SKILL_X.map((x) => gauss(x, m, s)),
      slot,
      dashed,
    })
    const out: XYSeries[] = []
    const prior = k === 3
    out.push(curve(prior ? 'A prior' : 'A skill', muA.value, sdA, 0, prior))
    out.push(curve(prior ? 'B prior' : 'B skill', muB.value, sdB, 1, prior))
    if (k === 1 || k === 2) {
      out.push(curve('A performance', muA.value, Math.sqrt(u.var1 + beta.value ** 2), 0, true))
      out.push(curve('B performance', muB.value, Math.sqrt(u.var2 + beta.value ** 2), 1, true))
    }
    if (k === 3) {
      out.push(curve('A posterior', u.p1.mu, u.p1.sigma, 0))
      out.push(curve('B posterior', u.p2.mu, u.p2.sigma, 1))
    }
    return out
  }, [k, muA.value, muB.value, sdA, sdB, u, beta.value])

  const diff = useMemo(() => {
    const lo = u.t - 4 * u.c
    const hi = u.t + 4 * u.c
    const xs = linspace(lo, hi, 401)
    const inside = (d: number) => (outcome === 'win' ? d > eps : outcome === 'loss' ? d < -eps : Math.abs(d) <= eps)
    const series: XYSeries[] = [
      { name: 'prior of d', type: 'line', x: xs, y: xs.map((d) => gauss(d, u.t, u.c)), muted: true, dashed: true },
      {
        name: 'truncated by the outcome',
        type: 'line',
        x: xs,
        y: xs.map((d) => (inside(d) ? gauss(d, u.t, u.c) / u.pOutcome : 0)),
        slot: 2,
        area: true,
      },
      {
        name: 'moment-matched Gaussian',
        type: 'line',
        x: xs,
        y: xs.map((d) => gauss(d, u.dMean, Math.sqrt(u.dVar))),
        emphasis: true,
      },
    ]
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
  }, [u, eps, outcome])

  const spec = useMemo(() => graph(k), [k])

  return (
    <Interactive
      title="One TrueSkill update, message by message"
      caption="Set the two skill beliefs and the outcome, then step through the messages. The lower chart shows the performance difference d = p_A − p_B: its Gaussian prior (dashed), the part the outcome allows (shaded, between the grey lines at ±ε for a draw), and the Gaussian with the same mean and variance (dark). An unexpected result gives a large v, so large shifts in both means; the player with the larger σ moves further."
      controls={
        <>
          <ParamSlider label="message step" param={step} format={(v) => String(v)} withArrows />
          <ParamChoice
            label="outcome"
            value={outcome}
            onChange={setOutcome}
            options={[
              { value: 'win', label: 'A wins' },
              { value: 'draw', label: 'draw' },
              { value: 'loss', label: 'B wins' },
            ]}
          />
          <ParamSlider label="draw probability (sets ε)" param={pDraw} />
          <ParamSlider label="μ of A" param={muA} />
          <ParamSlider label="σ of A" param={sA} />
          <ParamSlider label="μ of B" param={muB} />
          <ParamSlider label="σ of B" param={sB} />
          <ParamSlider label="performance noise β" param={beta} />
        </>
      }
      readout={
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
          <XYChart series={skillSeries} xLabel="skill" yLabel="density" xRange={SKILL_RANGE} height={240} />
          {k >= 2 && (
            <XYChart
              series={diff.series}
              segments={diff.segments}
              xLabel="performance difference d"
              yLabel="density"
              xRange={diff.range}
              height={200}
            />
          )}
        </div>
      </div>
    </Interactive>
  )
}
