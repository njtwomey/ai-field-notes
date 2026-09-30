import { Normal } from 'aifn/probability/distributions'
import { drawMargin, trueSkillEp, trueSkillUpdate, type Rating } from 'aifn-applied/inference/rating-models'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { trace } from 'aifn/foundation/trace'
import { useMemo, useState } from 'react'
import { Player, Select, Slider, useParam } from '@lab/controls'
import { Figure } from '@lab/layout'
import { seriesColor, useTheme } from '@lab/design'
import { Readout, XYChart, formatNumber, type XYSeries } from '@lab/viz'

const GRID = linspace(0, 50, 401)
const X = toFlat(GRID)
const pdf = (r: Rating) => toFlat(Normal(r.mean, r.sd).prob(GRID))

/** One two-player TrueSkill update: priors, outcome, posteriors. */
export function MatchSpecimen() {
  const mu1 = useParam(25, { min: 5, max: 45, step: 0.5 })
  const sd1 = useParam(25 / 3, { min: 1, max: 12, step: 0.1 })
  const mu2 = useParam(30, { min: 5, max: 45, step: 0.5 })
  const sd2 = useParam(4, { min: 1, max: 12, step: 0.1 })
  const pDraw = useParam(0.1, { min: 0, max: 0.5, step: 0.01 })
  const [outcome, setOutcome] = useState<'win' | 'draw' | 'loss'>('win')
  const beta = 25 / 6
  const mode = useTheme().resolved
  const r1 = { mean: mu1.value, sd: sd1.value }
  const r2 = { mean: mu2.value, sd: sd2.value }
  const u = trueSkillUpdate(r1, r2, outcome, { beta, drawMargin: drawMargin(pDraw.value, beta) })
  const series: XYSeries[] = [
    { name: 'player 1 before', type: 'line', x: X, y: pdf(r1), slot: 0, dashed: true },
    { name: 'player 1 after', type: 'line', x: X, y: pdf(u.player1), slot: 0 },
    { name: 'player 2 before', type: 'line', x: X, y: pdf(r2), slot: 1, dashed: true },
    { name: 'player 2 after', type: 'line', x: X, y: pdf(u.player2), slot: 1 },
  ]
  return (
    <Figure
      title="A TrueSkill match update"
      description="An upset moves the ratings far and shrinks the uncertain player's sd most; an expected result barely moves them."
      defaultSize="M"
      controls={
        <>
          <Slider label="player 1 mean" param={mu1} />
          <Slider label="player 1 sd" param={sd1} />
          <Slider label="player 2 mean" param={mu2} />
          <Slider label="player 2 sd" param={sd2} />
          <Select label="player 1's result" value={outcome} onChange={setOutcome} options={['win', 'draw', 'loss']} />
          <Slider label="draw probability" param={pDraw} />
        </>
      }
      readouts={
        <>
          <Readout label="p(this outcome)" value={formatNumber(u.probability)} />
          <Readout label="v, w" value={`${formatNumber(u.v)}, ${formatNumber(u.w)}`} />
          <Readout
            label="player 1"
            value={`${formatNumber(u.player1.mean)} ± ${formatNumber(u.player1.sd)}`}
            color={seriesColor(mode, 0)}
          />
          <Readout
            label="player 2"
            value={`${formatNumber(u.player2.mean)} ± ${formatNumber(u.player2.sd)}`}
            color={seriesColor(mode, 1)}
          />
        </>
      }
      caption="Skills are Gaussian beliefs; performances add N(0, β²) noise with β = 25/6, and a draw means the performances differ by less than the draw margin ε. The update (Herbrich et al. 2007) truncates the performance difference to the observed outcome and moment-matches: v shifts the means and w shrinks the variances, each player in proportion to their own variance. The default is an upset: the weaker, less certain player 1 beats player 2."
    >
      <XYChart series={series} xLabel="skill" yLabel="density" rescaleOnChange={false} holdFit="union" />
    </Figure>
  )
}

const PLAYERS = ['Ann', 'Bo', 'Cy', 'Di']
const MATCHES = [
  { winner: 0, loser: 1 },
  { winner: 1, loser: 2 },
  { winner: 2, loser: 3 },
  { winner: 3, loser: 0 },
  { winner: 0, loser: 2 },
  { winner: 1, loser: 3, draw: true },
]

/** EP over a fixed set of matches: the first sweep is online TrueSkill; later sweeps revise early matches. */
export function MatchSetSpecimen() {
  const run = useMemo(
    () =>
      trace(trueSkillEp({ players: PLAYERS.map(() => ({ mean: 25, sd: 25 / 3 })), matches: MATCHES }), undefined, 200, {
        record: { mean: (s) => s.means, sd: (s) => s.sds },
      }),
    [],
  )
  const [step, setStep] = useState(MATCHES.length)
  const mode = useTheme().resolved
  const means = run.series.mean
  const P = PLAYERS.length
  const t = run.index
  const series: XYSeries[] = PLAYERS.map((name, p) => ({
    name,
    type: 'line',
    x: t,
    y: t.map((_, k) => means.data[k * P + p]),
    slot: p,
    showPoints: true,
  }))
  const s = run.steps[Math.min(step, run.steps.length - 1)]
  const m = s.match >= 0 ? MATCHES[s.match] : null
  return (
    <Figure
      title="EP over a set of matches"
      description="Treating skills as fixed across the matches, EP revisits each match until the ratings agree with all of them."
      defaultSize="M"
      controls={
        <div className="col-span-full">
          <Player label="match update" value={step} onChange={setStep} count={run.steps.length} defaultSpeed={3} />
        </div>
      }
      readouts={
        <>
          <Readout
            label="this update"
            value={
              m
                ? `${PLAYERS[m.winner]} ${m.draw ? 'draws with' : 'beats'} ${PLAYERS[m.loser]} (sweep ${s.sweep + (s.position === 0 ? 0 : 1)})`
                : 'priors'
            }
          />
          {PLAYERS.map((name, p) => (
            <Readout
              key={name}
              label={name}
              value={`${formatNumber(s.means.data[p])} ± ${formatNumber(s.sds.data[p])}`}
              color={seriesColor(mode, p)}
            />
          ))}
          <Readout label="converged" value={s.converged ? `yes, after ${s.sweep} sweeps` : 'not yet'} />
        </>
      }
      caption="Six matches among four players, processed in order: Ann beats Bo, Bo beats Cy, Cy beats Di, Di beats Ann, Ann beats Cy, Bo draws with Di. The first six updates are online TrueSkill without dynamics; its ratings depend on the order. Later sweeps remove each match's old contribution (its site) and apply it again with what the other matches have taught, until the ratings stop moving (EP's fixed point, as in TrueSkill Through Time)."
    >
      <XYChart
        series={series}
        xLabel="update"
        yLabel="mean skill"
        live={[
          {
            name: 'now',
            type: 'line',
            x: [run.index[Math.min(step, t.length - 1)], run.index[Math.min(step, t.length - 1)]],
            y: [20, 30],
            muted: true,
          },
        ]}
      />
    </Figure>
  )
}
