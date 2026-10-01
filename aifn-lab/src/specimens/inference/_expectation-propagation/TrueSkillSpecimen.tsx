import { Normal } from 'aifn/probability/distributions'
import { drawMargin, trueSkillEp, trueSkillUpdate, type Rating } from 'aifn-applied/inference/rating-models'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { trace } from 'aifn/foundation/trace'
import { useMemo, useState } from 'react'
import { Player } from '@lab/controls'
import { Figure } from '@lab/layout'
import { seriesColor, useTheme } from '@lab/design'
import { choice, row, slider, useFigureState } from '@lab/state'
import { Curve, Handle, Plot, Readout, formatNumber, useAxis } from '@lab/viz'

const GRID = linspace(0, 50, 401)
const X = toFlat(GRID)
const pdf = (r: Rating) => toFlat(Normal(r.mean, r.sd).prob(GRID))

/** One two-player TrueSkill update: priors, outcome, posteriors. */
export function MatchSpecimen() {
  const state = useFigureState({
    p1: row('1 · player 1', {
      mu1: slider(5, 45, 25, { label: 'mean', step: 0.5 }),
      sd1: slider(1, 12, 25 / 3, { label: 'sd', step: 0.1 }),
    }),
    p2: row('2 · player 2', {
      mu2: slider(5, 45, 30, { label: 'mean', step: 0.5 }),
      sd2: slider(1, 12, 4, { label: 'sd', step: 0.1 }),
    }),
    match: row('3 · the match', {
      outcome: choice(['win', 'draw', 'loss'], 'win', { label: "player 1's result" }),
      pDraw: slider(0, 0.5, 0.1, { label: 'draw probability', step: 0.01 }),
    }),
  })
  const { mu1, sd1 } = state.p1
  const { mu2, sd2 } = state.p2
  const { outcome, pDraw } = state.match
  const beta = 25 / 6
  const mode = useTheme().resolved
  const r1 = useMemo(() => ({ mean: mu1, sd: sd1 }), [mu1, sd1])
  const r2 = useMemo(() => ({ mean: mu2, sd: sd2 }), [mu2, sd2])
  const u = trueSkillUpdate(r1, r2, outcome, { beta, drawMargin: drawMargin(pDraw, beta) })
  const curves = useMemo(
    () => ({ b1: pdf(r1), a1: pdf(u.player1), b2: pdf(r2), a2: pdf(u.player2) }),
    [r1, r2, u.player1, u.player2],
  )
  const skill = useAxis({ label: 'skill', range: [0, 50] })
  const dens = useAxis({ label: 'density', hold: 'union' })
  return (
    <Figure
      title="A TrueSkill match update"
      purpose="An upset moves the ratings far and shrinks the uncertain player's sd most; an expected result barely moves them."
      defaultSize="M"
      state={state}
      readouts={{
        update: (
          <>
            <Readout label="p(this outcome)" value={formatNumber(u.probability)} />
            <Readout label="v, w" value={`${formatNumber(u.v)}, ${formatNumber(u.w)}`} />
          </>
        ),
        after: (
          <>
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
        ),
      }}
      caption="Skills are Gaussian beliefs; performances add N(0, β²) noise with β = 25/6, and a draw means the performances differ by less than the draw margin ε. The update (Herbrich et al. 2007) truncates the performance difference to the observed outcome and moment-matches: v shifts the means and w shrinks the variances, each player in proportion to their own variance. The default is an upset: the weaker, less certain player 1 beats player 2."
    >
      <Plot x={skill} y={dens}>
        <Curve name="player 1 before" x={X} y={curves.b1} slot={0} dashed />
        <Curve name="player 1 after" x={X} y={curves.a1} slot={0} />
        <Curve name="player 2 before" x={X} y={curves.b2} slot={1} dashed />
        <Curve name="player 2 after" x={X} y={curves.a2} slot={1} />
      </Plot>
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
  const [step, setStep] = useState(0)
  const mode = useTheme().resolved
  const means = run.series.mean
  const P = PLAYERS.length
  const t = run.index
  const lines = useMemo(() => PLAYERS.map((_, p) => t.map((_, k) => means.data[k * P + p])), [t, means, P])
  const upd = useAxis({ label: 'update' })
  const skillAxis = useAxis({ label: 'mean skill' })
  const s = run.steps[Math.min(step, run.steps.length - 1)]
  const m = s.match >= 0 ? MATCHES[s.match] : null
  return (
    <Figure
      title="EP over a set of matches"
      purpose="Treating skills as fixed across the matches, EP revisits each match until the ratings agree with all of them."
      defaultSize="M"
      controls={
        <div className="col-span-full">
          <Player
            label="match update"
            value={Math.min(step, run.steps.length - 1)}
            onChange={setStep}
            count={run.steps.length}
            defaultSpeed={3}
          />
        </div>
      }
      readouts={{
        'this update': (
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
        ),
      }}
      caption="Play from the priors, or drag the vertical line. Six matches among four players, processed in order: Ann beats Bo, Bo beats Cy, Cy beats Di, Di beats Ann, Ann beats Cy, Bo draws with Di. The first six updates are online TrueSkill without dynamics; its ratings depend on the order. Later sweeps remove each match's old contribution (its site) and apply it again with what the other matches have taught, until the ratings stop moving (EP's fixed point, as in TrueSkill Through Time)."
    >
      <Plot x={upd} y={skillAxis}>
        {PLAYERS.map((name, p) => (
          <Curve key={name} name={name} x={t} y={lines[p]} slot={p} showPoints />
        ))}
        <Handle
          kind="x"
          at={t[Math.min(step, t.length - 1)]}
          label="now"
          onDrag={(v) => setStep(Math.max(0, Math.min(t.length - 1, Math.round(v))))}
        />
      </Plot>
    </Figure>
  )
}
