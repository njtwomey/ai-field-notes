import { useMemo, useState } from 'react'
import { trueSkillEp } from 'aifn-methods/inference/rating-models'
import { trace } from 'aifn-compute/foundation/trace'
import {
  Figure,
  ControlGroup,
  NumberSelector,
  Player,
  Plot,
  Curve,
  Handle,
  Readout,
  formatNumber,
  useAxis,
} from 'aifn-render'

const PLAYERS = ['Ann', 'Bo', 'Cy', 'Di']
const MATCHES = [
  { winner: 0, loser: 1 },
  { winner: 1, loser: 2 },
  { winner: 2, loser: 3 },
  { winner: 3, loser: 0 },
  { winner: 0, loser: 2 },
  { winner: 1, loser: 3, draw: true },
]

export function TournamentEpExplorer() {
  const run = useMemo(
    () =>
      trace(
        trueSkillEp({
          players: PLAYERS.map(() => ({ mean: 25, sd: 25 / 3 })),
          matches: MATCHES,
        }),
        undefined,
        200,
        {
          record: { mean: (s) => s.means, sd: (s) => s.sds },
        },
      ),
    [],
  )

  const [step, setStep] = useState(0)
  const means = run.series.mean
  const P = PLAYERS.length
  const t = run.index
  const lines = useMemo(() => PLAYERS.map((_, p) => t.map((_, k) => means.data[k * P + p])), [t, means, P])

  const updAxis = useAxis({ label: 'match update step' })
  const skillAxis = useAxis({ label: 'mean skill rating μ' })

  const s = run.steps[Math.min(step, run.steps.length - 1)]
  const m = s.match >= 0 ? MATCHES[s.match] : null

  return (
    <Figure
      title="Expectation propagation over historical matches"
      purpose="Treating skills as fixed across games, EP iteratively revisits each match, removing its cavity contribution and recalculating Gaussian beliefs until all ratings reach a consistent fixed point."
      defaultSize="M"
      controls={
        <ControlGroup>
          <NumberSelector
            label="Match step"
            value={step}
            onChange={(val) => setStep(Math.max(0, Math.min(run.steps.length - 1, val)))}
            min={0}
            max={run.steps.length - 1}
            step={1}
            suggestions={[0, 6, 12, run.steps.length - 1]}
          />
          <Player
            label="Step through EP sweeps"
            value={Math.min(step, run.steps.length - 1)}
            onChange={setStep}
            count={run.steps.length}
          />
        </ControlGroup>
      }
      readouts={
        <>
          <Readout
            label="Current update"
            value={
              m
                ? `${PLAYERS[m.winner]} ${m.draw ? 'draws with' : 'beats'} ${PLAYERS[m.loser]} (sweep ${s.sweep + (s.position === 0 ? 0 : 1)})`
                : 'Prior ratings'
            }
          />
          {PLAYERS.map((name, p) => (
            <Readout
              key={name}
              label={name}
              value={`${formatNumber(s.means.data[p])} ± ${formatNumber(s.sds.data[p])}`}
            />
          ))}
          <Readout label="Convergence" value={s.converged ? `Converged after ${s.sweep} sweeps` : 'Iterating…'} />
        </>
      }
      caption="Six matches among four players processed cyclically: Ann beats Bo, Bo beats Cy, Cy beats Di, Di beats Ann, Ann beats Cy, Bo draws with Di. The first six steps correspond to single-pass online TrueSkill (sensitive to order). Subsequent sweeps subtract each match's cavity and re-estimate marginals until messages converge."
    >
      <Plot x={updAxis} y={skillAxis}>
        {PLAYERS.map((name, p) => (
          <Curve key={name} name={name} x={t} y={lines[p]} slot={p} showPoints />
        ))}
        <Handle
          kind="x"
          at={t[Math.min(step, t.length - 1)]}
          label="step"
          onDrag={(v) => setStep(Math.max(0, Math.min(t.length - 1, Math.round(v))))}
        />
      </Plot>
    </Figure>
  )
}
