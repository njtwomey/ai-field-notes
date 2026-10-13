import { useMemo, useState } from 'react'
import {
  choice,
  Curve,
  Figure,
  formatNumber,
  int,
  Player,
  Plot,
  Points,
  Raster,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { stream } from 'aifn-compute/foundation/random'
import { linspace, toArray, toFlat } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { pbtToyProblem, populationBasedTrainingSteps } from 'aifn-methods/optimisation/population'

const PROBLEM = pbtToyProblem()
const AXIS = toFlat(linspace(0, 1, 61))
const SCORE = AXIS.map((t1) => AXIS.map((t0) => PROBLEM.score([t0, t1])))

/**
 * Population-based training on the toy of Jaderberg et al. (2017), Figure 2, against the same members trained apart:
 * the members' paths over the score Q(θ) and their scores over the steps.
 */
export function PbtToy() {
  const state = useFigureState({
    population: int(2, { ge: 2, le: 10, suggestions: [2, 4, 10], label: 'population P' }),
    explore: choice(
      [
        { value: 'perturb', label: 'perturb' },
        { value: 'resample', label: 'resample from the prior' },
      ],
      'perturb',
      { label: 'explore' },
    ),
    ready: int(4, { ge: 1, le: 20, suggestions: [2, 4, 8], label: 'steps between exploits' }),
    steps: int(60, { ge: 10, le: 300, suggestions: [40, 60, 120], label: 'training steps' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  })
  const { population, explore, ready, steps, seed } = state

  const runs = useMemo(() => {
    const go = (exploit: 'truncation' | 'none') =>
      trace(
        populationBasedTrainingSteps({
          population,
          exploit,
          explore: exploit === 'none' ? 'none' : (explore as 'perturb' | 'resample'),
          readyInterval: ready,
        }),
        undefined,
        steps,
        { stream: stream(`pbt-toy/${seed}`), keep: 'all' },
      ).steps
    const pbt = go('truncation')
    const apart = go('none')
    const unpack = (tr: typeof pbt) => ({
      theta: tr.map((s) => toArray(s.theta) as number[][]),
      scores: tr.map((s) => Array.from(toFlat(s.scores))),
      best: tr.map((s) => s.best),
      h: tr.map((s) => toArray(s.h) as number[][]),
      events: tr.map((s) => s.events),
    })
    return { pbt: unpack(pbt), apart: unpack(apart) }
  }, [population, explore, ready, steps, seed])

  const key = `${population}|${explore}|${ready}|${steps}|${seed}`
  const [cursor, setCursor] = useState({ key, step: 0 })
  const count = runs.pbt.theta.length
  const at = Math.min(cursor.key === key ? cursor.step : 0, count - 1)
  const members = Array.from({ length: population }, (_, k) => k)
  const ts = Array.from({ length: at + 1 }, (_, t) => t)
  const few = population <= 3
  const events = runs.pbt.events[at]
  const eventText =
    events.length === 0
      ? 'none'
      : events
          .filter((e) => e.kind === 'exploit')
          .map((e) => `${e.member} copies ${e.source}`)
          .join(', ') || 'explore only'

  const thetaX = useAxis({ label: 'θ₀', range: [0, 1] })
  const thetaY = useAxis({ label: 'θ₁', range: [0, 1], equal: thetaX })
  const stepAxis = useAxis({ label: 'training step', range: [0, steps] })
  const scoreAxis = useAxis({ label: 'score Q', range: [-0.5, 1.25] })
  return (
    <Figure
      title="Population-based training on a two-coordinate toy"
      state={state}
      caption="The toy of Jaderberg et al.: the score Q(θ) = 1.2 − θ₀² − θ₁² (colour) is maximised at the origin, but a member with hyperparameters h can only move the coordinates where h is non-zero. Two members start at (0.9, 0.9) with h = (1, 0) and h = (0, 1). Left: each member's path up to the current step under population-based training (solid) and trained apart, as in grid search (dashed). Right: each member's score over the steps. Every few steps the weaker member copies the stronger member's θ, keeps its own direction and perturbs it slightly, so the two take turns and reach the optimum; trained apart, each stalls at Q = 0.39. With four or more members, drawn from the prior, some member can move both coordinates and training apart succeeds too."
      controls={<Player value={at} onChange={(v) => setCursor({ key, step: v })} count={count} label="step" />}
      readouts={
        <>
          <Readout label="step" value={String(at)} />
          <Readout label="best Q, population-based training" value={formatNumber(runs.pbt.best[at])} />
          <Readout label="best Q, trained apart" value={formatNumber(runs.apart.best[at])} />
          <Readout
            label="h of member 0"
            value={`(${formatNumber(runs.pbt.h[at][0][0])}, ${formatNumber(runs.pbt.h[at][0][1])})`}
          />
          <Readout label="exploits this step" value={eventText} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={thetaX} y={thetaY} height={340}>
          <Raster x={AXIS} y={AXIS} z={SCORE} valueLabel="score Q" colorBar={false} />
          {members.map((k) => (
            <Curve
              key={`apart-${k}`}
              name="trained apart"
              x={ts.map((t) => runs.apart.theta[t][k][0])}
              y={ts.map((t) => runs.apart.theta[t][k][1])}
              {...(few ? { slot: k } : { muted: true })}
              dashed
              thin
            />
          ))}
          {members.map((k) => (
            <Curve
              key={`pbt-${k}`}
              name={few ? `member ${k}` : 'member'}
              x={ts.map((t) => runs.pbt.theta[t][k][0])}
              y={ts.map((t) => runs.pbt.theta[t][k][1])}
              {...(few ? { slot: k } : { muted: true })}
            />
          ))}
          <Points
            name="now"
            x={members.map((k) => runs.pbt.theta[at][k][0])}
            y={members.map((k) => runs.pbt.theta[at][k][1])}
            emphasis
          />
        </Plot>
        <Plot x={stepAxis} y={scoreAxis} height={340}>
          {members.map((k) => (
            <Curve
              key={`s-apart-${k}`}
              name="trained apart"
              x={ts}
              y={ts.map((t) => runs.apart.scores[t][k])}
              {...(few ? { slot: k } : { muted: true })}
              dashed
              thin
            />
          ))}
          {members.map((k) => (
            <Curve
              key={`s-pbt-${k}`}
              name={few ? `member ${k}` : 'member'}
              x={ts}
              y={ts.map((t) => runs.pbt.scores[t][k])}
              {...(few ? { slot: k } : { muted: true })}
            />
          ))}
          <Curve name="optimum" x={[0, steps]} y={[1.2, 1.2]} muted dashed />
        </Plot>
      </div>
    </Figure>
  )
}
