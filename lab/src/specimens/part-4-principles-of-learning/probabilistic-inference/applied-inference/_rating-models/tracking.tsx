/**
 * Tracking a change in skill: one focal player whose true skill steps (or drifts, or wanders) plays opponents of known
 * skill; Elo, Glicko, Glicko-2 and TrueSkill follow it online and TrueSkill Through Time smooths over all the games.
 * The stream comes from `aifn-methods/data/synthetic` (`focalPlayerStream`), the ratings from
 * `aifn-methods/inference/rating-models` (`rateStream`, `trueSkillThroughTime`), and the scores from core
 * `trackingMetrics`.
 */
import { useMemo } from 'react'
import { focalPlayerStream, type SkillPath } from 'aifn-methods/data/synthetic'
import {
  THURSTONE_BETA,
  rateStream,
  trueSkillThroughTime,
  type RaterSpec,
  type RatingTrace,
} from 'aifn-methods/inference/rating-models'
import { trackingMetrics, type TrackingMetrics } from 'aifn-compute/inference/filtering'
import { stream } from 'aifn-compute/foundation/random'
import { ControlRow, Figure } from 'aifn-render/layout'
import {
  choice,
  float,
  int,
  row,
  setting,
  slider,
  useComputed,
  useFigureState,
  type AnyValues,
} from 'aifn-render/state'
import { Area, Curve, formatNumber, Handle, Plot, Plots, Readout, useAxis } from 'aifn-render/viz'

const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(digits))) : '—')
const pathOf = (v: AnyValues) => String(v.path)

const NAMES = ['Elo', 'Glicko', 'Glicko-2', 'TrueSkill', 'TrueSkill Through Time'] as const

type Run = { name: string; mean: Float64Array; sd: Float64Array | null; metrics: TrackingMetrics }

export function SkillTrackingSpecimen() {
  const state = useFigureState({
    truth: row('1 · true skill', {
      path: choice(
        [
          { value: 'step', label: 'step' },
          { value: 'drift', label: 'linear drift' },
          { value: 'random-walk', label: 'random walk' },
          { value: 'constant', label: 'constant' },
        ],
        'step',
        { label: 'change' },
      ),
      games: int(300, { ge: 20, le: 5000, suggestions: [100, 300, 1000], label: 'games' }),
      at: int(150, {
        ge: 1,
        le: 5000,
        label: 'change at game',
        when: (v) => pathOf(v) === 'step' || pathOf(v) === 'drift',
      }),
      size: slider(-500, 500, 200, { step: 10, label: 'step size (points)', when: (v) => pathOf(v) === 'step' }),
      rate: float(2, { suggestions: [0.5, 1, 2, 5], label: 'drift per game', when: (v) => pathOf(v) === 'drift' }),
      walk: float(10, {
        gt: 0,
        suggestions: [2, 5, 10, 20],
        label: 'walk sd per game',
        when: (v) => pathOf(v) === 'random-walk',
      }),
      pause: int(0, {
        ge: 0,
        suggestions: [0, 30, 100, 365],
        label: 'break before the change (days)',
        when: (v) => pathOf(v) === 'step' || pathOf(v) === 'drift',
      }),
      opponents: choice(
        [
          { value: 'close', label: 'matched to true skill' },
          { value: 'field', label: 'a fixed field' },
        ],
        'close',
        { label: 'opponents' },
      ),
      seed: int(2, { ge: 0, le: 9999, label: 'seed' }),
    }),
    raters: row('2 · algorithms', {
      k: float(32, { gt: 0, le: 400, suggestions: [8, 16, 32, 64], label: 'Elo K' }),
      c: float(15, { ge: 0, suggestions: [0, 5, 15, 30], label: 'Glicko c per day' }),
      tau2: float(0.5, { gt: 0, le: 2, suggestions: [0.3, 0.5, 1.2], label: 'Glicko-2 τ' }),
      tauTs: float(20, { gt: 0, suggestions: [5, 10, 20, 40], label: 'TrueSkill τ per day' }),
      beta: float(Math.round(THURSTONE_BETA), { gt: 0, suggestions: [100, 196, 300], label: 'TrueSkill β' }),
      smoother: setting(true, 'TrueSkill Through Time'),
    }),
  })
  const { path, games, at, size, rate, walk, pause, opponents, seed } = state.truth
  const { k, c, tau2, tauTs, beta, smoother } = state.raters
  const start = 1500
  const change = Math.min(at, games - 1)
  const skillPathSpec: SkillPath =
    path === 'step'
      ? { kind: 'step', at: change, size }
      : path === 'drift'
        ? { kind: 'drift', from: change, perRound: rate }
        : path === 'random-walk'
          ? { kind: 'random-walk', sd: walk }
          : { kind: 'constant' }
  const result = useComputed(
    () => {
      const f = focalPlayerStream(stream(`tracking/${seed}`), {
        games,
        start,
        path: skillPathSpec,
        opponents: opponents as 'close' | 'field',
        pause: pause > 0 && (path === 'step' || path === 'drift') ? { at: change, rounds: pause } : undefined,
      })
      // Every system starts at the true starting skill with its usual starting uncertainty.
      const specs: RaterSpec[] = [
        { kind: 'elo', k, initial: start },
        { kind: 'glicko', c, initial: start },
        { kind: 'glicko2', tau: tau2, initial: start },
        { kind: 'trueskill', tau: tauTs, beta, initial: start, drawProbability: 0 },
      ]
      const traces: RatingTrace[] = specs.map((s) => rateStream(f, s, { fixed: f.fixed }))
      if (smoother)
        traces.push(trueSkillThroughTime(f, { tau: tauTs, beta, initial: start, drawProbability: 0, fixed: f.fixed }))
      const P = f.players
      const burnIn = Math.min(30, Math.floor(games / 5))
      const runs: Run[] = traces.map((t, i) => {
        // The estimate after game g, against the skill during game g.
        const row = (g: number) => (f.gameRound[g] + 1) * P
        const mean = Float64Array.from({ length: games }, (_, g) => t.mean[row(g)])
        const sdRaw = Float64Array.from({ length: games }, (_, g) => t.sd[row(g)])
        const sd = Number.isNaN(sdRaw[0]) ? null : sdRaw
        return {
          name: NAMES[i],
          mean,
          sd,
          metrics: trackingMetrics(mean, f.truth, sd, {
            change: path === 'step' ? change : undefined,
            burnIn,
          }),
        }
      })
      return { truth: f.truth, runs }
    },
    [games, path, change, size, rate, walk, pause, opponents, seed, k, c, tau2, tauTs, beta, smoother],
    { mode: 'frame' },
  )
  const { truth, runs } = result.value
  const xs = useMemo(() => Float64Array.from({ length: truth.length }, (_, g) => g + 1), [truth])
  const lo = Math.min(start, start + (path === 'step' ? size : 0)) - 400
  const hi = Math.max(start, start + (path === 'step' ? size : 0)) + 400
  const gameAxis = useAxis({ label: 'game', range: [0, games], key: games })
  const ratingAxis = useAxis({
    label: 'rating (Elo scale)',
    hold: 'union',
    range: [lo, hi],
    key: `${path}/${games}/${seed}`,
  })
  const errorAxis = useAxis({ label: 'estimate − true skill', range: [-400, 400] })
  const errors = useMemo(() => runs.map((r) => Float64Array.from(r.mean, (v, g) => v - truth[g])), [runs, truth])
  return (
    <Figure
      title="Tracking a change in skill"
      purpose="A player's skill jumps; each rating system follows at a speed set by how much it trusts new games: Elo by its fixed K, Glicko, Glicko-2 and TrueSkill by an uncertainty that the drift parameter keeps from shrinking to zero; the smoother, using later games too, places the jump where it happened."
      state={state}
      defaultSize="XL"
      controls={
        path === 'step' ? (
          <ControlRow label="step">
            <span className="col-span-full text-xs text-muted-foreground">
              Drag the step on the left chart: across to move when it happens, up or down to change its size.
            </span>
          </ControlRow>
        ) : undefined
      }
      readouts={Object.fromEntries(
        runs.map((r) => [
          r.name,
          <>
            {path === 'step' && <Readout label="lag (games to 90 %)" value={fmt(r.metrics.lag)} />}
            {path === 'step' && <Readout label="overshoot" value={fmt(r.metrics.overshoot)} />}
            {path === 'step' && <Readout label="noise once settled" value={fmt(r.metrics.noise)} />}
            <Readout label={path === 'step' ? 'RMSE before' : 'RMSE'} value={fmt(r.metrics.rmseBefore)} />
            {path === 'step' && <Readout label="RMSE after" value={fmt(r.metrics.rmseAfter)} />}
            <Readout label="±2 sd coverage" value={r.sd ? `${fmt(100 * r.metrics.coverage)} %` : 'no sd'} />
          </>,
        ]),
      )}
      caption="aifn focalPlayerStream: one player starting at 1500 whose true skill steps, drifts linearly or follows a random walk, each game against a new opponent of known skill (the player's true skill plus N(0, 150²), or a fixed field around 1500); outcomes from the Bradley–Terry model. One game a day, with an optional break of some days just before the change (the skill changes during the break). rateStream runs Elo, Glicko-1 (RD grows by c per day, starting 350), Glicko-2 (volatility 0.06, τ, one rating period a day) and TrueSkill (σ₀ 350, β, dynamics τ per day, on Elo's scale) on the same games; trueSkillThroughTime smooths them with the same τ and β. Scores from trackingMetrics after a 30-game burn-in: the lag is the games after the change until the estimate covers 90 % of the step, the noise is the RMSE once there, coverage is the share of games whose true skill lies inside the ±2 sd band (a calibrated band covers about 95 %). Left: estimates (bands ±2 sd) over the true skill (ink); right: the errors. Bias against variance: a large K (or c, or τ) reacts in a few games but stays noisy; a small one is smooth but lags. Set a break: Glicko, Glicko-2 and TrueSkill then react faster than in steady play, because their uncertainty grows by c√t (or τ√t) over t idle days, while Elo's K is the same after a year away as after a day. The online systems only see the step after it (they lag); the smoother sees it from both sides and centres it. A logistic Kalman filter on the same games coincides with Glicko-1 against opponents of known skill (tested in aifn), so it is not drawn."
    >
      <Plots cols={2}>
        <Plot x={gameAxis} y={ratingAxis} title="estimates over the true skill">
          {runs.map((r, i) =>
            r.sd ? (
              <Area
                key={`band${i}`}
                name={r.name}
                slot={i}
                x={xs}
                y={Float64Array.from(r.mean, (v, g) => v + 2 * r.sd![g])}
                base={Float64Array.from(r.mean, (v, g) => v - 2 * r.sd![g])}
                opacity={0.07}
                line={false}
              />
            ) : null,
          )}
          {runs.map((r, i) => (
            <Curve key={i} name={r.name} slot={i} x={xs} y={r.mean} stale={result.stale} />
          ))}
          <Curve name="true skill" x={xs} y={truth} emphasis />
          {path === 'step' && (
            <Handle
              kind="point"
              at={[change + 1, start + size]}
              onDrag={([x, y]) => {
                state.set('truth.at', Math.max(1, Math.min(games - 1, Math.round(x - 1))))
                state.set('truth.size', Math.max(-500, Math.min(500, Math.round((y - start) / 10) * 10)))
              }}
              label={`step at game ${change + 1}, ${size > 0 ? '+' : ''}${size}`}
            />
          )}
        </Plot>
        <Plot x={gameAxis} y={errorAxis} title="errors">
          <Curve name="no error" x={[0, games]} y={[0, 0]} emphasis dashed thin />
          {errors.map((e, i) => (
            <Curve key={i} name={runs[i].name} slot={i} x={xs} y={e} stale={result.stale} thin />
          ))}
        </Plot>
      </Plots>
    </Figure>
  )
}
