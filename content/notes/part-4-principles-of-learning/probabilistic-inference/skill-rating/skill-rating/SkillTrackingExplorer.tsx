import { useMemo, useState } from 'react'
import { focalPlayerStream, type SkillPath } from 'aifn-methods/data/synthetic'
import {
  THURSTONE_BETA,
  rateStream,
  trueSkillThroughTime,
  type RaterSpec,
  type RatingTrace,
} from 'aifn-methods/inference/rating-models'
import { trackingMetrics, type TrackingMetrics } from 'aifn/inference/filtering'
import { stream } from 'aifn/foundation/random'
import {
  Area,
  ControlRow,
  Curve,
  Figure,
  Handle,
  Plot,
  Plots,
  Readout,
  Select,
  Slider,
  formatNumber,
  useAxis,
} from 'aifn-render'

const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(digits))) : '—')

const PATH_OPTIONS = [
  { value: 'step', label: 'abrupt step jump' },
  { value: 'drift', label: 'continuous linear drift' },
  { value: 'random-walk', label: 'Brownian random walk' },
  { value: 'constant', label: 'constant latent skill' },
]

const OPPONENT_OPTIONS = [
  { value: 'close', label: 'matched opponents (close in skill)' },
  { value: 'field', label: 'fixed pool of opponents (~1500)' },
]

const NAMES = ['Elo', 'Glicko', 'Glicko-2', 'TrueSkill', 'TrueSkill Through Time'] as const

type Run = {
  name: string
  mean: Float64Array
  sd: Float64Array | null
  metrics: TrackingMetrics
}

export function SkillTrackingExplorer() {
  const [path, setPath] = useState<'step' | 'drift' | 'random-walk' | 'constant'>('step')
  const [games, setGames] = useState(300)
  const [changeAt, setChangeAt] = useState(150)
  const [stepSize, setStepSize] = useState(200)
  const [driftRate, setDriftRate] = useState(2)
  const [walkSd, setWalkSd] = useState(10)
  const [pauseDays, setPauseDays] = useState(0)
  const [opponents, setOpponents] = useState<'close' | 'field'>('close')
  const [kElo, setKElo] = useState(32)
  const [cDrift, setCDrift] = useState(15)
  const [tau2, setTau2] = useState(0.5)
  const [tauTs, setTauTs] = useState(20)
  const [smoother, setSmoother] = useState(true)

  const start = 1500
  const change = Math.min(changeAt, games - 1)

  const skillPathSpec: SkillPath = useMemo(() => {
    if (path === 'step') return { kind: 'step', at: change, size: stepSize }
    if (path === 'drift') return { kind: 'drift', from: change, perRound: driftRate }
    if (path === 'random-walk') return { kind: 'random-walk', sd: walkSd }
    return { kind: 'constant' }
  }, [path, change, stepSize, driftRate, walkSd])

  const sim = useMemo(() => {
    const seed = 2
    const f = focalPlayerStream(stream(`tracking/${seed}`), {
      games,
      start,
      path: skillPathSpec,
      opponents,
      pause: pauseDays > 0 && (path === 'step' || path === 'drift') ? { at: change, rounds: pauseDays } : undefined,
    })

    const beta = Math.round(THURSTONE_BETA)
    const specs: RaterSpec[] = [
      { kind: 'elo', k: kElo, initial: start },
      { kind: 'glicko', c: cDrift, initial: start },
      { kind: 'glicko2', tau: tau2, initial: start },
      { kind: 'trueskill', tau: tauTs, beta, initial: start, drawProbability: 0 },
    ]

    const traces: RatingTrace[] = specs.map((s) => rateStream(f, s, { fixed: f.fixed }))
    if (smoother) {
      traces.push(trueSkillThroughTime(f, { tau: tauTs, beta, initial: start, drawProbability: 0, fixed: f.fixed }))
    }

    const P = f.players
    const burnIn = Math.min(30, Math.floor(games / 5))
    const runs: Run[] = traces.map((t, i) => {
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
  }, [games, path, change, skillPathSpec, opponents, pauseDays, kElo, cDrift, tau2, tauTs, smoother])

  const { truth, runs } = sim
  const xs = useMemo(() => Float64Array.from({ length: truth.length }, (_, g) => g + 1), [truth])

  const lo = Math.min(start, start + (path === 'step' ? stepSize : 0)) - 400
  const hi = Math.max(start, start + (path === 'step' ? stepSize : 0)) + 400

  const gameAxis = useAxis({ label: 'games played', range: [0, games], key: games })
  const ratingAxis = useAxis({
    label: 'rating (Elo scale)',
    hold: 'union',
    range: [lo, hi],
    key: `${path}/${games}/${stepSize}`,
  })
  const errorAxis = useAxis({ label: 'estimate − true skill (error)', range: [-400, 400] })

  const errors = useMemo(() => runs.map((r) => Float64Array.from(r.mean, (v, g) => v - truth[g])), [runs, truth])

  return (
    <Figure
      title="Tracking non-stationary skill: Elo, Glicko, TrueSkill, and smoothing"
      purpose="Compares how online rating algorithms adapt when a competitor's true skill changes over time, contrasting causal online filters (which inevitably lag behind abrupt shifts) with acausal smoothers (which incorporate subsequent match evidence)."
      controls={
        <>
          <ControlRow label="Skill trajectory dynamics">
            <Select
              label="Skill evolution"
              value={path}
              onChange={(v) => setPath(v as 'step' | 'drift' | 'random-walk' | 'constant')}
              options={PATH_OPTIONS}
            />
            <Select
              label="Matchmaking"
              value={opponents}
              onChange={(v) => setOpponents(v as 'close' | 'field')}
              options={OPPONENT_OPTIONS}
            />
            <Slider label="Total games" value={games} onChange={setGames} min={100} max={500} step={50} />
          </ControlRow>
          <ControlRow label="Change parameters">
            {(path === 'step' || path === 'drift') && (
              <Slider
                label="Change point (game)"
                value={changeAt}
                onChange={setChangeAt}
                min={20}
                max={games - 20}
                step={10}
              />
            )}
            {path === 'step' && (
              <Slider
                label="Step size (points)"
                value={stepSize}
                onChange={setStepSize}
                min={-400}
                max={400}
                step={20}
              />
            )}
            {path === 'drift' && (
              <Slider
                label="Drift rate (points/game)"
                value={driftRate}
                onChange={setDriftRate}
                min={0.5}
                max={5}
                step={0.5}
              />
            )}
            {path === 'random-walk' && (
              <Slider label="Walk innovation SD" value={walkSd} onChange={setWalkSd} min={2} max={25} step={1} />
            )}
            {(path === 'step' || path === 'drift') && (
              <Slider
                label="Inactivity break before jump (days)"
                value={pauseDays}
                onChange={setPauseDays}
                min={0}
                max={180}
                step={15}
              />
            )}
          </ControlRow>
          <ControlRow label="Algorithm tuning">
            <Select
              label="Retrospective smoother"
              value={smoother ? 'yes' : 'no'}
              onChange={(v) => setSmoother(v === 'yes')}
              options={[
                { value: 'yes', label: 'Include TrueSkill Through Time' },
                { value: 'no', label: 'Online raters only' },
              ]}
            />
            <Slider label="Elo K-factor" value={kElo} onChange={setKElo} min={8} max={64} step={4} />
            <Slider label="Glicko drift c/day" value={cDrift} onChange={setCDrift} min={0} max={30} step={2} />
            <Slider label="Glicko-2 volatility τ" value={tau2} onChange={setTau2} min={0.2} max={1.2} step={0.1} />
            <Slider label="TrueSkill dynamics τ/day" value={tauTs} onChange={setTauTs} min={5} max={40} step={5} />
          </ControlRow>
        </>
      }
      readouts={Object.fromEntries(
        runs.map((r) => [
          r.name,
          <>
            {path === 'step' && <Readout label="lag (games to 90%)" value={fmt(r.metrics.lag)} />}
            {path === 'step' && <Readout label="overshoot" value={fmt(r.metrics.overshoot)} />}
            {path === 'step' && <Readout label="noise once settled" value={fmt(r.metrics.noise)} />}
            <Readout label={path === 'step' ? 'RMSE before' : 'RMSE'} value={fmt(r.metrics.rmseBefore)} />
            {path === 'step' && <Readout label="RMSE after" value={fmt(r.metrics.rmseAfter)} />}
            <Readout label="±2 SD coverage" value={r.sd ? `${fmt(100 * r.metrics.coverage)}%` : 'point estimate'} />
          </>,
        ]),
      )}
      caption="Tracking non-stationary skill under noisy Bradley–Terry outcomes. Left: true latent skill path (heavy black curve) alongside Elo, Glicko-1, Glicko-2, TrueSkill, and TrueSkill Through Time (with shaded ±2 SD credible intervals where tracked). Right: tracking errors (estimate − true skill). When skill jumps abruptly, online filters lag because they must accumulate evidence game by game; a large learning rate (K, c, or τ) catches up rapidly but amplifies steady-state variance. If an inactivity break precedes the jump, Bayesian models (Glicko and TrueSkill) widen their uncertainty during the lull, enabling a swift initial update upon resumption, whereas Elo maintains a rigid constant K. TrueSkill Through Time uses full retrospective expectation propagation to smooth past the jump symmetrically without lag."
    >
      <Plots cols={2}>
        <Plot x={gameAxis} y={ratingAxis} title="skill estimates over true trajectory">
          {runs.map((r, i) =>
            r.sd ? (
              <Area
                key={`band${i}`}
                name={`${r.name} ± 2 SD`}
                slot={i}
                x={xs}
                y={Float64Array.from(r.mean, (v, g) => v + 2 * r.sd![g])}
                base={Float64Array.from(r.mean, (v, g) => v - 2 * r.sd![g])}
                opacity={0.08}
                line={false}
              />
            ) : null,
          )}
          {runs.map((r, i) => (
            <Curve key={i} name={r.name} slot={i} x={xs} y={r.mean} />
          ))}
          <Curve name="true latent skill" x={xs} y={truth} emphasis />
          {path === 'step' && (
            <Handle
              kind="point"
              at={[change + 1, start + stepSize]}
              onDrag={([x, y]) => {
                setChangeAt(Math.max(20, Math.min(games - 20, Math.round(x - 1))))
                setStepSize(Math.max(-400, Math.min(400, Math.round((y - start) / 20) * 20)))
              }}
              label={`jump at #${change + 1}, ${stepSize > 0 ? '+' : ''}${stepSize}`}
            />
          )}
        </Plot>
        <Plot x={gameAxis} y={errorAxis} title="estimation error (estimate − true skill)">
          <Curve name="zero error" x={[0, games]} y={[0, 0]} muted dashed thin />
          {errors.map((e, i) => (
            <Curve key={i} name={runs[i].name} slot={i} x={xs} y={e} thin />
          ))}
        </Plot>
      </Plots>
    </Figure>
  )
}
