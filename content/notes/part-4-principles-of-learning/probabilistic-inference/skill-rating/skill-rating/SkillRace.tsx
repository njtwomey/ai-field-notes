import { useMemo, useState } from 'react'
import {
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { TS_DEFAULTS, eloExpectedGaussian, simulateGames, trueSkill1v1, type Rating } from '../_shared/skill'

const NAMES = ['A', 'B', 'C', 'D']
const N_GAMES = 200
/** Performance noise that generates the games. The β slider sets what TrueSkill and Elo assume. */
const TRUE_BETA = TS_DEFAULTS.beta
const MARKER_X = N_GAMES + 8
const X_RANGE: [number, number] = [0, N_GAMES + 14]
const Y_RANGE: [number | undefined, number | undefined] = [5, 45]
const SKILL_MIN = 8
const SKILL_MAX = 42

/** First game after which the estimated order matches the true order for every later game, or null. */
function settledAt(estimates: number[][], truth: number[]): number | null {
  const order = (xs: number[]) => xs.map((_, i) => i).sort((a, b) => xs[b] - xs[a])
  const want = order(truth).join()
  let settled: number | null = null
  for (let g = 0; g < estimates[0].length; g++) {
    const ok = order(estimates.map((e) => e[g])).join() === want
    if (ok && settled === null) settled = g
    if (!ok) settled = null
  }
  return settled
}

/** Pairs of players whose estimates are in the wrong order. */
function misordered(est: number[], truth: number[]): number {
  let n = 0
  for (let i = 0; i < est.length; i++)
    for (let j = i + 1; j < est.length; j++) if ((est[i] - est[j]) * (truth[i] - truth[j]) < 0) n++
  return n
}

export function SkillRace() {
  const [skills, setSkills] = useState([33, 28, 24, 19])
  const state = useFigureState({
    shown: float(N_GAMES, { min: 0, max: N_GAMES, step: 1, label: 'games played', format: (v) => String(v) }),
    K: float(1.5, { min: 0.1, max: 6, step: 0.1, label: 'Elo K (skill units per game)' }),
    beta: float(TS_DEFAULTS.beta, { min: 1, max: 12, step: 0.1, label: 'assumed performance noise β' }),
    tau: float(TS_DEFAULTS.tau, { min: 0, max: 2, step: 0.01, label: 'TrueSkill dynamics τ' }),
    seed: int(3, { min: 1, max: 30, step: 1, label: 'seed', format: (v) => String(v) }),
  })

  const run = useMemo(() => {
    const games = simulateGames(skills, N_GAMES, TRUE_BETA, state.seed)
    const m = skills.length
    const elo = Array.from({ length: m }, () => [TS_DEFAULTS.mu])
    const mu = Array.from({ length: m }, () => [TS_DEFAULTS.mu])
    const sd = Array.from({ length: m }, () => [TS_DEFAULTS.sigma])
    const r = Array(m).fill(TS_DEFAULTS.mu)
    const ts: Rating[] = Array.from({ length: m }, () => ({ mu: TS_DEFAULTS.mu, sigma: TS_DEFAULTS.sigma }))
    for (const { i, j, y } of games) {
      const step = state.K * (y - eloExpectedGaussian(r[i], r[j], state.beta))
      r[i] += step
      r[j] -= step
      const u = trueSkill1v1(ts[i], ts[j], y ? 'win' : 'loss', { beta: state.beta, tau: state.tau, eps: 0 })
      ts[i] = u.p1
      ts[j] = u.p2
      for (let p = 0; p < m; p++) {
        elo[p].push(r[p])
        mu[p].push(ts[p].mu)
        sd[p].push(ts[p].sigma)
      }
    }
    return { elo, mu, sd, eloSettled: settledAt(elo, skills), tsSettled: settledAt(mu, skills) }
  }, [skills, state.K, state.beta, state.tau, state.seed])

  const g = state.shown
  const xs = useMemo(() => Array.from({ length: g + 1 }, (_, k) => k), [g])

  // One grouped scatter: each true skill takes its player's colour and a distinct marker shape.
  const truthMarkers = useMemo<SeriesSpec[]>(
    () => [
      {
        name: 'true skill',
        type: 'scatter',
        x: NAMES.map(() => MARKER_X),
        y: skills,
        group: NAMES.map((_, p) => p),
        groupNames: NAMES,
      },
    ],
    [skills],
  )

  const eloSeries = useMemo<SeriesSpec[]>(
    () => [
      ...NAMES.map<SeriesSpec>((name, p) => ({ name, type: 'line', x: xs, y: run.elo[p].slice(0, g + 1), slot: p })),
      // True skills as dashed levels across the whole chart; each is also a drag handle.
      ...NAMES.map<SeriesSpec>((name, p) => ({
        name,
        type: 'line',
        x: [0, N_GAMES],
        y: [skills[p], skills[p]],
        slot: p,
        dashed: true,
      })),
    ],
    [run, xs, g, skills],
  )
  const tsSeries = useMemo<SeriesSpec[]>(
    () => [
      ...NAMES.flatMap<SeriesSpec>((name, p) => {
        const m = run.mu[p].slice(0, g + 1)
        const s = run.sd[p].slice(0, g + 1)
        return [
          { name, type: 'line', x: xs, y: m, slot: p },
          { name, type: 'line', x: xs, y: m.map((v, k) => v + s[k]), slot: p, dashed: true },
          { name, type: 'line', x: xs, y: m.map((v, k) => v - s[k]), slot: p, dashed: true },
        ]
      }),
      ...truthMarkers,
    ],
    [run, xs, g, truthMarkers],
  )

  const handles: Handle[] = NAMES.map((name, p) => ({
    kind: 'y',
    at: skills[p],
    label: name,
    onDrag: (y) =>
      setSkills((prev) => prev.map((v, q) => (q === p ? Math.round(Math.min(SKILL_MAX, Math.max(SKILL_MIN, y))) : v))),
  }))

  const eloNow = run.elo.map((e) => e[g])
  const muNow = run.mu.map((e) => e[g])
  const games = (s: number | null) => (s === null ? `not within ${N_GAMES}` : String(s))
  // One unit of skill in logistic Elo points: Φ(x/(√2β)) ≈ σ(1.702 x/(√2β)) and Elo's logistic scale is 400/ln 10.
  const eloPoints = (state.K * (400 / Math.LN10) * 1.702) / (Math.SQRT2 * state.beta)

  const xAxis = useAxis({ label: 'game', range: X_RANGE })
  const yAxis = useAxis({ label: 'rating', range: Y_RANGE })
  const xAxis2 = useAxis({ label: 'game', range: X_RANGE })
  const yAxis2 = useAxis({ label: 'skill', range: Y_RANGE })
  return (
    <Figure
      title="Elo against TrueSkill on the same match stream"
      state={state}
      caption="Four players with fixed true skills (the dashed levels in the top chart, which you can drag up and down, and the markers on the right of the bottom chart) play random pairings. Each game is decided by the Thurstone model with performance noise 25/6. Elo (top) keeps one number per player and moves it by K times the surprise. TrueSkill (bottom) keeps a mean and a standard deviation; the dashed lines are μ ± σ. Early on, TrueSkill's large σ makes big steps, and the steps shrink as σ narrows. A large K makes Elo fast but noisy; a small K makes it smooth but slow. τ stops σ from shrinking to zero."

      readouts={
        <>
          <Readout label="Elo: order right from game" value={games(run.eloSettled)} />
          <Readout label="TrueSkill: order right from game" value={games(run.tsSettled)} />
          <Readout
            label="misordered pairs now, Elo / TrueSkill"
            value={`${misordered(eloNow, skills)} / ${misordered(muNow, skills)}`}
          />
          <Readout label="K in logistic Elo points" value={formatNumber(eloPoints)} />
        </>
      }
    >
      <div className="space-y-2">
        <p className="text-xs text-muted-foreground">Elo rating</p>
        <Plot x={xAxis} y={yAxis} height={240}>
          {seriesLayers(eloSeries)}
          {(handles ?? []).map((h, i) => (
            <Handle key={i} {...h} />
          ))}
        </Plot>
        <p className="text-xs text-muted-foreground">TrueSkill mean μ with μ ± σ</p>
        <Plot x={xAxis2} y={yAxis2} height={280}>
          {seriesLayers(tsSeries)}
        </Plot>
      </div>
    </Figure>
  )
}
