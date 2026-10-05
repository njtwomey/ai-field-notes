/**
 * Rating systems on a simulated league: Elo, Glicko and Glicko-2 ratings chasing known skills game by game; the four
 * systems (with Bradley–Terry refitted on the games so far) compared on predictive log loss and rank agreement; and
 * item response theory's curves and information. Leagues and responses come from `aifn-methods/data/synthetic`; every
 * rating, fit and score from `aifn-methods/inference/rating-models`.
 */
import { useMemo } from 'react'
import { irtResponses, tournament } from 'aifn-methods/data/synthetic'
import {
  eloRatings,
  fitIrt,
  glickoRatings,
  irtProbability,
  itemInformation,
  prequentialBradleyTerry,
  ratingAgreement,
  runningLogLoss,
  type RatingRun,
} from 'aifn-methods/inference/rating-models'
import { stream } from 'aifn-compute/foundation/random'
import { Player, usePlayhead } from 'aifn-render/controls'
import { ControlRow, Figure } from 'aifn-render/layout'
import { choice, float, int, row, slider, useComputed, useFigureState, type AnyValues } from 'aifn-render/state'
import { Area, Curve, formatNumber, Handle, Plot, Plots, Points, Readout, Segments, useAxis } from 'aifn-render/viz'

const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(digits))) : '—')
/** Log-odds skill on Elo's scale. */
const ELO = 400 / Math.LN10
const toElo = (s: Float64Array) => Float64Array.from(s, (v) => 1500 + ELO * v)

const SYSTEMS = [
  { value: 'elo', label: 'Elo' },
  { value: 'glicko', label: 'Glicko' },
  { value: 'glicko2', label: 'Glicko-2' },
] as const
const methodOf = (v: AnyValues) => String(v.method)

/** Column p of a row-major [rows × players] array. */
const column = (a: Float64Array, players: number, p: number) =>
  Float64Array.from({ length: a.length / players }, (_, r) => a[r * players + p])

const league = (label: string) =>
  row(label, {
    players: int(8, { ge: 2, le: 10, suggestions: [4, 6, 8, 10], label: 'players' }),
    games: int(400, { ge: 10, le: 5000, suggestions: [200, 400, 1000, 2000], label: 'games' }),
    spread: slider(0.2, 2, 1, { step: 0.05, label: 'skill spread (log-odds sd)' }),
    drift: slider(0, 0.1, 0, { step: 0.005, label: 'skill drift per game' }),
    seed: int(1, { ge: 0, le: 9999, label: 'seed' }),
  })

// ── 1 · Ratings chase the true skills ────────────────────────────────────────────────────────────────────────────────

export function RatingTrajectorySpecimen() {
  const state = useFigureState({
    league: league('1 · league'),
    system: row('2 · rating system', {
      method: choice(SYSTEMS, 'elo', { label: 'system' }),
      k: float(24, {
        gt: 0,
        le: 400,
        suggestions: [8, 16, 24, 32, 64],
        label: 'K-factor',
        when: (v) => methodOf(v) === 'elo',
      }),
      period: int(10, {
        ge: 1,
        le: 500,
        suggestions: [1, 5, 10, 50],
        label: 'games per rating period',
        when: (v) => methodOf(v) !== 'elo',
      }),
      tau: float(0.5, {
        gt: 0,
        le: 2,
        suggestions: [0.3, 0.5, 1.2],
        label: 'τ (volatility change)',
        when: (v) => methodOf(v) === 'glicko2',
      }),
    }),
  })
  const { players, games, spread, drift, seed } = state.league
  const { method, k, period, tau } = state.system
  const t = useMemo(
    () => tournament(stream(`ratings/${seed}`), { players, games, spread, drift }),
    [players, games, spread, drift, seed],
  )
  const run: RatingRun = useMemo(
    () =>
      method === 'elo'
        ? eloRatings(t.results, { players, k })
        : glickoRatings(t.results, { players, version: method as 'glicko' | 'glicko2', period, tau }),
    [t, method, k, period, tau, players],
  )
  const truth = useMemo(() => toElo(t.skills), [t])
  const agreement = useMemo(() => ratingAgreement(run.history, truth, players), [run, truth, players])
  const loss = useMemo(() => runningLogLoss(t.results, run.predicted), [t, run])
  const [g, setG] = usePlayhead(games + 1)
  const gameAxis = useAxis({ label: 'games played', range: [0, games], key: games })
  const span = Math.max(400, 3 * ELO * spread)
  const ratingAxis = useAxis({ label: 'rating (Elo scale)', range: [1500 - span, 1500 + span], key: `${spread}` })
  const truthAxis = useAxis({ label: 'true skill (Elo scale)', range: [1500 - span, 1500 + span], key: `${spread}` })
  const xs = useMemo(() => Float64Array.from({ length: games + 1 }, (_, i) => i), [games])
  const series = useMemo(
    () =>
      Array.from({ length: players }, (_, p) => ({
        rating: column(run.history, players, p),
        truth: column(truth, players, p),
        sd: run.deviations ? column(run.deviations, players, p) : null,
      })),
    [run, truth, players],
  )
  const now = Array.from({ length: players }, (_, p) => ({
    x: truth[g * players + p],
    y: run.history[g * players + p],
    sd: run.deviations ? run.deviations[g * players + p] : 0,
  }))
  const last = g > 0 ? t.results[g - 1] : null
  return (
    <Figure
      title="Ratings chase the true skills"
      purpose="Each result moves the two players' ratings by the surprise, score minus expected score; Elo's step is a fixed K, while Glicko scales it by the rating deviation, which shrinks as a player's games accumulate."
      state={state}
      defaultSize="L"
      controls={
        <ControlRow label="3 · games">
          <Player className="col-span-full" value={g} onChange={setG} count={games + 1} label="game" />
        </ControlRow>
      }
      readouts={{
        [`after game ${g}`]: (
          <>
            <Readout
              label="last result"
              value={
                last
                  ? `P${last.a + 1} ${last.score === 1 ? 'beat' : last.score === 0 ? 'lost to' : 'drew'} P${last.b + 1}`
                  : '—'
              }
            />
            <Readout label="predicted P(win)" value={last ? fmt(run.predicted[g - 1]) : '—'} />
            <Readout label="Spearman vs truth" value={fmt(agreement[g])} />
            <Readout label="log loss so far" value={g > 0 ? fmt(loss[g - 1]) : '—'} />
          </>
        ),
      }}
      caption="aifn tournament (random pairings, P(a beats b) = σ(s_a − s_b), skills optionally drifting by a random walk) rated by eloRatings or glickoRatings. Left: each player's rating (solid) and true skill on Elo's scale, 1500 + (400/ln 10) s (dashed, same colour); under Glicko a band of ±2 RD. Right: ratings against true skills after the current game, with ±2 RD bars, and the line rating = skill. Play the games, or drag the game marker on the left chart. A large K tracks drift but jitters; a small K settles but lags; Glicko's deviations start at 350 and shrink, so early games move ratings most."
    >
      <Plots cols={2}>
        <Plot x={gameAxis} y={ratingAxis} title="ratings over the season">
          {series.map((s, p) =>
            s.sd ? (
              <Area
                key={`band${p}`}
                name={`P${p + 1}`}
                slot={p}
                x={xs}
                y={Float64Array.from(s.rating, (v, i) => v + 2 * s.sd![i])}
                base={Float64Array.from(s.rating, (v, i) => v - 2 * s.sd![i])}
                opacity={0.08}
                line={false}
              />
            ) : null,
          )}
          {series.map((s, p) => (
            <Curve key={`r${p}`} name={`P${p + 1}`} slot={p} x={xs} y={s.rating} />
          ))}
          {series.map((s, p) => (
            <Curve key={`t${p}`} name={`P${p + 1}`} slot={p} x={xs} y={s.truth} dashed thin />
          ))}
          <Handle kind="x" at={g} onDrag={(v) => setG(Math.round(v))} label={`game ${g}`} />
        </Plot>
        <Plot x={truthAxis} y={ratingAxis} title={`after game ${g}`}>
          <Curve
            name="rating = skill"
            x={[1500 - span, 1500 + span]}
            y={[1500 - span, 1500 + span]}
            emphasis
            dashed
            thin
          />
          {run.deviations && (
            <Segments
              segments={now.map((q) => ({ from: [q.x, q.y - 2 * q.sd] as const, to: [q.x, q.y + 2 * q.sd] as const }))}
            />
          )}
          {now.map((q, p) => (
            <Points key={p} name={`P${p + 1}`} slot={p} x={[q.x]} y={[q.y]} size={8} />
          ))}
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── 2 · Which system predicts best ───────────────────────────────────────────────────────────────────────────────────

const COMPARED = ['Elo', 'Glicko', 'Glicko-2', 'Bradley–Terry (refit)'] as const

export function RatingComparisonSpecimen() {
  const state = useFigureState({
    league: league('1 · league'),
    systems: row('2 · systems', {
      k: float(24, { gt: 0, le: 400, suggestions: [8, 16, 24, 32, 64], label: 'Elo K' }),
      period: int(10, { ge: 1, le: 500, suggestions: [1, 5, 10, 50], label: 'Glicko period (games)' }),
      prior: float(2, { ge: 0, suggestions: [0.1, 0.5, 2, 5], label: 'Bradley–Terry prior α' }),
      refit: int(20, { ge: 1, le: 500, suggestions: [5, 10, 20, 50], label: 'Bradley–Terry refit every' }),
    }),
  })
  const { players, games, spread, drift, seed } = state.league
  const { k, period, refit, prior } = state.systems
  const result = useComputed(
    () => {
      const t = tournament(stream(`ratings/${seed}`), { players, games, spread, drift })
      const truth = toElo(t.skills)
      const runs = [
        eloRatings(t.results, { players, k }),
        glickoRatings(t.results, { players, version: 'glicko', period }),
        glickoRatings(t.results, { players, version: 'glicko2', period }),
      ].map((r) => ({ history: r.history, predicted: r.predicted }))
      runs.push(prequentialBradleyTerry(t.results, { players, refitEvery: refit, prior }))
      return runs.map((r) => ({
        loss: runningLogLoss(t.results, r.predicted),
        agreement: ratingAgreement(r.history, truth, players),
      }))
    },
    [players, games, spread, drift, seed, k, period, refit, prior],
    { mode: 'release' },
  )
  const curves = result.value
  const [g, setG] = usePlayhead(games + 1)
  const gameAxis = useAxis({ label: 'games played', range: [0, games], key: games })
  const lossAxis = useAxis({ label: 'mean log loss so far (nats)', range: [0.3, 0.9] })
  const rankAxis = useAxis({ label: 'Spearman ρ with true skill', range: [-0.5, 1] })
  const xs = useMemo(() => Float64Array.from({ length: games }, (_, i) => i + 1), [games])
  const xs0 = useMemo(() => Float64Array.from({ length: games + 1 }, (_, i) => i), [games])
  return (
    <Figure
      title="Which system predicts the next game"
      purpose="Every system is scored prequentially: each game is predicted from the games before it, so the running log loss measures prediction, and the rank correlation measures how well the ratings order the true skills."
      state={state}
      defaultSize="L"
      controls={
        <ControlRow label="3 · games">
          <Player className="col-span-full" value={g} onChange={setG} count={games + 1} label="game" />
        </ControlRow>
      }
      readouts={{
        [`after game ${g}`]: (
          <>
            {COMPARED.map((name, i) => (
              <Readout
                key={name}
                label={name}
                value={g > 0 ? `${fmt(curves[i].loss[g - 1])} nats, ρ ${fmt(curves[i].agreement[g], 2)}` : '—'}
              />
            ))}
          </>
        ),
      }}
      caption="aifn eloRatings, glickoRatings (Glicko and Glicko-2) and prequentialBradleyTerry (MM with a prior α, refitted every few games, predicting each game from the fit before it) on one simulated league; runningLogLoss and ratingAgreement score them. A coin flip scores ln 2 ≈ 0.69 nats. Bradley–Terry weighs every past game equally: with a small prior its early fits are overconfident and its log loss starts high, and with drifting skills its old games mislead it, while the online systems forget. Play the games, or drag the marker on either chart."
    >
      <Plots cols={2}>
        <Plot x={gameAxis} y={lossAxis} title="predictive log loss">
          <Curve name="coin flip" x={[0, games]} y={[Math.LN2, Math.LN2]} muted dashed thin />
          {curves.map((c, i) => (
            <Curve key={i} name={COMPARED[i]} slot={i} x={xs} y={c.loss} stale={result.stale} />
          ))}
          <Handle kind="x" at={g} onDrag={(v) => setG(Math.round(v))} label={`game ${g}`} />
        </Plot>
        <Plot x={gameAxis} y={rankAxis} title="agreement with the true order">
          {curves.map((c, i) => (
            <Curve key={i} name={COMPARED[i]} slot={i} x={xs0} y={c.agreement} stale={result.stale} />
          ))}
          <Handle kind="x" at={g} onDrag={(v) => setG(Math.round(v))} label={`game ${g}`} />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── 3 · Item response theory ─────────────────────────────────────────────────────────────────────────────────────────

export function ItemResponseSpecimen() {
  const state = useFigureState({
    data: row('1 · test', {
      persons: int(200, { ge: 10, le: 2000, suggestions: [50, 200, 500], label: 'persons' }),
      items: int(8, { ge: 2, le: 40, suggestions: [5, 8, 15, 30], label: 'items' }),
      model: choice(
        [
          { value: '2pl', label: '2PL (a and b per item)' },
          { value: '1pl', label: '1PL (Rasch: a = 1)' },
        ],
        '2pl',
        { label: 'fitted model' },
      ),
      seed: int(2, { ge: 0, le: 9999, label: 'seed' }),
    }),
    probe: row('2 · ability', { theta: slider(-4, 4, 0, { step: 0.05, label: 'ability θ' }) }),
  })
  const { persons, items, model, seed } = state.data
  const { theta } = state.probe
  const fitted = useComputed(
    () => {
      const d = irtResponses(stream(`irt/${seed}`), { persons, items })
      return { d, fit: fitIrt(d.responses, { model: model as '1pl' | '2pl' }) }
    },
    [persons, items, model, seed],
    { mode: 'release' },
  )
  const { d, fit } = fitted.value
  const grid = useMemo(() => Float64Array.from({ length: 161 }, (_, i) => -4 + i * 0.05), [])
  const shown = Math.min(items, 10)
  const curves = useMemo(
    () =>
      Array.from({ length: shown }, (_, i) =>
        Float64Array.from(grid, (th) => irtProbability(th, fit.discrimination[i], fit.difficulty[i])),
      ),
    [fit, grid, shown],
  )
  const information = useMemo(
    () =>
      Float64Array.from(grid, (th) => {
        let s = 0
        for (let i = 0; i < items; i++) s += itemInformation(th, fit.discrimination[i], fit.difficulty[i])
        return s
      }),
    [fit, grid, items],
  )
  let infoAt = 0
  for (let i = 0; i < items; i++) infoAt += itemInformation(theta, fit.discrimination[i], fit.difficulty[i])
  const thetaAxis = useAxis({ label: 'ability θ', range: [-4, 4] })
  const pAxis = useAxis({ label: 'P(correct)', range: [0, 1] })
  const infoAxis = useAxis({ label: 'test information I(θ)', hold: 'union', key: `${items}/${model}/${seed}` })
  const bAxis = useAxis({ label: 'true difficulty b', range: [-3, 3] })
  const bHatAxis = useAxis({ label: 'fitted difficulty b̂', range: [-3, 3] })
  return (
    <Figure
      title="Item response curves and information"
      purpose="An item's response curve P(correct | θ) = σ(a(θ − b)) rises through ½ at its difficulty b with slope set by its discrimination a; its information a²P(1 − P) peaks there, and the test's information I(θ) sets the standard error 1/√I(θ) of an ability estimate."
      state={state}
      defaultSize="L"
      readouts={{
        [`at θ = ${fmt(theta, 2)}`]: (
          <>
            <Readout label="I(θ)" value={fmt(infoAt)} />
            <Readout label="SE(θ̂) = 1/√I" value={fmt(1 / Math.sqrt(infoAt))} />
            <Readout
              label="expected score"
              value={fmt(
                curves.reduce((s, _, i) => s + irtProbability(theta, fit.discrimination[i], fit.difficulty[i]), 0),
              )}
            />
          </>
        ),
        fit: (
          <>
            <Readout label="log-likelihood" value={fmt(fit.logLikelihood, 5)} />
            <Readout label="converged" value={fit.converged ? 'yes' : 'no'} />
          </>
        ),
      }}
      caption="aifn irtResponses draws answers from a 2PL model (θ, b ~ N(0, 1), log a ~ N(0, 0.3²)); fitIrt estimates b and a by marginal maximum likelihood (θ integrated out on a grid; autodiff gradients, L-BFGS), then each θ by its posterior mean. Left: the fitted response curves of the first ten items, with the probe at θ. Middle: the test information, largest where the item difficulties cluster. Right: fitted against true difficulties. Drag θ on either of the first two charts."
    >
      <Plots cols={3}>
        <Plot x={thetaAxis} y={pAxis} title="item response curves">
          {curves.map((c, i) => (
            <Curve key={i} name={`item ${i + 1}`} slot={i} x={grid} y={c} />
          ))}
          <Handle
            kind="x"
            at={theta}
            onDrag={(v) => state.set('probe.theta', Math.max(-4, Math.min(4, v)))}
            label="θ"
          />
        </Plot>
        <Plot x={thetaAxis} y={infoAxis} title="test information">
          <Curve name="I(θ)" x={grid} y={information} emphasis />
          <Handle
            kind="x"
            at={theta}
            onDrag={(v) => state.set('probe.theta', Math.max(-4, Math.min(4, v)))}
            label="θ"
          />
        </Plot>
        <Plot x={bAxis} y={bHatAxis} title="difficulties recovered">
          <Curve name="b̂ = b" x={[-3, 3]} y={[-3, 3]} emphasis dashed thin />
          <Points name="items" x={d.difficulty} y={fit.difficulty} slot={0} />
        </Plot>
      </Plots>
    </Figure>
  )
}
