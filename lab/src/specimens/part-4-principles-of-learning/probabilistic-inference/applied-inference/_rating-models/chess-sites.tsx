/**
 * Chess ratings on three sites: one simulated population (the same true skills, the same games) rated by chess.com's
 * Glicko-1, Lichess's Glicko-2 and FIDE's Elo, each with its published (or assumed) settings. The population and its
 * games come from `aifn-methods/data/synthetic` (`ratingPopulation`); every rating, settled value, scale map and
 * settling count from `aifn-methods/inference/rating-models`.
 */
import { useMemo } from 'react'
import { ratingPopulation } from 'aifn-methods/data/synthetic'
import {
  LICHESS_PROVISIONAL_RD,
  chessComSpec,
  chessComStarts,
  fideSpec,
  lichessSpec,
  rateStream,
  ratingScaleMap,
  settlingGames,
  type RatingTrace,
} from 'aifn-methods/inference/rating-models'
import { child, standardNormals, stream } from 'aifn-compute/foundation/random'
import { Player, usePlayhead } from 'aifn-render/controls'
import { ControlRow, Figure } from 'aifn-render/layout'
import { choice, float, int, pinField, row, slider, useComputed, useFigureState, usePinned } from 'aifn-render/state'
import { Area, Bars, Curve, formatNumber, Handle, Plot, Plots, Points, Readout, useAxis } from 'aifn-render/viz'

const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(digits))) : '—')
const SITES = ['chess.com (Glicko-1)', 'Lichess (Glicko-2)', 'FIDE (Elo)'] as const

const START_CHOICES = [
  { value: 'declared', label: 'self-declared level' },
  { value: '400', label: 'all at 400 (new to chess)' },
  { value: '800', label: 'all at 800 (beginner)' },
  { value: '1200', label: 'all at 1200 (intermediate)' },
  { value: '1600', label: 'all at 1600 (advanced)' },
] as const

const population = () =>
  row('1 · population', {
    players: int(300, { ge: 10, le: 2000, suggestions: [100, 300, 1000], label: 'players' }),
    days: int(120, { ge: 2, le: 1000, suggestions: [30, 120, 365], label: 'days' }),
    perDay: int(2, { ge: 1, le: 20, suggestions: [1, 2, 5], label: 'games per day' }),
    spread: slider(100, 600, 350, { step: 10, label: 'true-skill sd (points)' }),
    matchmaking: choice(
      [
        { value: 'rating', label: 'similar rating' },
        { value: 'random', label: 'random opponents' },
      ],
      'rating',
      { label: 'pairing' },
    ),
    seed: int(1, { ge: 0, le: 9999, label: 'seed' }),
  })

const siteSettings = () =>
  row('2 · sites', {
    start: choice(START_CHOICES, 'declared', { label: 'chess.com start' }),
    c: float(18, { ge: 0, suggestions: [0, 10, 18, 40], label: 'chess.com c per day (assumed)' }),
  })

type Settings = {
  players: number
  days: number
  perDay: number
  spread: number
  matchmaking: string
  seed: number
  start: string
  c: number
}

/** The population and its three ratings. */
function simulate(s: Settings) {
  const pop = ratingPopulation(stream(`chess-sites/${s.seed}`), {
    players: s.players,
    rounds: s.days,
    gamesPerRound: s.perDay,
    spread: s.spread,
    matchmaking: s.matchmaking as 'rating' | 'random',
  })
  const relative = pop.skills.subarray(0, s.players)
  const starts =
    s.start === 'declared'
      ? chessComStarts(relative, standardNormals(child(stream(`chess-sites/${s.seed}`), 'declared'), s.players))
      : Number(s.start)
  const traces: RatingTrace[] = [
    rateStream(pop, { ...chessComSpec(starts), c: s.c }),
    rateStream(pop, lichessSpec()),
    rateStream(pop, fideSpec()),
  ]
  return { pop, relative, traces }
}

function useSimulation(s: Settings) {
  return useComputed(() => simulate(s), [s.players, s.days, s.perDay, s.spread, s.matchmaking, s.seed, s.start, s.c], {
    mode: 'release',
  })
}

/** Column p of a row-major [rows × players] array. */
const column = (a: ArrayLike<number>, players: number, p: number, rows: number) =>
  Float64Array.from({ length: rows }, (_, r) => a[r * players + p])

const EDGES = Array.from({ length: 31 }, (_, i) => i * 100)
const histogram = (values: ArrayLike<number>) => {
  const counts = new Array(EDGES.length - 1).fill(0)
  for (let i = 0; i < values.length; i++) {
    const v = values[i]
    if (!Number.isFinite(v)) continue
    const k = Math.min(EDGES.length - 2, Math.max(0, Math.floor(v / 100)))
    counts[k]++
  }
  return { x: counts.map((_, k) => EDGES[k] + 50), y: counts }
}

const SOURCES =
  'Settings: chess.com uses Glicko (support.chess.com/en/articles/8566476); its sign-up levels start at 400, 800, 1200 or 1600 (as reported in chess.com forums); RD₀ 350, c, the RD floor 30 and the rating floor 100 are not public and are assumed from Glickman’s glicko.pdf (glicko.net/glicko/glicko.pdf). Lichess uses Glicko-2 from 1500 ± 1000 (RD 500), provisional while RD > 110 (lichess.org/faq), RD within [45, 500], volatility 0.09 (≤ 0.1), rating floor 400 and 0.21436 rating periods per day (lila Glicko.scala), τ = 0.75 (lila PR #4034). FIDE: K 40 for the first 30 games, then 20, 10 from 2400; the initial rating after 5 games is Ra + dp with two draws against 1800, at most 2200; ratings below 1400 are not published (handbook.fide.com/chapter/B022024). FIDE’s unrated opponents count at 1500 here, an assumption of a pool where nobody starts rated.'

// ── 1 · The same players on three scales ─────────────────────────────────────────────────────────────────────────────

export function ChessSitesSpecimen() {
  const state = useFigureState({ population: population(), sites: siteSettings(), pin: pinField() })
  const settings: Settings = { ...state.population, ...state.sites }
  const sim = useSimulation(settings)
  const { pop, relative, traces } = sim.value
  const P = pop.players
  const rows = pop.rounds.length + 1
  const [day, setDay] = usePlayhead(rows, 0)
  const pins = usePinned(state.pin, (v) => state.set('pin', v), { valid: (v) => v < P })
  const median = useMemo(() => {
    const order = Array.from({ length: P }, (_, p) => p).sort((a, b) => relative[a] - relative[b])
    return order[Math.floor(P / 2)]
  }, [relative, P])
  const focus = pins.focus ?? median
  const at = (t: RatingTrace) => t.mean.subarray(day * P, (day + 1) * P)
  const now = traces.map(at)
  const maps = [ratingScaleMap(now[0], now[1]), ratingScaleMap(now[0], now[2])]
  const truthMaps = now.map((r) => ratingScaleMap(relative, r))
  const hist = now.map(histogram)
  const series = useMemo(
    () =>
      traces.map((t) => ({
        x: Float64Array.from(column(t.played, P, focus, rows)),
        mean: column(t.mean, P, focus, rows),
        sd: column(t.sd, P, focus, rows),
      })),
    [traces, P, focus, rows],
  )
  const gamesAxis = useAxis({ label: 'games played', range: [0, (rows - 1) * settings.perDay], key: rows })
  const ratingAxis = useAxis({ label: 'rating', range: [0, 3000] })
  // Fixed at a fifth of the pool: the day-0 spikes (everyone at a starting rating) are cut so the spread stays legible.
  const countAxis = useAxis({ label: 'players', range: [0, Math.max(5, Math.ceil(P / 5))], key: P })
  const ccAxis = useAxis({ label: 'chess.com rating', range: [0, 3000] })
  const otherAxis = useAxis({ label: 'Lichess or FIDE rating', range: [0, 3000] })
  const pickNearest = ([x, y]: [number, number]) => {
    let best = -1
    let bestD = Infinity
    for (let p = 0; p < P; p++)
      for (const other of [now[1], now[2]]) {
        const d = (now[0][p] - x) ** 2 + (other[p] - y) ** 2
        if (d < bestD) {
          bestD = d
          best = p
        }
      }
    if (best >= 0 && bestD < 150 ** 2) pins.toggle(best)
    else pins.clear()
  }
  const played = traces[0].played[day * P + focus]
  return (
    <Figure
      title="The same players on three rating scales"
      purpose="One population with fixed true skills plays the same games, rated by chess.com's Glicko-1, Lichess's Glicko-2 and FIDE's Elo; the three scales end up shifted against each other by their starting ratings and update rules, while the order of the players is the same."
      state={state}
      defaultSize="XL"
      controls={
        <ControlRow label="3 · days">
          <Player
            className="col-span-full"
            value={day}
            onChange={setDay}
            count={rows}
            label="day"
            format={(d) => `day ${d}`}
          />
        </ControlRow>
      }
      readouts={{
        [`player ${focus}${pins.pinned !== null ? ' (pinned)' : ' (median skill)'}, day ${day}`]: (
          <>
            <Readout label="true skill − pool mean" value={fmt(relative[focus])} />
            <Readout label="games" value={played} />
            {SITES.map((name, i) => (
              <Readout
                key={name}
                label={Number.isFinite(traces[i].sd[0]) ? `${name} ± 2 RD` : name}
                value={`${fmt(now[i][focus], 4)}${Number.isFinite(traces[i].sd[day * P + focus]) ? ` ± ${fmt(2 * traces[i].sd[day * P + focus], 2)}` : ''}`}
              />
            ))}
          </>
        ),
        [`scales on day ${day}`]: (
          <>
            {SITES.map((name, i) => (
              <Readout
                key={name}
                label={`${name.split(' ')[0]} = a + b·skill`}
                value={`a ${fmt(truthMaps[i].intercept, 4)}, b ${fmt(truthMaps[i].slope, 2)}`}
              />
            ))}
            <Readout label="Lichess − chess.com" value={`${fmt(maps[0].offset, 3)} ± ${fmt(maps[0].spread, 2)}`} />
            <Readout label="FIDE − chess.com" value={`${fmt(maps[1].offset, 3)} ± ${fmt(maps[1].spread, 2)}`} />
          </>
        ),
      }}
      caption={`aifn ratingPopulation: ${settings.players} players with true skills ~ N(0, ${settings.spread}²) relative to the pool, ${settings.perDay} games a day, paired by similar rating (the simulator's own Elo plus noise, so all three sites see identical games) or at random; outcomes from the Bradley–Terry model, P(win) = 1/(1 + 10^(−d/400)). rateStream rates the stream with lichessSpec, chessComSpec and fideSpec. Left: the pinned player's three ratings against the games played, with ±2 RD bands on the Glicko sites (FIDE's starts after five unrated games). Middle: the ratings of everyone on the chosen day, by site. Right: each player's Lichess (and FIDE) rating against their chess.com rating; the dashed line is equality. Click a point on the right to pin that player, Escape to unpin; play the days or drag the day marker on the left. A closed pool keeps the level it started at: chess.com's self-declared starts (average about 1200) and Lichess's 1500 make the same players about 300 points apart, and FIDE's anchors (two draws against 1800) pull its scale up and compress it. Real conversion charts show the same kind of offset, shrinking with strength (ChessGoals' survey of about 20,000 accounts: chess.com blitz 800 ≈ Lichess blitz 1290, 2000 ≈ 2080; chessgoals.com/rating-comparison); real pools differ in who plays where, which no simulation of one population can show. ${SOURCES}`}
    >
      <Plots cols={3}>
        <Plot x={gamesAxis} y={ratingAxis} title={`player ${focus}: start to settled`}>
          {series.map((s, i) =>
            Number.isFinite(s.sd[0]) ? (
              <Area
                key={`band${i}`}
                name={SITES[i]}
                slot={i}
                x={s.x}
                y={Float64Array.from(s.mean, (v, r) => v + 2 * s.sd[r])}
                base={Float64Array.from(s.mean, (v, r) => v - 2 * s.sd[r])}
                opacity={0.1}
                line={false}
              />
            ) : null,
          )}
          {series.map((s, i) => (
            <Curve key={i} name={SITES[i]} slot={i} x={s.x} y={s.mean} stale={sim.stale} />
          ))}
          <Handle kind="x" at={played} onDrag={(v) => setDay(Math.round(v / settings.perDay))} label={`day ${day}`} />
        </Plot>
        <Plot x={ratingAxis} y={countAxis} title={`ratings on day ${day}`}>
          {hist.map((h, i) => (
            <Bars key={i} name={SITES[i]} slot={i} x={h.x} y={h.y} edges={EDGES} opacity={0.4} stale={sim.stale} />
          ))}
        </Plot>
        <Plot x={ccAxis} y={otherAxis} title="the same player on two sites" onPlotClick={pickNearest}>
          <Curve name="equal ratings" x={[0, 3000]} y={[0, 3000]} muted dashed thin />
          <Points name={SITES[1]} slot={1} x={now[0]} y={now[1]} size={4} stale={sim.stale} />
          <Points name={SITES[2]} slot={2} x={now[0]} y={now[2]} size={4} stale={sim.stale} />
          <Curve
            name="Lichess fit"
            slot={1}
            x={[0, 3000]}
            y={[maps[0].intercept, maps[0].intercept + 3000 * maps[0].slope]}
            thin
          />
          <Curve
            name="FIDE fit"
            slot={2}
            x={[0, 3000]}
            y={[maps[1].intercept, maps[1].intercept + 3000 * maps[1].slope]}
            thin
          />
          <Points
            name={`player ${focus}`}
            emphasis
            x={[now[0][focus], now[0][focus]]}
            y={[now[1][focus], now[2][focus]]}
            size={10}
          />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── 2 · How fast new players settle ─────────────────────────────────────────────────────────────────────────────────

const quantile = (sorted: number[], q: number) =>
  sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))] : NaN

export function SettlingSpecimen() {
  const state = useFigureState({
    population: population(),
    sites: siteSettings(),
    settle: row('3 · settling', {
      tolerance: float(50, { gt: 0, suggestions: [25, 50, 100], label: 'settled within ± (points)' }),
      hold: int(10, { ge: 1, suggestions: [1, 5, 10, 30], label: 'for days' }),
    }),
  })
  const settings: Settings = { ...state.population, ...state.sites }
  const { tolerance, hold } = state.settle
  const sim = useSimulation(settings)
  const { pop, traces } = sim.value
  const P = pop.players
  const rows = pop.rounds.length + 1
  const settle = useMemo(
    () =>
      traces.map((t) =>
        Float64Array.from({ length: P }, (_, p) => settlingGames(t, p, { tolerance, hold }))
          .filter(Number.isFinite)
          .sort(),
      ),
    [traces, P, tolerance, hold],
  )
  const sorted = settle.map((s) => Array.from(s))
  // RD across players by day: the median and the 10–90 % band, for the two Glicko sites.
  const rd = useMemo(
    () =>
      traces.slice(0, 2).map((t) => {
        const q = (r: number, f: number) => {
          const v = Array.from(t.sd.subarray(r * P, (r + 1) * P)).sort((a, b) => a - b)
          return v[Math.min(P - 1, Math.floor(f * P))]
        }
        return {
          x: Float64Array.from({ length: rows }, (_, r) => r * settings.perDay),
          median: Float64Array.from({ length: rows }, (_, r) => q(r, 0.5)),
          low: Float64Array.from({ length: rows }, (_, r) => q(r, 0.1)),
          high: Float64Array.from({ length: rows }, (_, r) => q(r, 0.9)),
        }
      }),
    [traces, P, rows, settings.perDay],
  )
  const provisional = (() => {
    const at = rd[1].median.findIndex((v) => v <= LICHESS_PROVISIONAL_RD)
    return at < 0 ? NaN : at * settings.perDay
  })()
  const maxGames = (rows - 1) * settings.perDay
  const binWidth = Math.max(1, Math.ceil(maxGames / 30))
  const edges = Array.from({ length: Math.ceil(maxGames / binWidth) + 2 }, (_, i) => i * binWidth)
  const hists = sorted.map((s) => {
    const counts = new Array(edges.length - 1).fill(0)
    for (const v of s) counts[Math.min(edges.length - 2, Math.floor(v / binWidth))]++
    return { x: counts.map((_, k) => edges[k] + binWidth / 2), y: counts }
  })
  const gamesAxis = useAxis({ label: 'games played', range: [0, maxGames], key: maxGames })
  const rdAxis = useAxis({ label: 'rating deviation (points)', range: [0, 520] })
  const countAxis = useAxis({ label: 'players', hold: 'union', key: `${P}/${settings.seed}/${tolerance}/${hold}` })
  return (
    <Figure
      title="How fast new players settle"
      purpose="A new player's rating deviation starts wide (350 on chess.com, 500 on Lichess) and shrinks with every game, so early games move the rating most; the rating settles once the deviation reaches the floor set by the drift allowed per day."
      state={state}
      defaultSize="L"
      readouts={{
        'games to settle (median, 90th percentile)': (
          <>
            {SITES.map((name, i) => (
              <Readout
                key={name}
                label={name}
                value={`${fmt(quantile(sorted[i], 0.5))}, ${fmt(quantile(sorted[i], 0.9))} (${sorted[i].length} of ${P} settled)`}
              />
            ))}
          </>
        ),
        deviations: (
          <>
            <Readout label="Lichess: median RD ≤ 110 (not provisional) after" value={`${fmt(provisional)} games`} />
            <Readout label="chess.com RD at the end (median)" value={fmt(rd[0].median[rows - 1])} />
            <Readout label="Lichess RD at the end (median)" value={fmt(rd[1].median[rows - 1])} />
          </>
        ),
      }}
      caption={`The same simulation as above (aifn ratingPopulation rated by chessComSpec, lichessSpec and fideSpec). Left: the median rating deviation across players against games played, with the band between the 10th and 90th percentiles, for the two Glicko sites; the dashed lines are Lichess's provisional threshold (RD 110: a "?" next to the rating) and its leaderboard threshold (RD 75). FIDE has no deviation: its K is 40 for a newcomer's first 30 rated games, then 20. Right: settlingGames, the games each player needed before their rating stayed within ±${tolerance} of its settled value (the mean over the last quarter of the days) for ${hold} days running. RD stops shrinking where a game's information balances the drift added per day: chess.com's assumed c = ${settings.c} a day against Lichess's volatility of 0.09 over 0.21 rating periods a day. ${SOURCES}`}
    >
      <Plots cols={2}>
        <Plot x={gamesAxis} y={rdAxis} title="the provisional deviation shrinking">
          {rd.map((s, i) => (
            <Area
              key={`band${i}`}
              name={SITES[i]}
              slot={i}
              x={s.x}
              y={s.high}
              base={s.low}
              opacity={0.12}
              line={false}
            />
          ))}
          {rd.map((s, i) => (
            <Curve key={i} name={SITES[i]} slot={i} x={s.x} y={s.median} stale={sim.stale} />
          ))}
          <Curve name="Lichess provisional (RD 110)" x={[0, maxGames]} y={[110, 110]} muted dashed thin />
          <Curve name="Lichess leaderboard (RD 75)" x={[0, maxGames]} y={[75, 75]} muted dashed thin />
        </Plot>
        <Plot x={gamesAxis} y={countAxis} title={`games to settle within ±${tolerance}`}>
          {hists.map((h, i) => (
            <Bars key={i} name={SITES[i]} slot={i} x={h.x} y={h.y} edges={edges} opacity={0.4} stale={sim.stale} />
          ))}
        </Plot>
      </Plots>
    </Figure>
  )
}
