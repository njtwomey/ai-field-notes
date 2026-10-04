import { useMemo, useState } from 'react'
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
import { child, standardNormals, stream } from 'aifn/foundation/random'
import {
  Area,
  Bars,
  ControlRow,
  Curve,
  Figure,
  Handle,
  Player,
  Plot,
  Plots,
  Points,
  Readout,
  Select,
  Slider,
  formatNumber,
  useAxis,
} from 'aifn-render'

const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(digits))) : '—')

const SITES = ['chess.com (Glicko-1)', 'Lichess (Glicko-2)', 'FIDE (Elo)'] as const

const START_OPTIONS = [
  { value: 'declared', label: 'self-declared tier (400, 800, 1200, 1600)' },
  { value: '400', label: 'all at 400 (newcomer)' },
  { value: '800', label: 'all at 800 (beginner)' },
  { value: '1200', label: 'all at 1200 (intermediate)' },
  { value: '1600', label: 'all at 1600 (advanced)' },
]

const PAIRING_OPTIONS = [
  { value: 'rating', label: 'similar rating' },
  { value: 'random', label: 'random opponents' },
]

const VIEW_OPTIONS = [
  { value: 'trajectories', label: 'Rating trajectories & scale offsets' },
  { value: 'settling', label: 'Provisional RD decay & settling times' },
]

const EDGES = Array.from({ length: 31 }, (_, i) => i * 100)
function histogram(values: ArrayLike<number>) {
  const counts = new Array(EDGES.length - 1).fill(0)
  for (let i = 0; i < values.length; i++) {
    const v = values[i]
    if (!Number.isFinite(v)) continue
    const k = Math.min(EDGES.length - 2, Math.max(0, Math.floor(v / 100)))
    counts[k]++
  }
  return { x: counts.map((_, k) => EDGES[k] + 50), y: counts }
}

const column = (a: ArrayLike<number>, players: number, p: number, rows: number) =>
  Float64Array.from({ length: rows }, (_, r) => a[r * players + p])

const quantile = (sorted: number[], q: number) =>
  sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))] : NaN

export function ChessSitesExplorer() {
  const [view, setView] = useState<'trajectories' | 'settling'>('trajectories')
  const players = 250
  const days = 120
  const perDay = 2
  const [spread, setSpread] = useState(350)
  const [matchmaking, setMatchmaking] = useState<'rating' | 'random'>('rating')
  const [start, setStart] = useState('declared')
  const [cDrift, setCDrift] = useState(18)
  const [day, setDay] = useState(60)
  const [pinned, setPinned] = useState<number | null>(null)
  const [tolerance, setTolerance] = useState(50)
  const [hold, setHold] = useState(10)

  // Simulation: fixed seed 1 for reproducible comparison
  const sim = useMemo(() => {
    const seed = 1
    const pop = ratingPopulation(stream(`chess-sites/${seed}`), {
      players,
      rounds: days,
      gamesPerRound: perDay,
      spread,
      matchmaking,
    })
    const relative = pop.skills.subarray(0, players)
    const starts =
      start === 'declared'
        ? chessComStarts(relative, standardNormals(child(stream(`chess-sites/${seed}`), 'declared'), players))
        : Number(start)
    const traces: RatingTrace[] = [
      rateStream(pop, { ...chessComSpec(starts), c: cDrift }),
      rateStream(pop, lichessSpec()),
      rateStream(pop, fideSpec()),
    ]
    return { pop, relative, traces }
  }, [players, days, perDay, spread, matchmaking, start, cDrift])

  const { pop, relative, traces } = sim
  const P = pop.players
  const rows = pop.rounds.length + 1
  const effectiveDay = Math.min(day, rows - 1)

  const medianPlayer = useMemo(() => {
    const order = Array.from({ length: P }, (_, p) => p).sort((a, b) => relative[a] - relative[b])
    return order[Math.floor(P / 2)]
  }, [relative, P])

  const focus = pinned !== null && pinned < P ? pinned : medianPlayer

  const atDay = (t: RatingTrace) => t.mean.subarray(effectiveDay * P, (effectiveDay + 1) * P)
  const now = useMemo(() => traces.map(atDay), [traces, effectiveDay, P])
  const maps = useMemo(() => [ratingScaleMap(now[0], now[1]), ratingScaleMap(now[0], now[2])], [now])
  const truthMaps = useMemo(() => now.map((r) => ratingScaleMap(relative, r)), [now, relative])
  const hist = useMemo(() => now.map(histogram), [now])

  const series = useMemo(
    () =>
      traces.map((t) => ({
        x: Float64Array.from(column(t.played, P, focus, rows)),
        mean: column(t.mean, P, focus, rows),
        sd: column(t.sd, P, focus, rows),
      })),
    [traces, P, focus, rows],
  )

  const played = traces[0].played[effectiveDay * P + focus]

  // Settling statistics
  const settle = useMemo(
    () =>
      traces.map((t) =>
        Float64Array.from({ length: P }, (_, p) => settlingGames(t, p, { tolerance, hold }))
          .filter(Number.isFinite)
          .sort(),
      ),
    [traces, P, tolerance, hold],
  )
  const sortedSettle = useMemo(() => settle.map((s) => Array.from(s)), [settle])

  // Rating deviations over days
  const rd = useMemo(
    () =>
      traces.slice(0, 2).map((t) => {
        const q = (r: number, f: number) => {
          const v = Array.from(t.sd.subarray(r * P, (r + 1) * P)).sort((a, b) => a - b)
          return v[Math.min(P - 1, Math.floor(f * P))]
        }
        return {
          x: Float64Array.from({ length: rows }, (_, r) => r * perDay),
          median: Float64Array.from({ length: rows }, (_, r) => q(r, 0.5)),
          low: Float64Array.from({ length: rows }, (_, r) => q(r, 0.1)),
          high: Float64Array.from({ length: rows }, (_, r) => q(r, 0.9)),
        }
      }),
    [traces, P, rows, perDay],
  )

  const provisionalReached = useMemo(() => {
    const atIdx = rd[1].median.findIndex((v) => v <= LICHESS_PROVISIONAL_RD)
    return atIdx < 0 ? NaN : atIdx * perDay
  }, [rd, perDay])

  const maxGames = (rows - 1) * perDay
  const binWidth = Math.max(1, Math.ceil(maxGames / 25))
  const settleEdges = useMemo(
    () => Array.from({ length: Math.ceil(maxGames / binWidth) + 2 }, (_, i) => i * binWidth),
    [maxGames, binWidth],
  )
  const settleHists = useMemo(
    () =>
      sortedSettle.map((s) => {
        const counts = new Array(settleEdges.length - 1).fill(0)
        for (const v of s) counts[Math.min(settleEdges.length - 2, Math.floor(v / binWidth))]++
        return { x: counts.map((_, k) => settleEdges[k] + binWidth / 2), y: counts }
      }),
    [sortedSettle, settleEdges, binWidth],
  )

  // Axes
  const gamesAxis = useAxis({ label: 'games played', range: [0, (rows - 1) * perDay], key: `${rows}-${perDay}` })
  const ratingAxis = useAxis({ label: 'rating', range: [200, 2800] })
  const countAxis = useAxis({ label: 'players', range: [0, Math.max(5, Math.ceil(P / 4.5))], key: P })
  const ccAxis = useAxis({ label: 'chess.com rating (Glicko-1)', range: [200, 2800] })
  const otherAxis = useAxis({ label: 'Lichess / FIDE rating', range: [200, 2800] })

  const settleGamesAxis = useAxis({ label: 'games played', range: [0, maxGames], key: maxGames })
  const rdAxis = useAxis({ label: 'rating deviation RD (points)', range: [0, 520] })
  const settleCountAxis = useAxis({ label: 'players', hold: 'union', key: `${P}/${tolerance}/${hold}` })

  const pickNearest = ([x, y]: [number, number]) => {
    let best = -1
    let bestD = Infinity
    for (let p = 0; p < P; p++) {
      for (const other of [now[1], now[2]]) {
        const d = (now[0][p] - x) ** 2 + (other[p] - y) ** 2
        if (d < bestD) {
          bestD = d
          best = p
        }
      }
    }
    if (best >= 0 && bestD < 180 ** 2) {
      setPinned((prev) => (prev === best ? null : best))
    } else {
      setPinned(null)
    }
  }

  return (
    <Figure
      title="Chess ratings on three systems: Glicko-1, Glicko-2, and FIDE Elo"
      purpose="Compares how chess.com (Glicko-1), Lichess (Glicko-2), and FIDE (Elo) rate the same pool of players playing identical games, highlighting how initial uncertainty, volatility, and drift scale and offset the resulting ratings."
      controls={
        <>
          <ControlRow label="Mode & view">
            <Select
              label="Explorer perspective"
              value={view}
              onChange={(v) => setView(v as 'trajectories' | 'settling')}
              options={VIEW_OPTIONS}
            />
            <Select label="chess.com initial rating" value={start} onChange={setStart} options={START_OPTIONS} />
            <Select
              label="Matchmaking"
              value={matchmaking}
              onChange={(v) => setMatchmaking(v as 'rating' | 'random')}
              options={PAIRING_OPTIONS}
            />
          </ControlRow>
          <ControlRow label="Simulation parameters">
            <Slider
              label="True skill spread (points)"
              value={spread}
              onChange={setSpread}
              min={150}
              max={550}
              step={25}
            />
            <Slider label="chess.com drift c per day" value={cDrift} onChange={setCDrift} min={0} max={40} step={2} />
            {view === 'trajectories' && (
              <Player
                className="col-span-full"
                value={effectiveDay}
                onChange={setDay}
                count={rows}
                label="simulation day"
                format={(d) => `day ${d}`}
              />
            )}
            {view === 'settling' && (
              <>
                <Slider
                  label="Settling tolerance ± (points)"
                  value={tolerance}
                  onChange={setTolerance}
                  min={25}
                  max={100}
                  step={5}
                />
                <Slider label="Stability hold (days)" value={hold} onChange={setHold} min={2} max={30} step={2} />
              </>
            )}
          </ControlRow>
        </>
      }
      readouts={
        view === 'trajectories'
          ? {
              [`player ${focus}${pinned !== null ? ' (pinned)' : ' (median skill)'}, day ${effectiveDay}`]: (
                <>
                  <Readout label="true skill − pool mean" value={fmt(relative[focus])} />
                  <Readout label="games played" value={played} />
                  {SITES.map((name, i) => (
                    <Readout
                      key={name}
                      label={Number.isFinite(traces[i].sd[0]) ? `${name} ± 2 RD` : name}
                      value={`${fmt(now[i][focus], 4)}${
                        Number.isFinite(traces[i].sd[effectiveDay * P + focus])
                          ? ` ± ${fmt(2 * traces[i].sd[effectiveDay * P + focus], 2)}`
                          : ''
                      }`}
                    />
                  ))}
                </>
              ),
              [`pool conversion on day ${effectiveDay}`]: (
                <>
                  <Readout
                    label="Lichess − chess.com"
                    value={`${fmt(maps[0].offset, 3)} ± ${fmt(maps[0].spread, 2)}`}
                  />
                  <Readout label="FIDE − chess.com" value={`${fmt(maps[1].offset, 3)} ± ${fmt(maps[1].spread, 2)}`} />
                  <Readout
                    label="chess.com fit"
                    value={`a ${fmt(truthMaps[0].intercept, 4)}, b ${fmt(truthMaps[0].slope, 2)}`}
                  />
                  <Readout
                    label="Lichess fit"
                    value={`a ${fmt(truthMaps[1].intercept, 4)}, b ${fmt(truthMaps[1].slope, 2)}`}
                  />
                </>
              ),
            }
          : {
              'games to settle (median, 90th percentile)': (
                <>
                  {SITES.map((name, i) => (
                    <Readout
                      key={name}
                      label={name}
                      value={`${fmt(quantile(sortedSettle[i], 0.5))}, ${fmt(quantile(sortedSettle[i], 0.9))} (${
                        sortedSettle[i].length
                      }/${P} settled)`}
                    />
                  ))}
                </>
              ),
              'uncertainty decay & thresholds': (
                <>
                  <Readout label="Lichess: median RD ≤ 110 at" value={`${fmt(provisionalReached)} games`} />
                  <Readout label="chess.com median RD at end" value={fmt(rd[0].median[rows - 1])} />
                  <Readout label="Lichess median RD at end" value={fmt(rd[1].median[rows - 1])} />
                </>
              ),
            }
      }
      caption={
        view === 'trajectories'
          ? `One population of ${players} players with latent skills ~ N(0, ${spread}²) plays ${perDay} games daily under Bradley–Terry probabilities. Left: the focal player's rating trajectory with ±2 RD credible intervals for Glicko-1 and Glicko-2; FIDE Elo enters after 5 unrated games. Middle: rating distributions across the pool on day ${effectiveDay}. Right: player ratings on Lichess and FIDE plotted against chess.com, with linear calibration lines. Because Lichess anchors newcomers at 1500 ± 500 whereas chess.com uses self-declared starting tiers (average ~1200), ratings remain shifted by ~250–300 points while ranking order is preserved.`
          : `Uncertainty dynamics: Left: rating deviation (RD) decay as games accumulate. Chess.com starts at RD 350 and drifts by c = ${cDrift}/day; Lichess starts at RD 500 and updates volatility τ = 0.75 across rating periods. The dashed line at RD 110 marks the provisional cutoff on Lichess. Right: histogram of games required before each player's rating settles within ±${tolerance} points of its true asymptote for ${hold} consecutive days.`
      }
    >
      {view === 'trajectories' ? (
        <Plots cols={3}>
          <Plot x={gamesAxis} y={ratingAxis} title={`player ${focus}: trajectory & RD bands`}>
            {series.map((s, i) =>
              Number.isFinite(s.sd[0]) ? (
                <Area
                  key={`band${i}`}
                  name={`${SITES[i]} ± 2 RD`}
                  slot={i}
                  x={s.x}
                  y={Float64Array.from(s.mean, (v, r) => v + 2 * s.sd[r])}
                  base={Float64Array.from(s.mean, (v, r) => v - 2 * s.sd[r])}
                  opacity={0.12}
                  line={false}
                />
              ) : null,
            )}
            {series.map((s, i) => (
              <Curve key={i} name={SITES[i]} slot={i} x={s.x} y={s.mean} />
            ))}
            <Handle kind="x" at={played} onDrag={(v) => setDay(Math.round(v / perDay))} label={`day ${effectiveDay}`} />
          </Plot>
          <Plot x={ratingAxis} y={countAxis} title={`ratings on day ${effectiveDay}`}>
            {hist.map((h, i) => (
              <Bars key={i} name={SITES[i]} slot={i} x={h.x} y={h.y} edges={EDGES} opacity={0.4} />
            ))}
          </Plot>
          <Plot x={ccAxis} y={otherAxis} title="ratings compared against chess.com" onPlotClick={pickNearest}>
            <Curve name="equal ratings (y = x)" x={[200, 2800]} y={[200, 2800]} muted dashed thin />
            <Points name={SITES[1]} slot={1} x={now[0]} y={now[1]} size={4} />
            <Points name={SITES[2]} slot={2} x={now[0]} y={now[2]} size={4} />
            <Curve
              name="Lichess linear fit"
              slot={1}
              x={[200, 2800]}
              y={[maps[0].intercept + 200 * maps[0].slope, maps[0].intercept + 2800 * maps[0].slope]}
              thin
            />
            <Curve
              name="FIDE linear fit"
              slot={2}
              x={[200, 2800]}
              y={[maps[1].intercept + 200 * maps[1].slope, maps[1].intercept + 2800 * maps[1].slope]}
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
      ) : (
        <Plots cols={2}>
          <Plot x={settleGamesAxis} y={rdAxis} title="rating deviation RD shrinking over games">
            {rd.map((s, i) => (
              <Area
                key={`band${i}`}
                name={`${SITES[i]} (10th–90th %)`}
                slot={i}
                x={s.x}
                y={s.high}
                base={s.low}
                opacity={0.14}
                line={false}
              />
            ))}
            {rd.map((s, i) => (
              <Curve key={i} name={`${SITES[i]} median`} slot={i} x={s.x} y={s.median} />
            ))}
            <Curve name="Lichess provisional (RD 110)" x={[0, maxGames]} y={[110, 110]} muted dashed thin />
            <Curve name="Lichess established (RD 75)" x={[0, maxGames]} y={[75, 75]} muted dashed thin />
          </Plot>
          <Plot x={settleGamesAxis} y={settleCountAxis} title={`games to settle within ±${tolerance} points`}>
            {settleHists.map((h, i) => (
              <Bars key={i} name={SITES[i]} slot={i} x={h.x} y={h.y} edges={settleEdges} opacity={0.4} />
            ))}
          </Plot>
        </Plots>
      )}
    </Figure>
  )
}
