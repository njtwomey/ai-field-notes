import { useMemo } from 'react'
import {
  choice,
  Figure,
  formatNumber,
  int,
  Plot,
  Readout,
  type Segment,
  Segments,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { TS_DEFAULTS, trueSkill1v1, type Rating } from '../_shared/skill'

const NAMES = ['A', 'B', 'C', 'D', 'E', 'F']
/** Winner, loser. A beats B, C beats D, E beats F, then B beats C and D beats E. */
const PLAYED: [number, number][] = [
  [0, 1],
  [2, 3],
  [4, 5],
  [1, 2],
  [3, 4],
]
const OPTS = { beta: TS_DEFAULTS.beta, tau: 0, eps: 0 }
const PRIOR = { prec: 1 / TS_DEFAULTS.sigma ** 2, h: TS_DEFAULTS.mu / TS_DEFAULTS.sigma ** 2 }

/** One forward pass: each game updates the current beliefs once (Gaussian density filtering). */
function online(order: [number, number][]): Rating[] {
  const r: Rating[] = NAMES.map(() => ({ mu: TS_DEFAULTS.mu, sigma: TS_DEFAULTS.sigma }))
  for (const [w, l] of order) {
    const u = trueSkill1v1(r[w], r[l], 'win', OPTS)
    r[w] = u.p1
    r[l] = u.p2
  }
  return r
}

/**
 * Expectation propagation over all games with static skills. Each game keeps a Gaussian message to each of its two
 * players (natural parameters). A visit removes the game's old messages (the cavity), redoes the two-player update
 * against the cavity, and stores the new messages. The first sweep equals the online pass.
 */
function ep(order: [number, number][], sweeps: number): Rating[] {
  const msg = order.map(() => [
    { prec: 0, h: 0 },
    { prec: 0, h: 0 },
  ])
  const marginal = () => {
    const m = NAMES.map(() => ({ ...PRIOR }))
    order.forEach(([w, l], k) => {
      m[w].prec += msg[k][0].prec
      m[w].h += msg[k][0].h
      m[l].prec += msg[k][1].prec
      m[l].h += msg[k][1].h
    })
    return m
  }
  for (let s = 0; s < sweeps; s++) {
    order.forEach(([w, l], k) => {
      const m = marginal()
      const cw = { prec: m[w].prec - msg[k][0].prec, h: m[w].h - msg[k][0].h }
      const cl = { prec: m[l].prec - msg[k][1].prec, h: m[l].h - msg[k][1].h }
      const u = trueSkill1v1(
        { mu: cw.h / cw.prec, sigma: 1 / Math.sqrt(cw.prec) },
        { mu: cl.h / cl.prec, sigma: 1 / Math.sqrt(cl.prec) },
        'win',
        OPTS,
      )
      const pw = 1 / u.p1.sigma ** 2
      const pl = 1 / u.p2.sigma ** 2
      msg[k][0] = { prec: pw - cw.prec, h: u.p1.mu * pw - cw.h }
      msg[k][1] = { prec: pl - cl.prec, h: u.p2.mu * pl - cl.h }
    })
  }
  return marginal().map((m) => ({ mu: m.h / m.prec, sigma: 1 / Math.sqrt(m.prec) }))
}

const ranking = (r: Rating[]) =>
  r
    .map((x, i) => ({ x, i }))
    .sort((a, b) => b.x.mu - a.x.mu)
    .map(({ x, i }, k, all) => (k > 0 && Math.abs(all[k - 1].x.mu - x.mu) < 1e-6 ? '= ' : k > 0 ? '> ' : '') + NAMES[i])
    .join(' ')

export function OnlineVsSmoothed() {
  const state = useFigureState({
    sweeps: int(1, { min: 1, max: 8, step: 1, label: 'EP sweeps', format: (v) => String(v) }),
    order: choice<'played' | 'reversed'>(
      [
        { value: 'played', label: 'as played' },
        { value: 'reversed', label: 'reversed' },
      ],
      'played',
      { label: 'game order' },
    ),
  })

  const res = useMemo(() => {
    const games = state.order === 'played' ? PLAYED : [...PLAYED].reverse()
    return { online: online(games), ep: ep(games, state.sweeps) }
  }, [state.order, state.sweeps])

  const { series, segments } = useMemo(() => {
    const xs = NAMES.map((_, i) => i + 1)
    const mk = (name: string, r: Rating[], dx: number, slot: number): SeriesSpec => ({
      name,
      type: 'scatter',
      x: xs.map((x) => x + dx),
      y: r.map((v) => v.mu),
      slot,
    })
    const bars = (r: Rating[], dx: number): Segment[] =>
      r.map((v, i) => ({ from: [i + 1 + dx, v.mu - v.sigma], to: [i + 1 + dx, v.mu + v.sigma] }))
    return {
      series: [
        mk('online, one pass', res.online, -0.12, 0),
        mk(`EP, ${state.sweeps} sweep${state.sweeps > 1 ? 's' : ''}`, res.ep, 0.12, 1),
      ],
      segments: [...bars(res.online, -0.12), ...bars(res.ep, 0.12)],
    }
  }, [res, state.sweeps])

  const xAxis = useAxis({ label: 'player (A = 1, …, F = 6)', range: [0.5, 6.5] })
  const yAxis = useAxis({ label: 'skill μ ± σ', range: [10, 40] })
  return (
    <Figure
      title="Filtering against smoothing on five games"
      state={state}
      caption="Six players start with the same belief. The games are A beats B, C beats D, E beats F, then B beats C and D beats E. One online pass leaves A, C and E tied, and B and D tied, because each update sees only the past. Step through the EP sweeps: repeated sweeps let the later games revise the earlier ones: B's win over C shows that A's win was against a good player. Skills are held fixed here, as within one year of TrueSkill Through Time. Bars are μ ± σ."

      readouts={
        <>
          <Readout label="online order" value={ranking(res.online)} />
          <Readout label="EP order" value={ranking(res.ep)} />
          <Readout
            label="μ of A, online / EP"
            value={`${formatNumber(res.online[0].mu)} / ${formatNumber(res.ep[0].mu)}`}
          />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        {seriesLayers(series)}
        <Segments segments={segments} />
      </Plot>
    </Figure>
  )
}
