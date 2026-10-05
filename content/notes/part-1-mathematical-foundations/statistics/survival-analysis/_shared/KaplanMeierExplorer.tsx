import { useMemo } from 'react'
import {
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Readout,
  type Segment,
  Segments,
  seriesLayers,
  type SeriesSpec,
  setting,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { eventTable, logRank, median, stepPath } from './survival'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { stream, uniform } from 'aifn-compute/foundation/random'
import { normalCdf } from 'aifn-compute/numerics/special'

const N = 60
const T_MAX = 24
/** Control-group hazard per month: median survival log 2 / 0.08 ≈ 8.7 months. */
const BASE_HAZARD = 0.08
const GRID = toFlat(linspace(0, T_MAX, 97))

type Group = { t: number[]; d: number[] }

export function KaplanMeierExplorer() {
  const state = useFigureState({
    censorRate: float(0.05, { min: 0, max: 0.2, step: 0.01, label: 'dropout rate per month' }),
    hr: slider(0.3, 1.5, 0.6, { step: 0.05, label: 'hazard ratio' }),
    showNaive: setting(true, 'show drop-censored estimate'),
    seed: int(3, { ge: 0, label: 'seed' }),
  })

  const groups = useMemo(() => {
    const r = stream(state.seed)
    const draw = (hazard: number): Group => {
      const t: number[] = []
      const d: number[] = []
      for (let i = 0; i < N; i++) {
        const event = -Math.log(Math.max(uniform(r), 1e-12)) / hazard
        // Random dropout (exponential with the chosen rate) and administrative censoring at the end of the study.
        // The uniform is drawn even at rate 0, so moving the slider changes censoring but never the event times.
        const u = Math.max(uniform(r), 1e-12)
        const dropout = state.censorRate > 0 ? -Math.log(u) / state.censorRate : Infinity
        const c = Math.min(dropout, T_MAX)
        t.push(Math.min(event, c))
        d.push(event <= c ? 1 : 0)
      }
      return { t, d }
    }
    return [draw(BASE_HAZARD), draw(BASE_HAZARD * state.hr)] as const
  }, [state.seed, state.censorRate, state.hr])

  const km = groups.map((g) => eventTable(g.t, g.d).map((s) => ({ time: s.time, value: s.surv })))
  // "Drop the censored rows": the empirical survival curve of the observed event times alone.
  const naive = groups.map((g) => {
    const ev = g.t.filter((_, i) => g.d[i] === 1).sort((a, b) => a - b)
    return ev.map((time, i) => ({ time, value: 1 - (i + 1) / ev.length }))
  })
  const all = { t: [...groups[0].t, ...groups[1].t], d: [...groups[0].d, ...groups[1].d] }
  const test = logRank(all.t, all.d, [...Array(N).fill(0), ...Array(N).fill(1)])
  const p = 2 * (1 - normalCdf(Math.sqrt(test.chi2)))
  const censored = 1 - all.d.reduce((a, b) => a + b, 0) / (2 * N)
  const names = ['control', 'treated']

  const series: SeriesSpec[] = [0, 1].flatMap((j) => [
    { name: `Kaplan–Meier, ${names[j]}`, type: 'line' as const, ...stepPath(km[j], T_MAX), slot: j },
    ...(state.showNaive
      ? [
          {
            name: `drop censored, ${names[j]}`,
            type: 'line' as const,
            ...stepPath(naive[j], T_MAX),
            slot: j,
            dashed: true,
          },
        ]
      : []),
  ])
  // True survival curves exp(−λt) as thin muted lines, without legend entries.
  const truth: Segment[] = [BASE_HAZARD, BASE_HAZARD * state.hr].flatMap((h) =>
    GRID.slice(1).map((g, i) => ({ from: [GRID[i], Math.exp(-h * GRID[i])], to: [g, Math.exp(-h * g)] }) as Segment),
  )
  const fmtMedian = (v: number | null) => (v === null ? `> ${T_MAX}` : formatNumber(v))

  const xAxis = useAxis({ label: 'months', range: [0, T_MAX] })
  const yAxis = useAxis({ label: 'S(t)', range: [0, 1] })
  return (
    <Figure
      title="Kaplan–Meier under censoring"
      state={state}
      caption={
        <>
          Sixty patients per arm with exponential event times; the treated arm's hazard is the control hazard times the
          hazard ratio. Follow-up ends at 24 months, and patients also drop out at random at the chosen rate. Solid
          steps are Kaplan–Meier estimates; the thin grey curves are the true survival functions. The dashed steps drop
          the censored patients and treat the rest as a complete sample. They fall too fast, and the error grows with
          the censoring rate. The log-rank test uses the censored patients while they are at risk.
        </>
      }

      readouts={
        <>
          <Readout label="censored" value={`${Math.round(censored * 100)}%`} />
          <Readout
            label="KM median, control / treated"
            value={`${fmtMedian(median(km[0]))} / ${fmtMedian(median(km[1]))}`}
          />
          <Readout
            label="true median"
            value={`${formatNumber(Math.LN2 / BASE_HAZARD)} / ${formatNumber(Math.LN2 / (BASE_HAZARD * state.hr))}`}
          />
          <Readout label="log-rank χ²" value={formatNumber(test.chi2)} />
          <Readout label="p-value" value={formatNumber(p)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        {seriesLayers(series)}
        <Segments segments={truth} />
      </Plot>
    </Figure>
  )
}
