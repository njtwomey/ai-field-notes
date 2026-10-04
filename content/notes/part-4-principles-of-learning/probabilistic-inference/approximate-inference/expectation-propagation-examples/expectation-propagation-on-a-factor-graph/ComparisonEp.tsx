import { useMemo } from 'react'
import {
  Figure,
  float,
  formatNumber,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { comparisonEp } from '../_shared/comparisons'
import { grid, logNormalCdf, normalLogPdf, normalPdf, toMoments } from '../_shared/ep'

const GAMES = [
  { winner: 0, loser: 1 },
  { winner: 1, loser: 2 },
]
const NAMES = ['A', 'B', 'C']
const SWEEPS = 4
const XS = grid(-4, 4, 161)
const X_RANGE: [number, number] = [-4, 4]
const Y_RANGE: [number | undefined, number | undefined] = [0, undefined]
const G = grid(-5, 5, 61)

/** Exact marginals of the three skills, by summing the joint over a 61³ grid. */
function exactMarginals(noiseVar: number) {
  const s = Math.sqrt(noiseVar)
  const marg = [0, 1, 2].map(() => new Array<number>(G.length).fill(0))
  const prior = G.map((g) => normalLogPdf(g, 0, 1))
  const game = G.map((a) => G.map((b) => logNormalCdf((a - b) / s)))
  let total = 0
  for (let i = 0; i < G.length; i++)
    for (let j = 0; j < G.length; j++)
      for (let k = 0; k < G.length; k++) {
        const p = Math.exp(prior[i] + prior[j] + prior[k] + game[i][j] + game[j][k])
        marg[0][i] += p
        marg[1][j] += p
        marg[2][k] += p
        total += p
      }
  const h = G[1] - G[0]
  return marg.map((m) => {
    const d = m.map((v) => v / (total * h))
    const mean = d.reduce((a, v, i) => a + v * G[i] * h, 0)
    const variance = d.reduce((a, v, i) => a + v * (G[i] - mean) ** 2 * h, 0)
    return { density: d, mean, variance }
  })
}

/**
 * Message passing, one game factor at a time, on the two-game factor graph, against the exact marginals. B's messages
 * from the two games are revised in turn until they agree.
 */
export function ComparisonEp() {
  const state = useFigureState({
    step: float(1, { min: 0, max: GAMES.length * SWEEPS, step: 1, label: 'factor updates', format: (v) => String(v) }),
    noise: float(1, { min: 0.1, max: 3, step: 0.05, label: 'performance noise σ_n²' }),
  })

  const r = useMemo(
    () => ({
      steps: comparisonEp(3, GAMES, { mean: 0, variance: 1 }, state.noise, SWEEPS),
      exact: exactMarginals(state.noise),
    }),
    [state.noise],
  )

  const cur = state.step === 0 ? null : r.steps[state.step - 1]
  const marginals = cur ? cur.marginals : NAMES.map(() => ({ mean: 0, variance: 1 }))
  const series: SeriesSpec[] = [
    ...marginals.map((m, j) => ({
      name: `EP s_${NAMES[j]}`,
      type: 'line' as const,
      x: XS,
      y: XS.map((x) => normalPdf(x, m.mean, m.variance)),
      slot: j,
    })),
    ...r.exact.map((e, j) => ({
      name: `exact s_${NAMES[j]}`,
      type: 'line' as const,
      x: G,
      y: e.density,
      slot: j,
      dashed: true,
    })),
  ]
  const g = cur ? GAMES[cur.game] : null
  const where =
    cur && g ? `sweep ${cur.sweep + 1}, game ${cur.game + 1} (${NAMES[g.winner]} beat ${NAMES[g.loser]})` : 'prior'
  const msg = cur ? cur.messages.map((m) => toMoments(m)) : null

  const xAxis = useAxis({ label: 'skill', range: X_RANGE })
  const yAxis = useAxis({ label: 'density', range: Y_RANGE })
  return (
    <Figure
      title="Messages on the comparison graph"
      state={state}
      caption="Skill marginals after the chosen number of factor updates (solid) against the exact marginals from a grid over all three skills (dashed). Update 1 processes A's win over B with B's prior as its cavity. Update 2 processes B's win over C, with B already lowered by the first game. Update 3 revisits game 1 with the new cavity for B; after two sweeps nothing changes. Raise the performance noise σ_n² and the games say less."

      readouts={
        <>
          <Readout label="position" value={where} />
          {cur && msg && g && (
            <Readout
              label="messages sent"
              value={`to ${NAMES[g.winner]}: N(${formatNumber(msg[0].mean)}, ${formatNumber(msg[0].variance)}); to ${NAMES[g.loser]}: N(${formatNumber(msg[1].mean)}, ${formatNumber(msg[1].variance)})`}
            />
          )}
          {marginals.map((m, j) => (
            <Readout
              key={j}
              label={`s_${NAMES[j]} EP / exact`}
              value={`${formatNumber(m.mean)} ± ${formatNumber(Math.sqrt(m.variance))} / ${formatNumber(r.exact[j].mean)} ± ${formatNumber(Math.sqrt(r.exact[j].variance))}`}
            />
          ))}
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        {seriesLayers(series)}
      </Plot>
    </Figure>
  )
}
