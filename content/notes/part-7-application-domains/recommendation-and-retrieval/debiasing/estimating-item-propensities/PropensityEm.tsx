import { useMemo, useState } from 'react'
import { Curve, Figure, float, formatNumber, Player, Plot, Raster, Readout, useAxis, useFigureState } from 'aifn-render'
import { logGamma } from 'aifn/numerics/special'

/** Click counts C_ui for 4 users × 5 items. User 3 clicked item 5 six times, far above what the margins predict. */
const C = [
  [4, 2, 1, 1, 0],
  [6, 3, 2, 1, 1],
  [2, 1, 0, 0, 6],
  [8, 4, 2, 1, 1],
]
const U = C.length
const M = C[0].length
const ROW = C.map((r) => r.reduce((a, b) => a + b, 0))
const COL = C[0].map((_, i) => C.reduce((s, r) => s + r[i], 0))
const USERS = [1, 2, 3, 4]
const ITEMS = [1, 2, 3, 4, 5]
const ITERS = 40

type State = { alpha: number[]; beta: number[] }

const rescale = ({ alpha, beta }: State): State => {
  const m = alpha.reduce((a, b) => a + b, 0) / U
  return { alpha: alpha.map((x) => x / m), beta: beta.map((x) => x * m) }
}

/** Model A: the rank-1 Poisson maximum-likelihood fit, α_u = C_u· / Σβ and β_i = C_·i / Σα, from α = β = 1. */
function ipf(): State {
  let beta = new Array(M).fill(1)
  const alpha = ROW.map((r) => r / beta.reduce((a, b) => a + b, 0))
  beta = COL.map((c) => c / alpha.reduce((a, b) => a + b, 0))
  return rescale({ alpha, beta })
}

const posteriorMean = (s: State, a: number) =>
  C.map((row, u) => row.map((c, i) => (a + c) / (a + s.alpha[u] * s.beta[i])))

/** Model B: EM for C_ui | r_ui ~ Poisson(α_u β_i r_ui), r_ui ~ Gamma(a, a). Returns the state after each iteration. */
function emHistory(a: number): State[] {
  const out = [ipf()]
  for (let t = 0; t < ITERS; t++) {
    const s = out[out.length - 1]
    const Er = posteriorMean(s, a)
    const alpha = ROW.map((r, u) => r / s.beta.reduce((acc, b, i) => acc + b * Er[u][i], 0))
    const beta = COL.map((c, i) => c / alpha.reduce((acc, al, u) => acc + al * Er[u][i], 0))
    out.push(rescale({ alpha, beta }))
  }
  return out
}

/** Marginal log-likelihood under the negative binomial obtained by integrating out r_ui. */
function marginalLogLik(s: State, a: number): number {
  let ll = 0
  C.forEach((row, u) =>
    row.forEach((c, i) => {
      const lam = s.alpha[u] * s.beta[i]
      ll +=
        logGamma(a + c) - logGamma(a) - logGamma(c + 1) + a * Math.log(a / (a + lam)) + c * Math.log(lam / (a + lam))
    }),
  )
  return ll
}

const normalise = (beta: number[]) => {
  const s = beta.reduce((a, b) => a + b, 0)
  return beta.map((b) => b / s)
}

/** IPF against Gamma–Poisson EM on a small click matrix: per-user item propensities and posterior relevance. */
export function PropensityEm() {
  const state = useFigureState({
    a: float(1, { min: 0.25, max: 100, scale: 'log10', suggestions: [0.3, 1, 10, 100], label: 'shrinkage a' }),
  })
  const [iteration, setIteration] = useState(0)
  const a = state.a
  const hist = useMemo(() => emHistory(a), [a])
  const base = hist[0]
  const s = hist[iteration]
  const Er = useMemo(() => posteriorMean(s, a), [s, a])
  const logEr = useMemo(() => Er.map((row) => row.map(Math.log)), [Er])

  const series = [
    { name: 'IPF (model A)', x: ITEMS, y: normalise(base.beta), slot: 0 },
    { name: `EM (model B), iteration ${iteration}`, x: ITEMS, y: normalise(s.beta), slot: 1 },
  ] as const

  const xAxis = useAxis({ label: 'item', range: [1, 5] })
  const yAxis = useAxis({ label: 'per-user propensity π̂', range: [0, 0.5] })
  const xAxis2 = useAxis({ label: 'item' })
  const yAxis2 = useAxis({ label: 'user' })
  return (
    <Figure
      title="Exposure baseline and latent relevance"
      caption="Left: the per-user item propensity π̂_ui = β_i / Σ_j β_j from the rank-1 Poisson fit (IPF) and from Gamma–Poisson EM. Right: the logarithm of the posterior mean relevance E[r_ui], so 0 means average relevance. User 3's six clicks on item 5 are attributed to relevance (E[r] > 1), and the other users' few clicks on item 5 to low relevance, so EM raises item 5's exposure baseline. A large shrinkage a forces every r_ui towards 1 and recovers IPF; a small a lets pairwise relevance absorb more. The player steps through the EM iterations from the IPF start."
      state={state}
      controls={<Player value={iteration} onChange={setIteration} count={ITERS + 1} label="EM iteration" />}
      readouts={
        <>
          <Readout label="marginal log-likelihood, IPF" value={formatNumber(marginalLogLik(base, a))} />
          <Readout label="at this iteration" value={formatNumber(marginalLogLik(s, a))} />
          <Readout label="E[r] for user 3, item 5" value={formatNumber(Er[2][4])} />
          <Readout
            label="π̂ for item 5, IPF → EM"
            value={`${formatNumber(normalise(base.beta)[4])} → ${formatNumber(normalise(s.beta)[4])}`}
          />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={280}>
          <Curve {...series[0]} />
          <Curve {...series[1]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={280}>
          <Raster x={ITEMS} y={USERS} z={logEr} scale={'diverging'} range={[-1.2, 1.2]} valueLabel={'ln E[r_ui]'} />
        </Plot>
      </div>
    </Figure>
  )
}
