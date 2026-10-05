import * as I from 'aifn-compute/probability/information'
import { child, normals, stream } from 'aifn-compute/foundation/random'
import { toFlat, unwrap, type Value } from 'aifn-compute/foundation/tensor'
import { useMemo } from 'react'
import { Equation, Figure, live, tex } from 'aifn-render/layout'
import { row, slider, useFigureState } from 'aifn-render/state'
import { Annotation, Curve, Plot, Points, useAxis } from 'aifn-render/viz'

const num = (v: Value) => {
  const r = unwrap(v)
  return typeof r === 'number' ? r : toFlat(r)[0]
}

// ── Divergences ──────────────────────────────────────────────────────────────────────────────────────────────────────

const QS = Array.from({ length: 199 }, (_, i) => (i + 1) / 200)

export function DivergencesSpecimen() {
  const state = useFigureState({ p: slider(0.01, 0.99, 0.3, { label: 'p' }) })
  const p = state.p
  const curves = useMemo(() => {
    const P = [p, 1 - p]
    const at = (f: (q: number[]) => number) => QS.map((q) => f([q, 1 - q]))
    return [
      { name: 'KL(p ‖ q)', y: at((q) => num(I.klDivergence(P, q))) },
      { name: 'KL(q ‖ p)', y: at((q) => num(I.klDivergence(q, P))) },
      { name: 'Jensen–Shannon', y: at((q) => num(I.jensenShannonDivergence(P, q))) },
      { name: 'total variation', y: at((q) => num(I.totalVariation(P, q))) },
      { name: 'Hellinger', y: at((q) => num(I.hellingerDistance(P, q))) },
      { name: 'Pearson χ² (fDivergence)', y: at((q) => I.fDivergence(P, q, I.fGenerators.pearsonChiSquare)) },
    ]
  }, [p])
  const x = useAxis({ label: 'q', range: [0, 1] })
  const y = useAxis({ label: 'nats / distance', range: [0, 2] })
  return (
    <Figure
      title="Divergences between two Bernoulli distributions"
      purpose="Every divergence is zero exactly at q = p and grows as q moves away; KL is asymmetric and unbounded, while Jensen–Shannon, total variation and Hellinger stay bounded."
      state={state}
      caption="Bernoulli(p) against Bernoulli(q) for every q; the dotted line marks q = p. KL(p ‖ q) blows up as q nears 0 or 1 where p puts mass; Jensen–Shannon is at most log 2 ≈ 0.69; total variation and Hellinger are at most 1. The Pearson χ² is computed by the generic fDivergence."
    >
      <Plot x={x} y={y}>
        <Annotation x={p} text="q = p" dashed />
        {curves.map((c, i) => (
          <Curve key={c.name} name={c.name} x={QS} y={c.y} slot={i} dashed={i === 5} />
        ))}
      </Plot>
    </Figure>
  )
}

// ── Mutual information from samples ──────────────────────────────────────────────────────────────────────────────────

export function KsgSpecimen() {
  const state = useFigureState({
    data: row('1 · data', {
      rho: slider(-0.99, 0.99, 0.6, { label: 'correlation ρ' }),
      n: slider(50, 2000, 500, { step: 50, label: 'samples n' }),
    }),
    estimator: row('2 · estimator', { k: slider(1, 20, 3, { step: 1, label: 'neighbours k' }) }),
  })
  const { rho, n } = state.data
  const k = state.estimator.k
  const { x, y } = useMemo(() => {
    const s = stream('ksg')
    const a = toFlat(normals(child(s, 'x'), n))
    const e = toFlat(normals(child(s, 'e'), n))
    return { x: a, y: a.map((v, i) => rho * v + Math.sqrt(1 - rho * rho) * e[i]) }
  }, [rho, n])
  const estimate = useMemo(() => I.ksgMutualInformation(x, y, { k }), [x, y, k])
  const exact = -0.5 * Math.log(1 - rho * rho)
  const xa = useAxis({ label: 'x', range: [-4, 4] })
  const ya = useAxis({ label: 'y', range: [-4, 4], equal: xa })
  return (
    <Figure
      title="Kraskov–Stögbauer–Grassberger estimate of I(X; Y)"
      purpose="KSG estimates mutual information from distances to the k-th nearest neighbour, with no density estimate; on correlated Gaussians it tracks the closed form −½ log(1 − ρ²)."
      state={state}
      equation={
        <Equation>
          {tex`\hat I_{\mathrm{KSG}} = ${live(estimate, { digits: 4, strong: true })} \quad\text{against}\quad -\tfrac12 \log(1 - ${live(rho, { digits: 3 })}^2) = ${live(exact, { digits: 4 })} \text{ nats}`}
        </Equation>
      }
      caption="n draws of a standard bivariate normal with correlation ρ. Raise |ρ| and the cloud thins to a line while the information grows without bound; small n or large k bias the estimate low."
    >
      <Plot x={xa} y={ya}>
        <Points name="samples" x={x} y={y} slot={0} thin size={4} />
      </Plot>
    </Figure>
  )
}
