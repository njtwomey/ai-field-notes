import * as I from 'aifn/probability/information'
import { child, normals, stream } from 'aifn/foundation/random'
import { toFlat, unwrap, type Value } from 'aifn/foundation/tensor'
import { useMemo, useState } from 'react'
import { Slider } from '@lab/controls'
import { Figure } from '@lab/layout'
import { Readout, XYChart, type XYSeries } from '@lab/viz'
import { formatValue } from '@lab/views'

const num = (v: Value) => {
  const r = unwrap(v)
  return typeof r === 'number' ? r : toFlat(r)[0]
}

// ── Divergences ──────────────────────────────────────────────────────────────────────────────────────────────────────

export function DivergencesSpecimen() {
  const [p, setP] = useState(0.3)
  const qs = useMemo(() => Array.from({ length: 199 }, (_, i) => (i + 1) / 200), [])
  const series = useMemo((): XYSeries[] => {
    const P = [p, 1 - p]
    const at = (f: (q: number[]) => number) => qs.map((q) => f([q, 1 - q]))
    return [
      { name: 'KL(p ‖ q)', type: 'line', x: qs, y: at((q) => num(I.klDivergence(P, q))), slot: 0 },
      { name: 'KL(q ‖ p)', type: 'line', x: qs, y: at((q) => num(I.klDivergence(q, P))), slot: 1 },
      { name: 'Jensen–Shannon', type: 'line', x: qs, y: at((q) => num(I.jensenShannonDivergence(P, q))), slot: 2 },
      { name: 'total variation', type: 'line', x: qs, y: at((q) => num(I.totalVariation(P, q))), slot: 3 },
      { name: 'Hellinger', type: 'line', x: qs, y: at((q) => num(I.hellingerDistance(P, q))), slot: 4 },
      {
        name: 'Pearson χ² (fDivergence)',
        type: 'line',
        x: qs,
        y: at((q) => I.fDivergence(P, q, I.fGenerators.pearsonChiSquare)),
        slot: 5,
        dashed: true,
      },
    ]
  }, [p, qs])
  return (
    <Figure
      title="Divergences between two Bernoulli distributions"
      controls={<Slider label="p" value={p} min={0.01} max={0.99} onChange={setP} />}
      caption="All vanish at q = p. KL is asymmetric and unbounded; Jensen–Shannon is at most log 2; total variation and Hellinger are at most 1."
    >
      <XYChart series={series} xLabel="q" yLabel="nats / distance" yRange={[0, 2]} />
    </Figure>
  )
}

// ── Mutual information from samples ──────────────────────────────────────────────────────────────────────────────────

export function KsgSpecimen() {
  const [rho, setRho] = useState(0.6)
  const [n, setN] = useState(500)
  const [k, setK] = useState(3)
  const { x, y } = useMemo(() => {
    const s = stream('ksg')
    const a = toFlat(normals(child(s, 'x'), n))
    const e = toFlat(normals(child(s, 'e'), n))
    return { x: a, y: a.map((v, i) => rho * v + Math.sqrt(1 - rho * rho) * e[i]) }
  }, [rho, n])
  const estimate = useMemo(() => I.ksgMutualInformation(x, y, { k }), [x, y, k])
  return (
    <Figure
      title="Kraskov–Stögbauer–Grassberger estimate of I(X; Y)"
      controls={
        <>
          <Slider label="correlation ρ" value={rho} min={-0.99} max={0.99} onChange={setRho} />
          <Slider label="samples n" value={n} min={50} max={2000} step={50} onChange={setN} />
          <Slider label="neighbours k" value={k} min={1} max={20} step={1} onChange={setK} />
        </>
      }
      readouts={
        <>
          <Readout label="KSG estimate (nats)" value={formatValue(estimate)} />
          <Readout label="Gaussian −½ log(1 − ρ²)" value={formatValue(-0.5 * Math.log(1 - rho * rho))} />
        </>
      }
    >
      <XYChart
        series={[{ name: 'samples', type: 'scatter', x, y, slot: 0 }]}
        xLabel="x"
        yLabel="y"
        equalAspect
        rescaleOnChange={false}
        holdFit="union"
      />
    </Figure>
  )
}
