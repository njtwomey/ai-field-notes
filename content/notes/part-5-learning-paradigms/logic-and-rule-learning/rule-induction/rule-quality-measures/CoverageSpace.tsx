import { useMemo, useState } from 'react'
import {
  choice,
  Contours,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  Plot,
  Points,
  Raster,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import {
  mineAntecedents,
  nominalConditions,
  ruleText,
  RULE_MEASURES,
  ruleQuality,
  type RuleMeasure,
} from 'aifn-compute/learning/rules'
import { titanicRows, TITANIC_LEVELS, TITANIC_NAMES } from '../_shared/rule-figures'

const G = 61
const SHOWN: RuleMeasure[] = [
  'precision',
  'laplace',
  'm-estimate',
  'accuracy',
  'wracc',
  'foil-gain',
  'likelihood-ratio',
]
const MEASURE_OPTIONS = RULE_MEASURES.filter((m) => SHOWN.includes(m.key)).map((m) => ({ value: m.key, label: m.name }))
const TARGETS = [
  { value: 1, label: 'survived' },
  { value: 0, label: 'died' },
] as const

/** Quantile levels of the finite values of a field, for its isometrics. */
function levels(z: number[][], count: number): number[] {
  const v = z
    .flat()
    .filter(Number.isFinite)
    .sort((a, b) => a - b)
  if (v.length === 0) return []
  const out: number[] = []
  for (let k = 1; k <= count; k++) {
    const q = v[Math.floor((k / (count + 1)) * (v.length - 1))]
    if (out.length === 0 || q > out[out.length - 1] + 1e-12) out.push(q)
  }
  return out
}

/**
 * Coverage space of the Titanic data: every rule of at most two conditions as a point (negatives covered, positives
 * covered), the chosen measure as a field with its isometrics, and a draggable rule.
 */
export function CoverageSpace() {
  const fig = useFigureState({
    measure: choice<RuleMeasure>(MEASURE_OPTIONS, 'laplace', { label: 'measure' }),
    m: float(10, { min: 0, max: 200, step: 1, label: 'm of the m-estimate', suggestions: [2, 10, 50] }),
    target: choice<0 | 1>(TARGETS, 1, { label: 'rules predict' }),
  })
  const { measure, m, target } = fig

  const data = useMemo(() => titanicRows(), [])
  const ants = useMemo(
    () => mineAntecedents(data.rows, nominalConditions([4, 2, 2]), { minSupport: 0.01, maxLength: 2 }),
    [data],
  )
  const counts = useMemo(() => {
    const positive = data.y.map((v) => v === target)
    const P = positive.filter(Boolean).length
    const N = data.y.length - P
    const rules = ants.map((a) => {
      let p = 0
      let n = 0
      for (let i = 0; i < data.y.length; i++)
        if (a.mask[i]) {
          if (positive[i]) p++
          else n++
        }
      return { text: ruleText(a.conditions, { names: TITANIC_NAMES, levels: TITANIC_LEVELS }), p, n }
    })
    return { P, N, rules }
  }, [ants, data, target])
  const { P, N, rules } = counts

  const score = (k: RuleMeasure, p: number, n: number) => ruleQuality(k, { p, n, P, N }, { m })
  const field = useMemo(() => {
    const gx = Array.from({ length: G }, (_, i) => (N * i) / (G - 1))
    const gy = Array.from({ length: G }, (_, j) => (P * j) / (G - 1))
    const z = gy.map((p) => gx.map((n) => score(measure, p, n)))
    return { gx, gy, z, levels: levels(z, 9) }
  }, [measure, m, P, N]) // eslint-disable-line react-hooks/exhaustive-deps

  const ranked = useMemo(
    () =>
      rules
        .map((r) => ({ ...r, q: score(measure, r.p, r.n) }))
        .filter((r) => Number.isFinite(r.q))
        .sort((a, b) => b.q - a.q),
    [rules, measure, m, P, N], // eslint-disable-line react-hooks/exhaustive-deps
  )

  const [probe, setProbe] = useState<[number, number] | null>(null)
  const at: [number, number] = probe ?? [Math.round(N * 0.1), Math.round(P * 0.5)]
  const clamp = (v: number, hi: number) => Math.round(Math.min(hi, Math.max(0, v)))

  const xAxis = useAxis({ label: 'negatives covered n', range: [0, N], nice: false })
  const yAxis = useAxis({ label: 'positives covered p', range: [0, P], nice: false })
  const top = new Set(ranked.slice(0, 3).map((r) => r.text))

  return (
    <Figure
      title="Rule quality measures in coverage space"
      state={fig}
      caption={`Coverage space of the Titanic data with "${target === 1 ? 'survived' : 'died'}" as the positive class: P = ${P} positives and N = ${N} negatives. Each point is one rule of at most two conditions (class, sex, age), placed at the negatives n and positives p it covers; the three best under the chosen measure are in ink. The shading is the measure over the whole space and the curves are its isometrics, the rules it scores equally. The dashed diagonal holds rules whose precision equals the prior P/(P + N). Precision's isometrics are lines through the origin, so a rule covering a few positives and no negatives beats every larger rule; the Laplace and m-estimates pivot their lines about a point below and left of the origin, which favours coverage; rule accuracy's isometrics are the 45° lines p − n = constant, and weighted relative accuracy's are parallel to the diagonal, so it weighs a positive by N and a negative by P; the m-estimate turns from precision at m = 0 towards weighted relative accuracy as m grows. Drag the ink point to read every measure for a hypothetical rule.`}
      readouts={{
        [`rule at n = ${at[0]}, p = ${at[1]}`]: (
          <>
            {SHOWN.map((k) => (
              <Readout
                key={k}
                label={RULE_MEASURES.find((x) => x.key === k)?.name ?? k}
                value={formatNumber(score(k, at[1], at[0]))}
              />
            ))}
          </>
        ),
        'best rules under this measure': (
          <>
            {ranked.slice(0, 3).map((r, k) => (
              <Readout key={k} label={`${k + 1}. ${r.text}`} value={`p = ${r.p}, n = ${r.n}, ${formatNumber(r.q)}`} />
            ))}
          </>
        ),
      }}
    >
      <Plot x={xAxis} y={yAxis}>
        <Raster
          x={field.gx}
          y={field.gy}
          z={field.z}
          valueLabel={RULE_MEASURES.find((x) => x.key === measure)?.name}
          fillOpacity={0.5}
        />
        {field.levels.length > 0 && (
          <Contours x={field.gx} y={field.gy} z={field.z} levels={field.levels} labels={false} />
        )}
        <Curve name="precision = prior" x={[0, N]} y={[0, P]} dashed muted />
        <Points
          name="rules"
          x={rules.filter((r) => !top.has(r.text)).map((r) => r.n)}
          y={rules.filter((r) => !top.has(r.text)).map((r) => r.p)}
          labels={rules.filter((r) => !top.has(r.text)).map((r) => r.text)}
          labelPriority={rules.filter((r) => !top.has(r.text)).map((r) => r.p + r.n)}
          slot={0}
        />
        <Points
          name="best three"
          x={ranked.slice(0, 3).map((r) => r.n)}
          y={ranked.slice(0, 3).map((r) => r.p)}
          labels={ranked.slice(0, 3).map((r) => r.text)}
          emphasis
          size={10}
        />
        <Handle kind="point" at={at} label="a rule" onDrag={([n, p]) => setProbe([clamp(n, N), clamp(p, P)])} />
      </Plot>
    </Figure>
  )
}
