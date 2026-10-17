import { useMemo, useState } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  toggle,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { stream } from 'aifn-compute/foundation/random'
import {
  bitsetAndCount,
  bitsetCount,
  descriptionKey,
  formatDescription,
  sdMap,
  selectorLanguage,
  standardQuality,
  subgroupOf,
  type Description,
} from 'aifn-compute/learning/subgroups'
import { PLANTED_MAIN, plantedSubgroups } from 'aifn-methods/data/synthetic'

const N_ROWS = 800
const K = 5

/** The Klösgen quality $(n/N)^a (p - p_0)$ from counts. */
const klosgen = (n: number, tp: number, N: number, p0: number, a: number) =>
  n > 0 ? Math.pow(n / N, a) * (tp / n - p0) : -Infinity

/**
 * Subgroup discovery on a population with a planted subgroup: every description of up to `depth` selectors placed by
 * its size and outcome rate, the isometrics of the Klösgen quality, the top subgroups found by exhaustive search, and
 * a draggable hypothetical subgroup.
 */
export function SubgroupSpace() {
  const fig = useFigureState({
    a: float(1, { min: 0, max: 1, step: 0.05, label: 'size exponent a', suggestions: [0, 0.5, 1] }),
    depth: int(2, { min: 1, max: 3, label: 'selectors per description at most' }),
    bins: int(4, { min: 2, max: 8, label: 'intervals per numeric attribute', suggestions: [3, 4, 6] }),
    minSupport: int(10, { min: 1, max: 100, label: 'smallest subgroup (rows)', suggestions: [5, 10, 40] }),
    diverse: toggle(false, { label: 'drop subgroups whose rows overlap a better one' }),
    seed: int(1, { min: 1, max: 999, label: 'seed' }),
  })
  const { a, depth, bins, minSupport, diverse, seed } = fig

  const data = useMemo(() => plantedSubgroups(stream(`subgroups-${seed}`), { n: N_ROWS }), [seed])
  const outcome = useMemo(() => Array.from(data.table.outcome as unknown as ArrayLike<number>, Number), [data])
  const N = outcome.length
  const P = outcome.reduce((s, v) => s + v, 0)
  const p0 = P / N
  const language = useMemo(() => selectorLanguage(data.table, { exclude: ['outcome', 'cost'], bins }), [data, bins])
  const measure = useMemo(() => standardQuality(outcome, { a }), [outcome, a])

  const cloud = useMemo(() => {
    const out: { d: Description; n: number; tp: number }[] = []
    const target = standardQuality(outcome).target
    const visit = (d: Description) => {
      for (const r of language.refinements(d)) {
        const c = language.cover(r)
        const n = bitsetCount(c)
        if (n < minSupport) continue
        out.push({ d: r, n, tp: bitsetAndCount(c, target) })
        if (r.length < depth) visit(r)
      }
    }
    visit([])
    return out
  }, [language, outcome, depth, minSupport])

  const top = useMemo(() => {
    const found = sdMap(language, measure, { k: diverse ? 4 * K : K, maxDepth: depth, minSupport })
    if (!diverse) return found
    const kept: typeof found = []
    for (const g of found) {
      const overlaps = kept.some((h) => {
        const both = bitsetAndCount(g.cover, h.cover)
        return both / (g.size + h.size - both) >= 0.5
      })
      if (!overlaps) kept.push(g)
      if (kept.length === K) break
    }
    return kept
  }, [language, measure, depth, minSupport, diverse])

  const planted = useMemo(() => {
    const g = subgroupOf(language, measure, language.canonical(PLANTED_MAIN))
    const tp = bitsetAndCount(g.cover, measure.target)
    const better = cloud.filter((c) => klosgen(c.n, c.tp, N, p0, a) > g.quality).length
    return { g, tp, rank: better + 1 }
  }, [language, measure, cloud, N, p0, a])

  const [probe, setProbe] = useState<[number, number] | null>(null)
  const at: [number, number] = probe ?? [200, 0.5]
  const probeQ = (aa: number) => klosgen(at[0], at[0] * at[1], N, p0, aa)

  const isometrics = useMemo(() => {
    const levels = top.map((g) => g.quality).filter((q) => q > 0)
    const xs = Array.from({ length: 120 }, (_, i) => Math.max(1, (N * (i + 1)) / 120))
    return levels.slice(0, 3).map((q) => ({
      q,
      x: xs,
      y: xs.map((n) => {
        const p = p0 + q / Math.pow(n / N, a)
        return p <= 1 ? p : NaN
      }),
    }))
  }, [top, N, p0, a])

  const topKeys = new Set(top.map((g) => g.key))
  const rest = cloud.filter((c) => !topKeys.has(descriptionKey(c.d)))
  const xa = useAxis({ label: 'subgroup size n (rows)', range: [0, N], nice: false })
  const ya = useAxis({ label: 'outcome rate in the subgroup', range: [0, 1], nice: false })

  return (
    <Figure
      title="Subgroup discovery: size against unusualness"
      state={fig}
      caption={`A population of ${N} people with age, smoking, sex, region, exercise, BMI and income; the outcome rate is ${formatNumber(p0)} overall and was planted at 0.7 among smokers aged 50 or more. Each grey point is a description of up to ${depth} selectors (age and the numeric attributes cut into ${bins} equal-frequency intervals), placed by the rows it covers and the outcome rate among them. The quality is the Klösgen family (n/N)^a (p − p₀): a = 0 ranks by the rise in rate alone, a = 1/2 by the binomial test statistic, a = 1 by weighted relative accuracy. The curves are isometrics through the best qualities, and the coloured points are the top ${K} found by exhaustive search. Raising a moves the isometrics so that larger subgroups win; lowering it rewards small, pure ones. The dashed line is the overall rate. Drag the ink point to read the quality of any size and rate.`}
      readouts={{
        [`top ${K} at a = ${a}`]: (
          <>
            {top.map((g, k) => (
              <Readout
                key={g.key}
                label={`${k + 1}. ${formatDescription(g.description, 3)}`}
                value={`n = ${g.size}, rate ${formatNumber(bitsetAndCount(g.cover, measure.target) / g.size)}, q = ${formatNumber(g.quality)}`}
              />
            ))}
          </>
        ),
        'planted and probe': (
          <>
            <Readout
              label="age ≥ 50 ∧ smoker = yes (planted)"
              value={`n = ${planted.g.size}, rate ${formatNumber(planted.tp / planted.g.size)}, q = ${formatNumber(planted.g.quality)}, rank ${planted.rank}`}
            />
            <Readout
              label={`probe n = ${Math.round(at[0])}, rate ${formatNumber(at[1])}`}
              value={`q at a = 0, ½, 1: ${formatNumber(probeQ(0))}, ${formatNumber(probeQ(0.5))}, ${formatNumber(probeQ(1))}`}
            />
          </>
        ),
      }}
    >
      <Plot x={xa} y={ya}>
        <Curve name="overall rate p₀" x={[0, N]} y={[p0, p0]} dashed muted />
        {isometrics.map((c, k) => (
          <Curve key={k} name={`isometric q = ${formatNumber(c.q)}`} x={c.x} y={c.y} muted silent />
        ))}
        {rest.length > 0 && (
          <Points name="descriptions" x={rest.map((c) => c.n)} y={rest.map((c) => c.tp / c.n)} muted thin />
        )}
        {top.length > 0 && (
          <Points
            name="top subgroups"
            x={top.map((g) => g.size)}
            y={top.map((g) => bitsetAndCount(g.cover, measure.target) / g.size)}
            labels={top.map((_, k) => String(k + 1))}
            slot={0}
            size={10}
          />
        )}
        <Points
          name="planted subgroup"
          x={[planted.g.size]}
          y={[planted.tp / planted.g.size]}
          labels={['planted']}
          slot={1}
          shape={2}
          size={11}
        />
        <Handle
          kind="point"
          at={at}
          label="probe"
          onDrag={([n, p]) => setProbe([Math.min(N, Math.max(1, n)), Math.min(1, Math.max(0, p))])}
        />
      </Plot>
    </Figure>
  )
}
