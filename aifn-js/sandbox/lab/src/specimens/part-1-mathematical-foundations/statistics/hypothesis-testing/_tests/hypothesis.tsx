import { child, stream, type Stream } from 'aifn/foundation/random'
import { tensor, toFlat } from 'aifn/foundation/tensor'
import { normalCdf } from 'aifn/numerics/special'
import { Categorical, LogNormal, Normal, StudentT } from 'aifn/probability/distributions'
import {
  binomialTest,
  chiSquareGoodnessOfFit,
  chiSquareIndependence,
  fisherExact,
  grubbs,
  gTestIndependence,
  ksTest,
  mannWhitneyU,
  oneSampleTTest,
  oneWayAnova,
  pooledTTest,
  shapiroWilk,
  welchTTest,
  wilcoxonSignedRank,
  zTest,
  type Alternative,
  type TestResult,
} from 'aifn/probability/tests'
import { useMemo, useState } from 'react'
import { Figure } from '@lab/layout'
import { choice, slider, useFigureState, variants } from '@lab/state'
import { HypothesisTestPanel } from '@lab/views'
import { Bars, Handle, Plot, Points, useAxis } from '@lab/viz'

// ── Data generators ─────────────────────────────────────────────────────────────────────────────────────────────────

type Shape = 'normal' | 'skewed' | 'heavy'

/** n draws with mean `mean` and standard deviation `sd`, from a normal, a skewed (log-normal) or a heavy-tailed (t₃) law. */
function draws(s: Stream, n: number, mean: number, sd: number, shape: Shape): number[] {
  if (shape === 'normal') return toFlat(Normal(mean, sd).sample(s, { shape: [n] }) as never)
  if (shape === 'skewed') {
    // Log-normal with σ = 0.8, standardised to mean 0 and variance 1.
    const m = Math.exp(0.32)
    const v = (Math.exp(0.64) - 1) * Math.exp(0.64)
    return toFlat(LogNormal(0, 0.8).sample(s, { shape: [n] }) as never).map((x) => mean + (sd * (x - m)) / Math.sqrt(v))
  }
  // Student t with 3 degrees of freedom has variance 3.
  return toFlat(StudentT(3).sample(s, { shape: [n] }) as never).map((x) => mean + (sd * x) / Math.sqrt(3))
}

const round = (v: number) => Math.round(v * 1000) / 1000

// ── The tests and their data parameters ─────────────────────────────────────────────────────────────────────────────

const seed = slider(1, 50, 1, { step: 1, label: 'seed' })
const shape = choice(
  [
    { value: 'normal', label: 'normal' },
    { value: 'skewed', label: 'skewed (log-normal)' },
    { value: 'heavy', label: 'heavy-tailed (t₃)' },
  ],
  'normal',
  { label: 'population shape' },
)
const n = slider(3, 60, 15, { step: 1, label: 'n' })
const oneSample = {
  n,
  mean: slider(-2, 2, 0.5, { step: 0.05, label: 'true mean μ' }),
  sd: slider(0.2, 3, 1, { step: 0.05, label: 'sd σ' }),
  mu0: slider(-2, 2, 0, { step: 0.05, label: 'null mean μ₀' }),
  shape,
  seed,
}
const twoSample = {
  n1: slider(3, 60, 15, { step: 1, label: 'control n₁' }),
  n2: slider(3, 60, 15, { step: 1, label: 'treatment n₂' }),
  shift: slider(-2, 2, 0.6, { step: 0.05, label: 'true shift δ' }),
  ratio: slider(0.25, 4, 1, { step: 0.05, label: 'sd ratio σ₂/σ₁' }),
  shape,
  seed,
}
const table = {
  a: slider(0, 40, 12, { step: 1, label: 'a (row 1, col 1)' }),
  b: slider(0, 40, 5, { step: 1, label: 'b (row 1, col 2)' }),
  c: slider(0, 40, 7, { step: 1, label: 'c (row 2, col 1)' }),
  d: slider(0, 40, 9, { step: 1, label: 'd (row 2, col 2)' }),
}

const TESTS = variants(
  {
    oneSampleT: { label: 'one-sample t-test', params: oneSample },
    z: { label: 'z-test (σ known)', params: oneSample },
    wilcoxon: { label: 'Wilcoxon signed-rank', params: oneSample },
    ks1: { label: 'Kolmogorov–Smirnov against N(μ₀, σ²)', params: oneSample },
    shapiro: { label: 'Shapiro–Wilk (normality)', params: oneSample },
    grubbs: { label: "Grubbs' outlier test", params: oneSample },
    welch: { label: "Welch's t-test", params: twoSample },
    pooled: { label: 'pooled two-sample t-test', params: twoSample },
    mannWhitney: { label: 'Mann–Whitney U', params: twoSample },
    ks2: { label: 'two-sample Kolmogorov–Smirnov', params: twoSample },
    anova: {
      label: 'one-way ANOVA',
      params: {
        groups: slider(2, 5, 3, { step: 1, label: 'groups k' }),
        n: slider(2, 40, 10, { step: 1, label: 'n per group' }),
        spread: slider(0, 2, 0.5, { step: 0.05, label: 'spread of true means' }),
        shape,
        seed,
      },
    },
    binomial: {
      label: 'exact binomial test',
      params: {
        trials: slider(1, 200, 40, { step: 1, label: 'trials n' }),
        successes: slider(0, 200, 14, { step: 1, label: 'successes k' }),
        p0: slider(0.01, 0.99, 0.5, { step: 0.01, label: 'null p₀' }),
      },
    },
    chi2gof: {
      label: 'χ² goodness of fit (uniform)',
      params: {
        k: slider(2, 8, 4, { step: 1, label: 'categories k' }),
        n: slider(10, 500, 120, { step: 10, label: 'n' }),
        tilt: slider(0, 1, 0.3, { step: 0.05, label: 'true tilt from uniform' }),
        seed,
      },
    },
    fisher: { label: "Fisher's exact test (2 × 2)", params: table },
    chi2: { label: 'χ² test of independence (2 × 2)', params: table },
    g: { label: 'G-test of independence (2 × 2)', params: table },
  },
  {
    label: '1 · test and data',
    choiceLabel: 'test',
    shared: {
      alternative: choice(
        [
          { value: 'two-sided', label: 'two-sided' },
          { value: 'less', label: 'less' },
          { value: 'greater', label: 'greater' },
        ],
        'two-sided',
        { label: 'alternative' },
      ),
      alpha: slider(0.001, 0.2, 0.05, { step: 0.001, label: 'level α' }),
    },
  },
)

type P = Readonly<Record<string, number | string>>
const ONE = ['oneSampleT', 'z', 'wilcoxon', 'ks1', 'shapiro', 'grubbs']
const TWO = ['welch', 'pooled', 'mannWhitney', 'ks2']

/** The samples a test reads, generated from its parameters (one, two or k groups); counts tests have none. */
function generate(key: string, p: P): number[][] {
  const s = stream(`hypothesis/${p.seed ?? 0}`)
  const sh = (p.shape ?? 'normal') as Shape
  if (ONE.includes(key)) return [draws(s, p.n as number, p.mean as number, p.sd as number, sh).map(round)]
  if (TWO.includes(key))
    return [
      draws(child(s, 'control'), p.n1 as number, 0, 1, sh).map(round),
      draws(child(s, 'treatment'), p.n2 as number, p.shift as number, p.ratio as number, sh).map(round),
    ]
  if (key === 'anova') {
    const k = p.groups as number
    return Array.from({ length: k }, (_, g) =>
      draws(child(s, g), p.n as number, (p.spread as number) * (g - (k - 1) / 2), 1, sh).map(round),
    )
  }
  return []
}

/** Category counts of n draws from k categories whose probabilities tilt linearly away from uniform. */
function gofCounts(p: P): number[] {
  const k = p.k as number
  const probs = Array.from({ length: k }, (_, i) => 1 + (p.tilt as number) * ((2 * i) / Math.max(1, k - 1) - 1))
  const x = toFlat(Categorical(tensor(probs)).sample(stream(`gof/${p.seed}`), { shape: [p.n as number] }) as never)
  const counts = new Array<number>(k).fill(0)
  for (const v of x) counts[v]++
  return counts
}

function runTest(key: string, p: P, data: number[][], alternative: Alternative, alpha: number): TestResult {
  const o = { alternative, level: 1 - alpha }
  const [x, y] = data
  const tbl = [
    [p.a as number, p.b as number],
    [p.c as number, p.d as number],
  ]
  switch (key) {
    case 'oneSampleT':
      return oneSampleTTest(x, { ...o, mu: p.mu0 as number })
    case 'z':
      return zTest(x, { ...o, mu: p.mu0 as number, sigma: p.sd as number })
    case 'wilcoxon':
      return wilcoxonSignedRank(x, null, { alternative, mu: p.mu0 as number })
    case 'ks1':
      return ksTest(x, (v) => normalCdf((v - (p.mu0 as number)) / (p.sd as number)) as number, { alternative })
    case 'shapiro':
      return shapiroWilk(x)
    case 'grubbs':
      return grubbs(x, { alternative })
    case 'welch':
      return welchTTest(y, x, o)
    case 'pooled':
      return pooledTTest(y, x, o)
    case 'mannWhitney':
      return mannWhitneyU(y, x, { alternative })
    case 'ks2':
      return ksTest(y, x, { alternative })
    case 'anova':
      return oneWayAnova(data)
    case 'binomial':
      return binomialTest(Math.min(p.successes as number, p.trials as number), p.trials as number, {
        ...o,
        p: p.p0 as number,
      })
    case 'chi2gof':
      return chiSquareGoodnessOfFit(gofCounts(p))
    case 'fisher':
      return fisherExact(tbl, o)
    case 'chi2':
      return chiSquareIndependence(tbl)
    default:
      return gTestIndependence(tbl)
  }
}

/** Whether a test reads the alternative (the χ² family, Shapiro–Wilk and ANOVA test any departure). */
const TAKES_ALTERNATIVE = new Set([...ONE.filter((k) => k !== 'shapiro'), ...TWO, 'binomial', 'fisher'])

export function HypothesisTestSpecimen() {
  const state = useFigureState({ test: TESTS })
  const test = state.test
  const key = test.key as string
  const p = test.values as unknown as P
  const alternative = (TAKES_ALTERNATIVE.has(key) ? p.alternative : 'two-sided') as Alternative
  const alpha = p.alpha as number

  // The data: generated from the parameters, then edited by dragging points (edits last until a parameter changes).
  const dataKey = `${key}:${JSON.stringify({ ...p, alpha: 0, alternative: 0 })}`
  const generated = useMemo(() => generate(key, p), [dataKey]) // eslint-disable-line react-hooks/exhaustive-deps
  const [edits, setEdits] = useState<{ key: string; data: number[][] } | null>(null)
  const data = edits?.key === dataKey ? edits.data : generated
  const move = (g: number, i: number, v: number) =>
    setEdits({ key: dataKey, data: data.map((s, j) => (j === g ? s.map((u, k) => (k === i ? round(v) : u)) : s)) })

  const result = useMemo(() => {
    try {
      return { ok: runTest(key, p, data, alternative, alpha) }
    } catch (e) {
      return { error: (e as Error).message }
    }
  }, [key, p, data, alternative, alpha])

  const samples = data.length > 0
  const all = data.flat()
  const lo = Math.min(...all, ONE.includes(key) ? (p.mu0 as number) : Infinity)
  const hi = Math.max(...all, ONE.includes(key) ? (p.mu0 as number) : -Infinity)
  const pad = 0.1 * (hi - lo || 1)
  const xa = useAxis({ label: 'value', range: samples ? [lo - pad, hi + pad] : [0, 1], key: dataKey })
  const ya = useAxis({
    label: 'group',
    range: [-0.6, data.length - 0.4],
    integer: true,
    format: (v) => (TWO.includes(key) ? (['control', 'treatment'][v] ?? '') : key === 'anova' ? `g${v + 1}` : ''),
  })
  const counts = key === 'chi2gof' ? gofCounts(p) : null
  const ca = useAxis({ label: 'category', integer: true, key: dataKey })
  const cy = useAxis({ label: 'count', key: dataKey })

  const top = samples ? (
    <Plot x={xa} y={ya}>
      {data.map((s, g) => (
        <Points
          key={g}
          name={TWO.includes(key) ? ['control', 'treatment'][g] : `group ${g + 1}`}
          x={s}
          y={s.map(() => g)}
          slot={g}
        />
      ))}
      {data.flatMap((s, g) =>
        s.length <= 60
          ? s.map((v, i) => <Handle key={`${g}-${i}`} kind="point" at={[v, g]} onDrag={([x]) => move(g, i, x)} />)
          : [],
      )}
      {ONE.includes(key) && <Handle {...state.handle('test.mu0', { label: 'μ₀' })} />}
    </Plot>
  ) : counts ? (
    <Plot x={ca} y={cy}>
      <Bars name="observed" x={counts.map((_, i) => i)} y={counts} slot={0} />
      <Points
        name="expected (uniform)"
        x={counts.map((_, i) => i)}
        y={counts.map(() => (p.n as number) / counts.length)}
        emphasis
      />
    </Plot>
  ) : undefined

  return (
    <Figure
      title="A test's statistic, its null law and the p-value"
      purpose="A test reduces the data to one statistic and compares it with the statistic's law when the null hypothesis holds: the p-value is the null probability of a value at least as extreme, and the test rejects at level α when that falls in the shaded region."
      state={state}
      defaultSize="L"
      caption={
        <>
          Pick a test and move its data parameters, or drag a data point (and, for one-sample tests, the null mean μ₀).
          The red region holds α of the null law; the blue area is the p-value. Exact tests (binomial, Fisher, the rank
          tests on small samples) have discrete null laws, so their regions hold at most α. The interval below the null
          law is at level 1 − α, so it excludes the null value exactly when a two-sided test rejects. Try a skewed
          population with small n: the t-test's nominal α no longer holds, while the rank tests keep it.
          {'error' in result && ` (${result.error})`}
        </>
      }
    >
      {'ok' in result && result.ok ? (
        <HypothesisTestPanel result={result.ok} alpha={alpha} axisKey={key} top={top} />
      ) : (
        <div className="text-sm text-muted-foreground">{'error' in result ? result.error : ''}</div>
      )}
    </Figure>
  )
}
