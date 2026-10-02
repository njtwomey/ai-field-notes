import { additiveData, type AdditiveFamily } from 'aifn-applied/data/synthetic'
import type { AdditiveTruth } from 'aifn-applied/data'
import {
  cyclic,
  gamModel,
  gamProblem,
  linearTerm,
  s,
  thinPlate,
  type GamFitMethod,
  type GamFitState,
  type GamModel,
  type GamProblemChoices,
  type GamRunRequest,
  type GamSpec,
  type GamTrainingRun,
  type ShapeConstraint,
  type SmoothingMethod,
  type TermSpec,
} from 'aifn-applied/learning/generalised/gam'
import { stream } from 'aifn/foundation/random'
import { fromData, linspace, toFlat, toRows, type Tensor } from 'aifn/foundation/tensor'
import { family as familyByName, type LinkName } from 'aifn/probability/likelihoods'
import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react'
import { Player } from '@lab/controls'
import { Dashboard, DashboardCell, DashboardRow, Equation, Figure, live, tex } from '@lab/layout'
import { call, choice, row, setting, slider, useComputed, useFigureState, variants, when, int } from '@lab/state'
import { Annotation, Area, Curve, Handle, Plot, Plots, Points, Raster, Readout, useAxis } from '@lab/viz'
import { formatValue } from '@lab/views'

const f3 = (v: number) =>
  !Number.isFinite(v) ? '—' : v !== 0 && Math.abs(v) < 1e-3 ? v.toExponential(2) : formatValue(Number(v.toPrecision(3)))
const GRID = linspace(0, 1, 101)
const GRID_X = toFlat(GRID)
/**
 * A value outside React props: the fixed λ, written by the figure and read by the criterion panel alone, so a λ drag
 * re-renders that panel and not every chart.
 */
type Store<T> = { get: () => T; set: (v: T) => void; subscribe: (f: () => void) => () => void }
function createStore<T>(initial: T): Store<T> {
  let value = initial
  const listeners = new Set<() => void>()
  return {
    get: () => value,
    set: (v) => {
      if (Object.is(v, value)) return
      value = v
      listeners.forEach((f) => f())
    },
    subscribe: (f) => {
      listeners.add(f)
      return () => listeners.delete(f)
    },
  }
}

/** The columns of an [m, c] tensor as c arrays. */
const columnsOf = (t: Tensor) => {
  const rows = toRows(t) as number[][]
  return rows[0].map((_, c) => rows.map((r) => r[c]))
}
const FEATURES = [0, 1, 2] as const
const FEATURE_LABELS = ['x₁', 'x₂', 'x₃']
const SHAPE_NAMES = ['periodic', 'monotone', 'wiggly']
const GAM = 'applied/learning/generalised/gam'

// Every family the data generator draws from, each with the links aifn says it takes (its default first).
const FAMILY_LIST: AdditiveFamily[] = ['gaussian', 'binomial', 'poisson', 'gamma']
const linksOf = (f: AdditiveFamily): LinkName[] => {
  const fam = familyByName(f)
  return [fam.defaultLink, ...fam.links.filter((l) => l !== fam.defaultLink)]
}
const LINKS = Object.fromEntries(FAMILY_LIST.map((f) => [f, linksOf(f)])) as Record<AdditiveFamily, LinkName[]>

const METHODS: { value: GamFitMethod; label: string }[] = [
  { value: 'p-irls', label: 'P-IRLS' },
  { value: 'backfitting', label: 'backfitting' },
  { value: 'gradient-descent', label: 'gradient descent' },
  { value: 'sgd', label: 'SGD' },
  { value: 'adam', label: 'Adam' },
  { value: 'lbfgs', label: 'L-BFGS' },
]
const METHOD_LABEL = Object.fromEntries(METHODS.map((m) => [m.value, m.label])) as Record<GamFitMethod, string>
// Palette slot per method (an entity), fixed so a method keeps its colour when it is the compared one.
const METHOD_SLOT: Record<GamFitMethod, number> = {
  'p-irls': 0,
  backfitting: 1,
  'gradient-descent': 2,
  sgd: 3,
  adam: 4,
  lbfgs: 5,
}
// The REML/GCV profile over a common log₁₀ λ.
const PROFILE_LOG10 = toFlat(linspace(-4, 6, 41))
const PROFILE_LN = PROFILE_LOG10.map((v) => v * Math.LN10)

const k = (initial: number) => slider(4, 30, initial, { label: 'basis size k', step: 1 })
// Degree and penalty order in one choice, so each term's controls fit one line.
const SPLINE_OPTIONS = [1, 2, 3, 4].flatMap((d) =>
  [1, 2, 3].map((o) => ({ value: `${d}-${o}`, label: `degree ${d}, order ${o}` })),
)
const spline = choice(SPLINE_OPTIONS, '3-2', { label: 'degree, penalty order' })
const termCases = (k0: number, initial: 'pspline' | 'cyclic') =>
  variants(
    {
      pspline: {
        label: 'P-spline',
        params: {
          k: k(k0),
          spline,
          shape: choice(['none', 'increasing', 'decreasing', 'convex', 'concave'], 'none', { label: 'shape' }),
        },
      },
      cyclic: { label: 'cyclic P-spline', params: { k: k(k0), spline } },
      thinplate: { label: 'thin plate', params: { k: k(k0) } },
      linear: { label: 'linear', params: {} },
      off: { label: 'off', params: {} },
    },
    { initial, choiceLabel: 'basis' },
  )

type TermChoice = { key: string; values: Record<string, unknown> }

/** The term spec for feature j from its row, or null when the feature is off. */
function termOf(j: number, t: TermChoice): TermSpec | null {
  const raw = t.values as { k?: number; spline?: string; shape?: string }
  const [degree, order] = (raw.spline ?? '3-2').split('-').map(Number)
  const v = { ...raw, degree, order }
  switch (t.key) {
    case 'pspline':
      return s(j, {
        k: v.k,
        degree: v.degree,
        order: v.order,
        constraint: v.shape === 'none' ? undefined : (v.shape as ShapeConstraint),
      })
    case 'cyclic':
      return cyclic(j, { range: [0, 1], k: v.k, degree: v.degree, order: v.order })
    case 'thinplate':
      return thinPlate(j, { k: v.k })
    case 'linear':
      return linearTerm(j)
    default:
      return null
  }
}

/** A run as it landed: the training result with the spec and data it was computed for. */
type Landed = { run: GamTrainingRun; spec: GamSpec; data: { x: Tensor; y: Tensor }; key: string; specKey: string }

/** The figure's controls, declared once (a stable schema lets the figure skip its rows when only results change). */
const SCHEMA = {
  data: variants(
    {
      gaussian: {
        label: 'Gaussian',
        params: { link: choice(LINKS.gaussian), noise: slider(0.05, 1.5, 0.4, { label: 'noise sd' }) },
      },
      binomial: { label: 'binomial', params: { link: choice(LINKS.binomial) } },
      poisson: { label: 'Poisson', params: { link: choice(LINKS.poisson) } },
      gamma: {
        label: 'gamma',
        params: { link: choice(LINKS.gamma), noise: slider(0.05, 1, 0.3, { label: 'coefficient of variation' }) },
      },
    },
    {
      label: '1 · data',
      choiceLabel: 'family',
      shared: {
        n: slider(50, 1000, 300, { label: 'n', step: 10 }),
        seed: slider(1, 20, 1, { label: 'seed', step: 1 }),
      },
    },
  ),
  x1: { ...termCases(10, 'cyclic'), label: '2 · x₁ (periodic)' },
  x2: { ...termCases(10, 'pspline'), label: '2 · x₂ (monotone)' },
  x3: { ...termCases(20, 'pspline'), label: '2 · x₃ (wiggly)' },
  smoothing: row('3 · smoothing', {
    method: choice(
      [
        { value: 'reml', label: 'REML' },
        { value: 'gcv', label: 'GCV / UBRE' },
        { value: 'fixed', label: 'fixed λ' },
      ],
      'reml',
      { label: 'λ chosen by' },
    ),
    logLambda: slider(-4, 6, 0, { label: 'log₁₀ λ (every smooth)', step: 0.1, when: when('method', 'fixed') }),
  }),
  training: variants(
    {
      'p-irls': { label: 'P-IRLS', params: {} },
      backfitting: { label: 'backfitting', params: {} },
      'gradient-descent': {
        label: 'gradient descent',
        params: { scale: slider(0.1, 2.4, 1, { label: 'step size × L', step: 0.05 }) },
      },
      sgd: {
        label: 'SGD',
        params: {
          batch: choice([8, 16, 32, 64, 128], 32, { label: 'batch size' }),
          scale: slider(0.05, 2.4, 0.5, { label: 'step size × L', step: 0.05 }),
        },
      },
      adam: { label: 'Adam', params: { logRate: slider(-3, 0, -1.3, { label: 'log₁₀ learning rate', step: 0.05 }) } },
      lbfgs: { label: 'L-BFGS', params: { memory: choice([3, 5, 10, 20], 10, { label: 'memory m' }) } },
    },
    {
      label: '4 · training',
      choiceLabel: 'method',
      initial: 'adam',
      shared: {
        steps: int(500, { ge: 1, suggestions: [20, 50, 100, 200, 500, 1000, 2000], label: 'max iterations' }),
        compare: choice([{ value: 'none', label: 'none' }, ...METHODS], 'p-irls', { label: 'compare with' }),
      },
    },
  ),
  view: row('5 · view', {
    term: choice(
      FEATURES.map((j) => ({ value: j, label: FEATURE_LABELS[j] })),
      2,
      { label: 'basis and slice of' },
    ),
    trace: choice(
      [
        { value: 'gap', label: 'J − J*' },
        { value: 'grad', label: '‖∇J‖' },
      ],
      'gap',
      { label: 'trace' },
    ),
    residuals: setting(true, 'partial residuals'),
  }),
}

export function GamShowcase() {
  const state = useFigureState(SCHEMA)

  // ── Data: drawn on the page (cheap); its truth gives the true effects. ───────────────────────────────────────────
  const family = state.data.key as AdditiveFamily
  const { link } = state.data.values as { link: LinkName }
  const noise = (state.data.values as { noise?: number }).noise
  const { n, seed } = state.data.values
  const dataKey = `${family}|${link}|${n}|${noise ?? ''}|${seed}`
  const dataset = useMemo(
    () => additiveData(stream(`gam-showcase-${seed}`), { family, link, n, noise }),
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed by value
    [dataKey],
  )
  const data = useMemo(() => ({ x: dataset.x as Tensor, y: dataset.y as Tensor }), [dataset])
  const truth = dataset.meta!.truth as AdditiveTruth

  // ── The model specification. ─────────────────────────────────────────────────────────────────────────────────────
  const termChoices = [state.x1, state.x2, state.x3] as TermChoice[]
  const termSpecs = termChoices.map((t, j) => termOf(j, t))
  const smoothingMethod = state.smoothing.method as SmoothingMethod
  const lambda = 10 ** state.smoothing.logLambda
  const specKey = JSON.stringify({
    dataKey,
    termSpecs,
    smoothingMethod,
    lambda: smoothingMethod === 'fixed' ? lambda : 0,
  })
  const spec = useMemo(
    (): GamSpec => ({
      terms: termSpecs.filter((t): t is TermSpec => t !== null),
      family,
      link,
      method: smoothingMethod,
      lambda,
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed by value
    [specKey],
  )
  const termsKey = termSpecs.map((t) => (t ? 1 : 0)).join('')
  const featureOfTerm = useMemo(() => termsKey.split('').flatMap((on, j) => (on === '1' ? [j] : [])), [termsKey])

  // ── Training: the problem (λ by REML/GCV) and every fitter's states, in a Web Worker. ────────────────────────────
  const method = state.training.key as GamFitMethod
  const tv = state.training.values as {
    steps: number
    compare: string
    scale?: number
    batch?: number
    logRate?: number
    memory?: number
  }
  const compare = tv.compare === 'none' || tv.compare === method ? null : (tv.compare as GamFitMethod)
  const requests = useMemo((): GamRunRequest[] => {
    const options =
      method === 'gradient-descent'
        ? { stepScale: tv.scale }
        : method === 'sgd'
          ? { stepScale: tv.scale, batchSize: tv.batch }
          : method === 'adam'
            ? { stepSize: 10 ** tv.logRate! }
            : method === 'lbfgs'
              ? { memory: tv.memory }
              : {}
    const main: GamRunRequest = { method, options, steps: tv.steps, seed: 'gam-showcase' }
    return compare ? [main, { method: compare, steps: tv.steps, seed: 'gam-showcase' }] : [main]
  }, [method, compare, tv.steps, tv.scale, tv.batch, tv.logRate, tv.memory])
  // λ found for a spec is reused when only the training changes, so switching method skips the REML search.
  const choicesFor = useRef(new Map<string, GamProblemChoices>())
  const training = useComputed(
    () => {
      const known = choicesFor.current.get(specKey)
      return call<GamTrainingRun>(`${GAM}/gamTrainingRun`, known ? { ...spec, ...known } : spec, data, requests)
    },
    [spec, data, requests, specKey],
    {
      mode: 'worker',
      initial: null as Landed | null,
      then: (run): Landed => ({ run, spec, data, key: `${specKey}|${JSON.stringify(requests)}`, specKey }),
      cancelAfter: 400,
    },
  )
  const landed = training.value
  useEffect(() => {
    if (landed) choicesFor.current.set(landed.specKey, landed.run.choices)
  }, [landed])

  // The criterion over a common λ (independent of the training and of λ itself), also in a worker.
  const criterion: 'reml' | 'gcv' = smoothingMethod === 'gcv' ? 'gcv' : 'reml'
  const profileKey = JSON.stringify({ dataKey, termSpecs, criterion })
  const profile = useComputed(
    () =>
      call<number[]>(
        `${GAM}/smoothingProfile`,
        call(`${GAM}/gamProblem`, { ...spec, method: 'fixed' }, data),
        PROFILE_LN,
        criterion,
      ),
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed by value
    [profileKey],
    { mode: 'worker', initial: null as number[] | null },
  )

  // ── The played step. ─────────────────────────────────────────────────────────────────────────────────────────────
  const states = landed?.run.runs[0].states ?? []
  const [step, setStep] = useState(0)
  const last = Math.max(0, states.length - 1)
  // A new run keeps the step; one at the end of the old run moves to the end of the new one.
  const atEnd = useRef(false)
  const shown = Math.min(step, last)
  useEffect(() => {
    if (atEnd.current && step !== last) setStep(last)
    // eslint-disable-next-line react-hooks/exhaustive-deps -- only when a new run lands
  }, [landed])
  useEffect(() => {
    atEnd.current = states.length > 0 && shown === last
  })

  return (
    <GamFigure
      state={state}
      landed={landed}
      stale={training.stale}
      error={training.error}
      truth={truth}
      dataset={dataset}
      featureOfTerm={featureOfTerm}
      profile={profile.value}
      profileStale={profile.stale}
      step={shown}
      setStep={setStep}
      count={states.length}
    />
  )
}

type FigureProps = {
  // oxlint-disable-next-line typescript/no-explicit-any -- the figure reads the typed values it needs
  state: any
  landed: Landed | null
  stale: boolean
  error?: string
  truth: AdditiveTruth
  dataset: ReturnType<typeof additiveData>
  featureOfTerm: number[]
  profile: number[] | null
  profileStale: boolean
  step: number
  setStep: (s: number) => void
  count: number
}

function GamFigureImpl(props: FigureProps) {
  const { state, landed, stale, error, truth, dataset, featureOfTerm, profile, profileStale, step, setStep, count } =
    props
  const view = state.view as {
    term: number
    trace: 'gap' | 'grad'
    residuals: boolean
  }
  const family = state.data.key as AdditiveFamily
  const dataKey = `${family}|${state.data.values.link}|${state.data.values.n}|${state.data.values.seed}`
  const smoothingMethod = state.smoothing.method as SmoothingMethod
  const method = state.training.key as GamFitMethod

  // The problem rebuilt on the page from the landed run's choices (no search), for the model at each step.
  const problem = useMemo(
    () => (landed ? gamProblem({ ...landed.spec, ...landed.run.choices }, landed.data) : null),
    [landed],
  )
  const states = landed?.run.runs[0].states
  const current: GamFitState | null = states ? states[Math.min(step, states.length - 1)] : null
  const model = useMemo(
    (): GamModel | null => (problem && current ? gamModel(problem, toFlat(current.coefficients)) : null),
    [problem, current],
  )

  // Columns of the data, for residuals and the response slice.
  const columns = useMemo(() => {
    const rows = toRows(dataset.x as Tensor) as number[][]
    return FEATURES.map((j) =>
      fromData(
        Float64Array.from(rows, (r) => r[j]),
        [rows.length],
      ),
    )
  }, [dataset])
  const y = useMemo(() => toFlat(dataset.y as Tensor), [dataset])

  // Per feature: the fitted effect at this step with ± 2 se, partial residuals, and the true effect centred as the fit is.
  const trueEffects = useMemo(() => FEATURES.map((j) => toFlat(truth.partial(j, GRID, columns[j]))), [truth, columns])
  const featureXs = useMemo(() => FEATURES.map((j) => toFlat(columns[j])), [columns])
  const effects = useMemo(
    () =>
      FEATURES.map((j) => {
        const t = featureOfTerm.indexOf(j)
        if (!model || t < 0) return { term: t, truth: trueEffects[j], fit: null }
        const p = model.partial(t, GRID)
        const f = toFlat(p.fit)
        const se = toFlat(p.se)
        return {
          term: t,
          truth: trueEffects[j],
          fit: f,
          upper: f.map((v, i) => v + 2 * se[i]),
          lower: f.map((v, i) => v - 2 * se[i]),
          residuals: toFlat(model.partialResiduals(t)),
          xs: featureXs[j],
        }
      }),
    [model, featureOfTerm, trueEffects, featureXs],
  )

  // The chosen feature's basis (raw, constrained, or weighted by the coefficients) and its penalty matrix.
  const j = view.term
  const basisTerm = featureOfTerm.indexOf(j)
  // The raw and constrained columns and the penalty depend on the term and the data only, not on λ or the step: they
  // are kept while a λ drag or the player changes the coefficients, so those charts are not redrawn.
  const termChoice = [state.x1, state.x2, state.x3][j] as TermChoice
  const termKey = `${dataKey}|${j}|${termChoice.key}|${JSON.stringify(termChoice.values)}|${basisTerm}`
  const modelRef = useRef(model)
  modelRef.current = model
  const hasModel = model !== null
  const basisFixed = useMemo(() => {
    const m = modelRef.current
    if (!m || basisTerm < 0 || m.terms[basisTerm].spec.kind !== 'smooth') return null
    const b = m.basis(basisTerm, GRID)
    const S = toRows(b.rawPenalties[0]) as number[][]
    return {
      raw: columnsOf(b.raw),
      constrained: columnsOf(b.constrained),
      penalty: S,
      index: S.map((_, i) => i + 1),
      rawSize: b.raw.shape[1],
      size: b.constrained.shape[1],
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed by the term and data, not by the coefficients
  }, [termKey, hasModel])
  const basisWeighted = useMemo(() => {
    if (!model || !basisFixed) return null
    const b = model.basis(basisTerm, GRID)
    return { lines: columnsOf(b.weighted), sum: toFlat(b.sum) }
  }, [model, basisFixed, basisTerm])
  const basis = useMemo(
    () =>
      basisFixed && {
        ...basisFixed,
        // Always the columns weighted by the played iteration's coefficients, and their sum: the term itself.
        lines: basisWeighted?.lines ?? [],
        sum: basisWeighted?.sum ?? null,
      },
    [basisFixed, basisWeighted],
  )

  // The response scale along feature j, the other features at ½: the fit at this step and the truth.
  const sliceRows = useMemo(
    () =>
      fromData(Float64Array.from(GRID_X.flatMap((v) => FEATURES.map((q) => (q === j ? v : 0.5)))), [GRID_X.length, 3]),
    [j],
  )
  const sliceFixed = useMemo(
    () => ({ truth: toFlat(truth.mean(sliceRows)), xs: toFlat(columns[j]) }),
    [truth, sliceRows, columns, j],
  )
  const sliceFit = useMemo(() => (model ? toFlat(model.expect(sliceRows)) : null), [model, sliceRows])
  const slice = useMemo(() => ({ ...sliceFixed, fit: sliceFit }), [sliceFixed, sliceFit])

  // The training traces: J − J* (floored at 1e-16) or ‖∇J‖ against the iteration.
  const traces = useMemo(() => {
    if (!landed) return []
    const best = landed.run.optimum.objective
    return landed.run.runs.map((r) => ({
      method: r.method,
      x: r.states.map((st) => st.t),
      y: r.states.map((st) =>
        view.trace === 'gap' ? Math.max(st.objective - best, 1e-16) : Math.max(st.gradNorm, 1e-16),
      ),
    }))
  }, [landed, view.trace])
  const marker = useMemo(
    () => (traces[0] ? { x: [traces[0].x[step]], y: [traces[0].y[step]] } : { x: [], y: [] }),
    [traces, step],
  )

  const [lambdaStore] = useState(() => createStore<number | null>(null))
  const fixed = smoothingMethod === 'fixed'
  const fixedLogLambda = fixed ? (state.smoothing.logLambda as number) : null
  useLayoutEffect(() => lambdaStore.set(fixedLogLambda), [lambdaStore, fixedLogLambda])
  const setLogLambda = useCallback(
    (x: number) => state.set('smoothing.logLambda', Math.max(-4, Math.min(6, x))),
    [state.set], // eslint-disable-line react-hooks/exhaustive-deps -- the setter is stable
  )
  // Each penalty's λ, labelled by the feature of the term it belongs to.
  const logLambdas = useMemo(
    () =>
      landed && problem
        ? landed.run.choices.lambdas.map((l, q) => ({
            at: Math.log10(l),
            label: `λ${featureOfTerm[problem.design.penalties[q].term] + 1}`,
          }))
        : [],
    [landed, problem, featureOfTerm],
  )

  // ── Equation and readouts. ───────────────────────────────────────────────────────────────────────────────────────
  const nObs = state.data.values.n as number
  const termNames = featureOfTerm.map((q) => `f_${q + 1}(x_${q + 1})`)
  const etaTex = termNames.length ? termNames.join(' + ') : '0'
  const pens = model?.termPenalties ?? []
  // KaTeX is rendered only when a value in the band changes (not on every control move).
  const intercept = model ? model.intercept : 0
  const devianceNow = model ? model.deviance : NaN
  const penaltyNow = pens.reduce((a, b) => a + b, 0)
  const objectiveNow = current?.objective ?? NaN
  const equation = useMemo(
    () => (
      <Equation>
        {tex`\begin{aligned} g(\mu) &= ${live(intercept, { digits: 3 })} + ${etaTex}, \qquad f_j = B_j\beta_j \\ J(\beta) &= \tfrac{1}{2n}\Big[D(\beta) + \textstyle\sum_j \lambda_j\,\beta_j^\top S_j\beta_j\Big] = \tfrac{1}{${2 * nObs}}\big[${live(devianceNow, { digits: 4 })} + ${live(penaltyNow, { digits: 3 })}\big] = ${live(objectiveNow, { digits: 5, strong: true })} \end{aligned}`}
      </Equation>
    ),
    [intercept, etaTex, nObs, devianceNow, penaltyNow, objectiveNow],
  )
  const run = landed?.run
  const converged = current?.converged ? 'yes' : current?.diverged ? 'diverged' : 'no'
  const firstConverged = states?.findIndex((st) => st.converged) ?? -1

  return (
    <Figure
      title="An interactive GAM: terms, links and training methods"
      purpose="Every fitter minimises the same penalised deviance at the same λ, so they reach one model; the basis, the penalty and the link fix what that model is, and the method only fixes how fast it is reached."
      state={state}
      defaultSize="XL"
      equation={equation}
      controls={
        <Player
          value={step}
          onChange={setStep}
          count={Math.max(1, count)}
          label="iteration"
          format={(p) => `iteration ${p}`}
        />
      }
      readouts={{
        [`at iteration ${current?.t ?? 0}`]: (
          <>
            <Readout label="J" value={f3(current?.objective ?? NaN)} />
            <Readout label="J − J*" value={run && current ? f3(current.objective - run.optimum.objective) : '—'} />
            <Readout label="‖∇J‖" value={f3(current?.gradNorm ?? NaN)} />
            <Readout
              label="deviance explained"
              value={model ? `${(100 * model.devianceExplained).toFixed(1)}%` : '—'}
            />
            <Readout label="EDF" value={model ? model.termEdf.map((e) => f3(e)).join(', ') : '—'} />
            {current?.stepSize !== undefined && Number.isFinite(current.stepSize) && (
              <Readout label="step size" value={f3(current.stepSize)} />
            )}
          </>
        ),
        smoothing: (
          <>
            <Readout label="λ per term" value={run ? run.choices.lambdas.map((l) => f3(l)).join(', ') : '—'} />
            {run && smoothingMethod !== 'fixed' && (
              <Readout
                label={smoothingMethod === 'reml' ? 'REML score' : 'GCV / UBRE'}
                value={f3(run.choices.smoothing.value)}
              />
            )}
            <Readout label="AIC" value={model ? f3(model.aic) : '—'} />
            <Readout label="curvature L" value={run ? f3(run.curvature) : '—'} />
          </>
        ),
        run: (
          <>
            <Readout label={METHOD_LABEL[method]} value={`${Math.max(0, count - 1)} iterations`} />
            <Readout label="converged" value={converged} />
            {firstConverged >= 0 && <Readout label="converged at" value={firstConverged} />}
            {(stale || profileStale || !landed) && (
              <Readout label="computing" value={<span aria-busy="true">in a worker…</span>} />
            )}
            {error && <Readout label="error" value={error} />}
          </>
        ),
      }}
      caption={`${dataset.meta!.description} Top: each term's partial effect at the played iteration with ± 2 se (from H = XᵀWX + S_λ at that β), the partial residuals, and the true effect (ink, dashed) centred on the data as the fit is. Middle: the chosen feature's basis columns weighted by their coefficients at the played iteration (thin), whose sum is the term (bold), its penalty matrix, and the response scale along that feature with the others at ½. Bottom: the training trace for the chosen method and the compared one (each at its default settings), and the ${smoothingMethod === 'gcv' ? 'GCV/UBRE' : 'REML'} criterion for one common λ; with “fixed λ” drag the λ line on it. Gradient descent and SGD take steps in units of 1/L for the curvature L of J at the optimum: above 2 they diverge.`}
    >
      <GamCharts
        effects={effects}
        basis={basis}
        j={j}
        step={current?.t ?? 0}
        residuals={view.residuals}
        slice={slice}
        y={y}
        traces={traces}
        marker={marker}
        traceKind={view.trace}
        runKey={landed?.key}
        count={count}
        profile={profile}
        profileStale={profileStale}
        stale={stale}
        dataKey={dataKey}
        smoothingMethod={smoothingMethod}
        logLambda={lambdaStore}
        setLogLambda={setLogLambda}
        logLambdas={logLambdas}
      />
    </Figure>
  )
}

const GamFigure = memo(GamFigureImpl)

type ChartsProps = {
  effects: {
    term: number
    truth: number[]
    fit: number[] | null
    upper?: number[]
    lower?: number[]
    residuals?: number[]
    xs?: number[]
  }[]
  basis: {
    lines: number[][]
    /** The raw B-spline columns, drawn faintly under the weighted ones. */
    raw: number[][]
    sum: number[] | null
    penalty: number[][]
    index: number[]
    rawSize: number
    size: number
  } | null
  j: number
  /** The played iteration (the weighted basis is that iteration's). */
  step: number
  residuals: boolean
  slice: { truth: number[]; fit: number[] | null; xs: number[] }
  y: number[]
  traces: { method: GamFitMethod; x: number[]; y: number[] }[]
  marker: { x: number[]; y: number[] }
  traceKind: 'gap' | 'grad'
  runKey: string | undefined
  count: number
  profile: number[] | null
  profileStale: boolean
  stale: boolean
  dataKey: string
  smoothingMethod: SmoothingMethod
  /** The fixed log₁₀ λ (dragged on the criterion curve), or null when λ is selected. */
  /** The fixed log₁₀ λ (null when λ is selected), in a store the criterion panel subscribes to. */
  logLambda: Store<number | null>
  setLogLambda: (x: number) => void
  logLambdas: { at: number; label: string }[]
}

/**
 * The charts, apart from the figure's controls and readouts: memoised on their data, so a control that changes none
 * of it (or a λ drag, which changes only the criterion panel) does not rebuild them.
 */
const GamCharts = memo(function GamCharts(props: ChartsProps) {
  const { effects, basis, j, residuals, slice, y, traces, marker, count, stale, dataKey, smoothingMethod } = props
  const view = { trace: props.traceKind }
  // ── Axes: held per data and family; refit when the entity changes, not when a parameter moves. ───────────────────
  const effectAxis = useAxis({ label: 'fⱼ (link scale)', hold: 'initial', key: dataKey })
  const xAxes = [
    useAxis({ label: 'x₁', range: [0, 1] }),
    useAxis({ label: 'x₂', range: [0, 1] }),
    useAxis({ label: 'x₃', range: [0, 1] }),
  ]
  const basisX = useAxis({ label: FEATURE_LABELS[j], range: [0, 1] })
  // The columns are live layers, which an axis does not fit to, so the range comes from the curves shown: the
  // weighted columns and their sum reach far beyond the raw basis's [0, 1]. Held as a union while the fit plays.
  const basisRange = useMemo<[number, number]>(() => {
    const all = [...(basis?.lines.flat() ?? []), ...(basis?.sum ?? [])]
    if (!all.length) return [0, 1]
    const [lo, hi] = [Math.min(...all), Math.max(...all)]
    const pad = 0.06 * (hi - lo || 1)
    return [lo - pad, hi + pad]
  }, [basis])
  const basisY = useAxis({
    label: 'basis value',
    range: basisRange,
    hold: 'union',
    key: `${dataKey}|${j}`,
  })
  const penaltyX = useAxis({ label: '', nice: false, key: `${j}|${basis?.rawSize}` })
  const penaltyY = useAxis({ label: '', nice: false, key: `${j}|${basis?.rawSize}` })
  const responseX = useAxis({ label: `${FEATURE_LABELS[j]} (others at ½)`, range: [0, 1] })
  const responseY = useAxis({ label: 'y and μ', hold: 'initial', key: `${dataKey}|${j}` })
  const iterAxis = useAxis({ label: 'iteration', range: [0, Math.max(1, count - 1)] })
  // Fixed decades, so the traces patch in place (live) as runs land during a drag.
  const traceY = useAxis({
    label: view.trace === 'gap' ? 'J − J*' : '‖∇J‖',
    log: true,
    range: view.trace === 'gap' ? [1e-16, 100] : [1e-12, 100],
    key: view.trace,
  })

  return (
    <Dashboard>
      <DashboardRow ratio={1.1} minHeight={210}>
        <DashboardCell>
          <Plots cols={3}>
            {effects.map((e, q) => (
              <Plot
                key={q}
                x={xAxes[q]}
                y={effectAxis}
                legend={q === 0}
                title={`${FEATURE_LABELS[q]}: ${SHAPE_NAMES[q]}${e.term < 0 ? ' (off)' : ''}`}
              >
                {e.fit && residuals && (
                  <Points name="partial residuals" x={e.xs!} y={e.residuals!} muted size={3} live />
                )}
                {e.fit && (
                  <Area
                    name="± 2 se"
                    x={GRID_X}
                    y={e.upper!}
                    base={e.lower!}
                    slot={q}
                    opacity={0.2}
                    line={false}
                    stale={stale}
                    live
                  />
                )}
                <Curve name="true effect" x={GRID_X} y={e.truth} emphasis dashed />
                {e.fit && <Curve name="fitted effect" x={GRID_X} y={e.fit} slot={q} stale={stale} live />}
              </Plot>
            ))}
          </Plots>
        </DashboardCell>
      </DashboardRow>
      <DashboardRow ratio={1} minHeight={200}>
        <DashboardCell ratio={1.5}>
          <Plot
            x={basisX}
            y={basisY}
            title={
              !basis
                ? `basis of ${FEATURE_LABELS[j]}: not a smooth`
                : `${FEATURE_LABELS[j]}: basis columns × β at iteration ${props.step}${props.step === 0 ? ' (β = 0: press play)' : ''}`
            }
            legend={false}
          >
            {basis?.lines.map((c, i) => (
              <Curve key={i} name={`column ${i + 1}`} x={GRID_X} y={c} slot={j} thin live />
            ))}
            {basis?.sum && <Curve name="sum: fⱼ" x={GRID_X} y={basis.sum} emphasis live />}
          </Plot>
        </DashboardCell>
        <DashboardCell aspect="square">
          {basis ? (
            <Plot x={penaltyX} y={penaltyY} title="penalty S (raw)">
              <Raster x={basis.index} y={basis.index} z={basis.penalty} scale="diverging" valueLabel="S" />
            </Plot>
          ) : (
            <div />
          )}
        </DashboardCell>
        <DashboardCell ratio={1.2}>
          <Plot x={responseX} y={responseY} title="response scale">
            <Points name="data" x={slice.xs} y={y} muted size={3} />
            <Curve name="true mean" x={GRID_X} y={slice.truth} emphasis dashed />
            {slice.fit && <Curve name="fitted mean" x={GRID_X} y={slice.fit} slot={j} stale={stale} live />}
          </Plot>
        </DashboardCell>
      </DashboardRow>
      <DashboardRow ratio={0.9} minHeight={180}>
        <DashboardCell ratio={1.6}>
          <Plot x={iterAxis} y={traceY} title="training trace">
            {traces.map((t) => (
              <Curve
                key={t.method}
                name={METHOD_LABEL[t.method]}
                x={t.x}
                y={t.y}
                slot={METHOD_SLOT[t.method]}
                showPoints={t.x.length <= 60}
                stale={stale}
                live
              />
            ))}
            <Points name="this iteration" x={marker.x} y={marker.y} emphasis live />
          </Plot>
        </DashboardCell>
        <DashboardCell>
          <CriterionPlot
            profile={props.profile}
            profileStale={props.profileStale}
            smoothingMethod={smoothingMethod}
            dataKey={dataKey}
            logLambda={props.logLambda}
            setLogLambda={props.setLogLambda}
            logLambdas={props.logLambdas}
          />
        </DashboardCell>
      </DashboardRow>
    </Dashboard>
  )
})

const CriterionPlot = memo(function CriterionPlot({
  profile,
  profileStale,
  smoothingMethod,
  dataKey,
  logLambda,
  setLogLambda,
  logLambdas,
}: Pick<
  ChartsProps,
  'profile' | 'profileStale' | 'smoothingMethod' | 'dataKey' | 'logLambda' | 'setLogLambda' | 'logLambdas'
>) {
  const at = useSyncExternalStore(logLambda.subscribe, logLambda.get, logLambda.get)
  const lamAxis = useAxis({ label: 'log₁₀ λ (every smooth)', range: [-4, 6] })
  const critAxis = useAxis({
    label: smoothingMethod === 'gcv' ? 'GCV / UBRE' : 'REML criterion',
    key: `${dataKey}|${smoothingMethod === 'gcv'}`,
  })
  return (
    <Plot x={lamAxis} y={critAxis} title={`${smoothingMethod === 'gcv' ? 'GCV / UBRE' : 'REML'} against a common λ`}>
      {profile && <Curve name="criterion" x={PROFILE_LOG10} y={profile} slot={6} stale={profileStale} />}
      {at !== null ? (
        <Handle kind="x" at={at} label="λ" onDrag={setLogLambda} />
      ) : (
        logLambdas.map((l, q) => <Annotation key={q} x={l.at} text={l.label} dashed />)
      )}
    </Plot>
  )
})
