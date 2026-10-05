import type { Curve1dTruth } from 'aifn-methods/data'
import { CURVE1D_CASES, curve1d, type Curve1dCase } from 'aifn-methods/data/synthetic'
import {
  expectileProblem,
  gamLinkBand,
  gamModel,
  gamProblem,
  s,
  type ExpectileFitMethod,
  type ExpectileRunRequest,
  type ExpectileTrainingRun,
  type GamFitState,
  type GamModel,
  type GamProblem,
  type GamRunRequest,
  type GamSpec,
  type GamTrainingRun,
  type SmoothingMethod,
  type SmoothingPath,
  type TermSpec,
} from 'aifn-methods/learning/generalised/gam'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, linspace, reshape, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import {
  family as familyByName,
  link as linkByName,
  type FamilyName,
  type LinkName,
} from 'aifn-compute/probability/likelihoods'
import {
  memo,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
  type ReactNode,
} from 'react'
import { Player } from 'aifn-render/controls'
import { Dashboard, DashboardCell, DashboardRow, Equation, Figure, live, tex } from 'aifn-render/layout'
import {
  call,
  choice,
  row,
  setting,
  slider,
  toggle,
  useComputed,
  useFigureState,
  variants,
  when,
  int,
} from 'aifn-render/state'
import {
  Area,
  Curve,
  Handle,
  Plot,
  Plots,
  Points,
  Readout,
  useAxis,
  useScaleColor,
  type AxisModel,
} from 'aifn-render/viz'
import { formatValue } from '@lab/views'

const GAM = 'applied/learning/generalised/gam'
const f3 = (v: number) =>
  !Number.isFinite(v) ? '—' : v !== 0 && Math.abs(v) < 1e-3 ? v.toExponential(2) : formatValue(Number(v.toPrecision(3)))
const pct = (v: number) => (Number.isFinite(v) ? `${(100 * v).toFixed(1)}%` : '—')

// Inputs on which curves are drawn, and the λ the criterion panel uses.
const GRID_X = toFlat(linspace(0, 1, 201))
const GRID = reshape(fromData(Float64Array.from(GRID_X), [GRID_X.length]), [GRID_X.length, 1])
const NO_TAUS: number[] = []
const PATH_LOG10 = toFlat(linspace(-6, 6, 61))
const PATH_LN = PATH_LOG10.map((v) => v * Math.LN10)

// The cases whose response is not continuous on the real line: a GLM with a link is their natural model.
const GLM = new Set<string>(['counts', 'binary', 'gamma'])
const FAMILY_LABEL: Partial<Record<FamilyName, string>> = { poisson: 'Poisson', binomial: 'binomial', gamma: 'gamma' }
/**
 * The fits offered for a case: its family with every link aifn says it takes (the case's own link first, the default),
 * and a plain Gaussian smoother.
 */
const fitChoice = (which: Curve1dCase) => {
  const { family: natural, link: own } = CURVE1D_CASES[which]
  const links: LinkName[] = [own, ...familyByName(natural).links.filter((l) => l !== own)]
  return choice(
    [
      ...links.map((l) => ({ value: `${natural}:${l}`, label: `${FAMILY_LABEL[natural]}, ${l} link` })),
      { value: 'gaussian:identity', label: 'Gaussian, identity link' },
    ],
    `${natural}:${own}`,
    { label: 'fitted family, link' },
  )
}
const NOISE = [
  { value: 'normal', label: 'normal (symmetric)' },
  { value: 'skewed', label: 'skewed (shifted log-normal)' },
] as const

const METHOD_LABEL: Record<string, string> = {
  'p-irls': 'P-IRLS (LAWS)',
  'gradient-descent': 'gradient descent',
  sgd: 'SGD',
  adam: 'Adam',
  lbfgs: 'L-BFGS',
}

/** The figure's controls, declared once. */
const SCHEMA = {
  data: variants(
    {
      sine: {
        label: 'sine, noise spreads and pinches',
        params: {
          noiseShape: choice(NOISE, 'normal', { label: 'noise' }),
          hetero: slider(0, 4, 2, { label: 'heteroscedasticity h', step: 0.1 }),
        },
      },
      skewed: {
        label: 'bump, skewed noise',
        params: {
          noiseShape: choice(NOISE, 'skewed', { label: 'noise' }),
          hetero: slider(0, 4, 2, { label: 'heteroscedasticity h', step: 0.1 }),
        },
      },
      counts: { label: 'Poisson counts', params: { model: fitChoice('counts') } },
      binary: { label: 'binary outcomes', params: { model: fitChoice('binary') } },
      gamma: {
        label: 'gamma responses',
        params: {
          model: fitChoice('gamma'),
          cv: slider(0.2, 1, 0.5, { label: 'coefficient of variation', step: 0.05 }),
        },
      },
    },
    {
      label: '1 · data',
      choiceLabel: 'case',
      shared: {
        n: slider(50, 1000, 300, { label: 'n', step: 10 }),
        seed: slider(1, 20, 1, { label: 'seed', step: 1 }),
      },
    },
  ),
  smooth: row('2 · P-spline', {
    k: slider(4, 40, 20, { label: 'basis size k', step: 1 }),
    degree: choice([1, 2, 3, 4], 3, { label: 'B-spline degree' }),
    order: choice([1, 2, 3], 2, { label: 'penalty order' }),
    raw: setting(false, 'raw basis'),
  }),
  smoothing: row('3 · smoothing', {
    method: choice(
      [
        { value: 'reml', label: 'REML' },
        { value: 'gcv', label: 'GCV' },
        { value: 'fixed', label: 'fixed λ' },
      ],
      'reml',
      { label: 'λ chosen by' },
    ),
    logLambda: slider(-6, 6, 0, { label: 'log₁₀ λ', step: 0.05, when: when('method', 'fixed') }),
  }),
  training: variants(
    {
      'p-irls': { label: 'P-IRLS (LAWS)', params: {} },
      'gradient-descent': {
        label: 'gradient descent',
        params: { scale: slider(0.1, 2.4, 1, { label: 'step size × L', step: 0.05 }) },
      },
      sgd: {
        label: 'SGD',
        params: {
          batch: int(32, { ge: 1, suggestions: [8, 16, 32, 64, 128], label: 'batch size' }),
          scale: slider(0.05, 2.4, 0.5, { label: 'step size × L', step: 0.05 }),
        },
      },
      adam: { label: 'Adam', params: { logRate: slider(-3, 0, -1.3, { label: 'log₁₀ learning rate', step: 0.05 }) } },
      lbfgs: {
        label: 'L-BFGS',
        params: { memory: int(10, { ge: 1, le: 50, suggestions: [3, 5, 10, 20], label: 'memory m' }) },
      },
    },
    {
      label: '4 · training',
      choiceLabel: 'method',
      initial: 'adam',
      shared: {
        steps: int(500, { ge: 1, suggestions: [20, 50, 100, 200, 500, 1000, 2000], label: 'max iterations' }),
      },
    },
  ),
  respExpectiles: toggle(false, { label: 'expectile band of y', when: (v) => GLM.has(v.data as string) }),
  band: {
    ...row('5 · expectile band', {
      level: slider(0.5, 0.99, 0.9, { label: 'level', step: 0.01 }),
      edge: choice(
        [
          { value: 'hi', label: 'upper edge τ_hi' },
          { value: 'lo', label: 'lower edge τ_lo' },
        ],
        'hi',
        { label: 'basis, λ and criterion of' },
      ),
    }),
    when: (v: Record<string, unknown>) => !GLM.has(v.data as string) || v.respExpectiles === true,
  },
}

/**
 * A value outside React props: the λ line's position, written by the figure and read by the criterion panel alone, so
 * a λ drag re-renders that panel and not every chart.
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

type Data = { x: Tensor; y: Tensor }
type FitState = GamFitState & { below?: number }

/**
 * One trained model as the page plays it: its states, the reference minimum J*, λ and the criterion where λ settled,
 * and its problem, so the model at any step is `gamModel(problem, β)`.
 */
type Trained = {
  /** τ for an expectile; null for the family GAM's mean. */
  tau: number | null
  states: FitState[]
  best: number
  lambdas: number[]
  criterion: number
  problem: GamProblem
  data: Data
}

/** The τ-expectile problem at the λ LAWS settled on (the problem the chosen method minimised). */
const expectileProblemOf = (terms: readonly TermSpec[], data: Data, tau: number, lambdas: number[]) =>
  expectileProblem(gamProblem({ terms, family: 'gaussian', lambdas }, data), tau)

/** The model at the played step (the last state once a run has finished). */
function modelAt(t: Trained | null, step: number): { model: GamModel | null; state: FitState | null } {
  if (!t || !t.states.length) return { model: null, state: null }
  const state = t.states[Math.min(step, t.states.length - 1)]
  try {
    return { model: gamModel(t.problem, toFlat(state.coefficients)), state }
  } catch {
    return { model: null, state }
  }
}

export function ExpectileShowcase() {
  const state = useFigureState(SCHEMA)

  // ── Data: drawn on the page; its truth knows the whole law, so the true expectiles. ──────────────────────────────
  const which = state.data.key as Curve1dCase
  const dv = state.data.values as {
    n: number
    seed: number
    noiseShape?: 'normal' | 'skewed'
    hetero?: number
    model?: string
    cv?: number
  }
  const glm = GLM.has(which)
  const dataKey = [which, dv.n, dv.seed, dv.noiseShape ?? '', dv.hetero ?? '', dv.cv ?? ''].join('|')
  const dataset = useMemo(
    () =>
      curve1d(stream(`gam-expectile-${dv.seed}`), {
        case: which,
        n: dv.n,
        noiseShape: dv.noiseShape,
        heteroscedastic: dv.hetero,
        noise: dv.cv,
      }),
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed by value
    [dataKey],
  )
  const data = useMemo((): Data => ({ x: dataset.x as Tensor, y: dataset.y as Tensor }), [dataset])
  const truth = dataset.meta!.truth as Curve1dTruth

  // ── The smooth and λ. ────────────────────────────────────────────────────────────────────────────────────────────
  const { k, degree, order } = state.smooth
  const method = state.smoothing.method as SmoothingMethod
  const logLambda = state.smoothing.logLambda
  const fixed = method === 'fixed'
  const termsKey = `${k}|${degree}|${order}`
  // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed by value
  const terms = useMemo(() => [s(0, { k, degree, order })], [termsKey])
  const lambda = 10 ** logLambda
  const params = useMemo(
    () => ({ terms, method, lambda }),
    // eslint-disable-next-line react-hooks/exhaustive-deps -- λ matters only when fixed
    [terms, method, fixed ? lambda : 0],
  )
  const [famName, linkName] = (glm ? (dv.model ?? 'gaussian:identity') : 'gaussian:identity').split(':') as [
    FamilyName,
    LinkName,
  ]
  const showExp = !glm || state.respExpectiles
  const level = state.band.level as number
  const edge = state.band.edge as 'lo' | 'hi'
  // The band's levels: τ_lo = (1 − level)/2 and τ_hi = 1 − τ_lo, rounded to drop float noise.
  const tauLo = Math.round(((1 - level) / 2) * 1e6) / 1e6
  const tauHi = Math.round((1 - tauLo) * 1e6) / 1e6

  // ── The training request: one method. ──────────────────────────────────────────────────────────────────────────
  const fitMethod = state.training.key as ExpectileFitMethod
  const tv = state.training.values as {
    steps: number
    scale?: number
    batch?: number
    logRate?: number
    memory?: number
  }
  const requests = useMemo((): ExpectileRunRequest[] => {
    const options =
      fitMethod === 'gradient-descent'
        ? { stepScale: tv.scale }
        : fitMethod === 'sgd'
          ? { stepScale: tv.scale, batchSize: tv.batch }
          : fitMethod === 'adam'
            ? { stepSize: 10 ** tv.logRate! }
            : fitMethod === 'lbfgs'
              ? { memory: tv.memory }
              : {}
    return [{ method: fitMethod, options, steps: tv.steps, seed: 'gam-expectile' }]
  }, [fitMethod, tv.steps, tv.scale, tv.batch, tv.logRate, tv.memory])

  // ── Training, in workers: the two expectile models of the band, or the family GAM. ───────────────────────────────
  const taus = useMemo(() => (showExp ? [tauLo, tauHi] : NO_TAUS), [showExp, tauLo, tauHi])
  const bandRun = useComputed(
    () => call<ExpectileTrainingRun>(`${GAM}/expectileTrainingRun`, params, data, taus, requests),
    [params, data, taus, requests],
    {
      mode: 'worker',
      initial: null as Trained[] | null,
      then: (result): Trained[] =>
        result.runs.map((r) => ({
          tau: r.tau,
          states: r.fits[0].states,
          best: r.optimum.objective,
          lambdas: r.lambdas,
          criterion: r.criterion,
          problem: expectileProblemOf(params.terms, data, r.tau, r.lambdas),
          data,
        })),
    },
  )
  const familySpec = useMemo(
    (): GamSpec => ({ ...params, family: famName, link: linkName }),
    [params, famName, linkName],
  )
  const familyRequests = useMemo(() => (showExp ? [] : requests) as GamRunRequest[], [showExp, requests])
  const familyRun = useComputed(
    () => call<GamTrainingRun>(`${GAM}/gamTrainingRun`, familySpec, data, familyRequests),
    [familySpec, data, familyRequests],
    {
      mode: 'worker',
      initial: null as Trained | null,
      then: (run): Trained | null =>
        run.runs[0]
          ? {
              tau: null,
              states: run.runs[0].states,
              best: run.optimum.objective,
              lambdas: run.choices.lambdas,
              criterion: run.choices.smoothing.value,
              problem: gamProblem({ ...familySpec, ...run.choices }, data),
              data,
            }
          : null,
    },
  )
  const models = useMemo(
    (): Trained[] => (showExp ? (bandRun.value ?? []) : familyRun.value ? [familyRun.value] : []),
    [showExp, bandRun.value, familyRun.value],
  )
  // The model the hyperparameter panels describe: the chosen edge of the band, or the family GAM.
  const subject = showExp ? (models[edge === 'lo' ? 0 : 1] ?? null) : (models[0] ?? null)

  // ── The played iteration (both models step together; a finished one stays at its last state). ──────────────────
  const count = models.reduce((m, t) => Math.max(m, t.states.length), 0)
  const [step, setStep] = useState(0)
  const last = Math.max(0, count - 1)
  const shown = Math.min(step, last)
  // A new run keeps the step; one at the end of the old run moves to the end of the new one.
  const atEnd = useRef(false)
  useEffect(() => {
    if (atEnd.current && step !== last) setStep(last)
    // eslint-disable-next-line react-hooks/exhaustive-deps -- only when a new run lands
  }, [models])
  useEffect(() => {
    atEnd.current = count > 1 && shown === last
  })

  // ── The criterion and EDF against a common λ, for the subject. ───────────────────────────────────────────────────
  // For an expectile, at its asymmetric weights at the optimum, where λ settled (held while λ is fixed, so a drag does
  // not move the curve).
  const optimumWeights = useMemo(() => {
    if (!showExp || !subject) return null
    const W = subject.problem.working(subject.problem.optimum.beta).W
    return { weights: fromData(W, [W.length]), data: subject.data, tau: subject.tau }
  }, [showExp, subject])
  const [held, setHeld] = useState<{ weights: Tensor; data: Data; tau: number | null } | null>(null)
  useEffect(() => {
    if (optimumWeights && (!fixed || !held || held.data !== optimumWeights.data || held.tau !== optimumWeights.tau))
      setHeld(optimumWeights)
    // eslint-disable-next-line react-hooks/exhaustive-deps -- held while λ is fixed
  }, [optimumWeights, fixed])
  const criterion: 'reml' | 'gcv' = method === 'gcv' ? 'gcv' : 'reml'
  // Weights belong to the data they were fitted on; until the run on new data lands, the path is unweighted.
  const pathWeights = showExp && held && held.data === data ? held.weights : null
  const pathRun = useComputed(
    () => {
      const spec: GamSpec = showExp
        ? { terms, family: 'gaussian', method: 'fixed' }
        : { terms, family: famName, link: linkName, method: 'fixed' }
      const d = pathWeights ? { ...data, weights: pathWeights } : data
      return call<SmoothingPath>(`${GAM}/smoothingPath`, call(`${GAM}/gamProblem`, spec, d), PATH_LN, criterion)
    },
    [terms, data, showExp, famName, linkName, pathWeights, criterion],
    { mode: 'worker', initial: null as SmoothingPath | null },
  )

  const setLogLambda = useCallback(
    (x: number) => {
      if (state.smoothing.method !== 'fixed') state.set('smoothing.method', 'fixed')
      state.set('smoothing.logLambda', Math.max(-6, Math.min(6, x)))
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps -- the setter is stable
    [state.set, state.smoothing.method],
  )

  const stale = showExp ? bandRun.stale : familyRun.stale
  const error = (showExp ? bandRun.error : familyRun.error) ?? pathRun.error

  return (
    <ExpectileFigure
      state={state}
      glm={glm}
      showExp={showExp}
      dataKey={dataKey}
      truth={truth}
      dataset={dataset}
      models={models}
      subject={subject}
      tauLo={tauLo}
      tauHi={tauHi}
      step={shown}
      setStep={setStep}
      count={count}
      path={pathRun.value}
      pathStale={pathRun.stale}
      setLogLambda={setLogLambda}
      stale={stale}
      error={error}
    />
  )
}

type FigureProps = {
  // oxlint-disable-next-line typescript/no-explicit-any -- the figure reads the typed values it needs
  state: any
  glm: boolean
  showExp: boolean
  dataKey: string
  truth: Curve1dTruth
  dataset: ReturnType<typeof curve1d>
  models: Trained[]
  subject: Trained | null
  tauLo: number
  tauHi: number
  step: number
  setStep: (s: number) => void
  count: number
  path: SmoothingPath | null
  pathStale: boolean
  setLogLambda: (x: number) => void
  stale: boolean
  error?: string
}

const ExpectileFigure = memo(function ExpectileFigure(props: FigureProps) {
  const { state, glm, showExp, dataKey, truth, dataset, models, subject, tauLo, tauHi, step, setStep, count } = props
  const { path, stale } = props
  const method = state.smoothing.method as SmoothingMethod
  const fixed = method === 'fixed'
  const rawBasis = state.smooth.raw as boolean
  const degree = state.smooth.degree as number
  const order = state.smooth.order as number
  const level = state.band.level as number
  const fitMethod = state.training.key as ExpectileFitMethod
  const linkName = (glm ? ((state.data.values.model as string) ?? 'gaussian:identity') : 'gaussian:identity').split(
    ':',
  )[1] as LinkName
  const tauColor = useScaleColor('sequential')
  // The two edges in an ordered colour scale (by τ), clear of its palest end.
  const colorOf = useCallback((t: number) => tauColor(0.45 + 0.55 * t), [tauColor])

  const x = useMemo(() => toFlat(dataset.x as Tensor), [dataset])
  const y = useMemo(() => toFlat(dataset.y as Tensor), [dataset])

  // ── The truth: mean and link scale per dataset; the true expectile band per level. ───────────────────────────────
  const truthFixed = useMemo(
    // The true mean on the fitted link's scale (the data's own link need not be the one fitted).
    () => {
      const mean = truth.mean(GRID)
      return { mean: toFlat(mean), eta: toFlat(linkByName(linkName).link(mean) as Tensor) }
    },
    [truth, linkName],
  )
  const truthBand = useMemo(
    () =>
      showExp
        ? {
            lo: toFlat(truth.expectile(GRID, tauLo)),
            hi: toFlat(truth.expectile(GRID, tauHi)),
            inside: truth.shareBelow(tauHi) - truth.shareBelow(tauLo),
          }
        : null,
    [truth, tauLo, tauHi, showExp],
  )

  // ── Each model at the played iteration. ──────────────────────────────────────────────────────────────────────────
  const played = useMemo(() => models.map((t) => ({ trained: t, ...modelAt(t, step) })), [models, step])
  const subjectPlayed = useMemo(() => played.find((p) => p.trained === subject) ?? null, [played, subject])
  const subjectModel = subjectPlayed?.model ?? null
  const subjectState = subjectPlayed?.state ?? null

  // The band: both edges on the grid, and the share of points inside it.
  const band = useMemo(() => {
    if (!showExp || played.length < 2 || !played[0].model || !played[1].model) return null
    const lo = toFlat(played[0].model.decide(GRID))
    const hi = toFlat(played[1].model.decide(GRID))
    // Inside the band at the data: below the upper edge and not below the lower one (the states' shares below).
    const inside = (played[1].state?.below ?? NaN) - (played[0].state?.below ?? NaN)
    return { lo, hi, inside, crossed: lo.some((v, i) => v > hi[i]) }
  }, [showExp, played])

  // ── The family GAM: the mean on the response scale with its band, and the link scale. ───────────────────────────
  const familyCurves = useMemo(() => {
    if (showExp || !subjectModel) return null
    const b = gamLinkBand(subjectModel, GRID)
    const eta = toFlat(b.fit)
    const se = toFlat(b.se)
    const lowerEta = eta.map((v, i) => v - 2 * se[i])
    const upperEta = eta.map((v, i) => v + 2 * se[i])
    const inv = (a: number[]) => toFlat(subjectModel.link.inverse(fromData(Float64Array.from(a), [a.length])) as Tensor)
    const lowerMu = inv(lowerEta)
    const upperMu = inv(upperEta)
    return {
      eta,
      lowerEta,
      upperEta,
      mu: inv(eta),
      // A decreasing link (inverse) swaps the ends of the band.
      lowerMu: lowerMu.map((v, i) => Math.min(v, upperMu[i])),
      upperMu: upperMu.map((v, i) => Math.max(v, lowerMu[i])),
    }
  }, [subjectModel, showExp])

  // ── The basis of the subject and the training traces. ──────────────────────────────────────────────────────────
  const basis = useMemo(() => {
    if (!subjectModel) return null
    const b = subjectModel.basis(0, GRID)
    const cols = (t: Tensor) => {
      const rows = t.shape[0]
      const c = t.shape[1]
      const flat = toFlat(t)
      return Array.from({ length: c }, (_, j) => Array.from({ length: rows }, (_, i) => flat[i * c + j]))
    }
    return { lines: cols(rawBasis ? b.raw : b.weighted), sum: rawBasis ? null : toFlat(b.sum), size: b.raw.shape[1] }
  }, [subjectModel, rawBasis])
  const traces = useMemo(
    () =>
      models.map((t) => ({
        tau: t.tau,
        x: t.states.map((st) => st.t),
        y: t.states.map((st) => Math.max(st.objective - t.best, 1e-16)),
      })),
    [models],
  )
  const markers = useMemo(
    () =>
      traces.map((tr) => {
        const i = Math.min(step, tr.x.length - 1)
        return i >= 0 ? { x: [tr.x[i]], y: [tr.y[i]] } : { x: [], y: [] }
      }),
    [traces, step],
  )

  const subjectLambda = subject?.lambdas[0] ?? NaN
  const handleAt = fixed ? (state.smoothing.logLambda as number) : Math.log10(subjectLambda)
  const [handleStore] = useState(() => createStore(handleAt))
  useLayoutEffect(() => handleStore.set(handleAt), [handleStore, handleAt])

  // ── Equation band: the loss and the objective of the subject at the played iteration. ───────────────────────────
  const dev = subjectModel?.deviance ?? NaN
  const pen = subjectModel?.termPenalties[0] ?? NaN
  const nObs = state.data.values.n as number
  const subjectTau = subject?.tau ?? NaN
  const equation = useMemo(
    () => (
      <Equation>
        {showExp
          ? tex`\begin{aligned} \rho_\tau(u) &= \lvert\tau - \mathbb{1}[u < 0]\rvert\,u^2, \qquad \tau_{\mathrm{lo}} = \tfrac{1 - ${live(level, { digits: 2 })}}{2} = ${live(tauLo, { digits: 3 })},\ \ \tau_{\mathrm{hi}} = ${live(tauHi, { digits: 3 })} \\ J_{${subjectTau}}(\beta) &= \tfrac{1}{2n}\Big[\textstyle\sum_i \rho_\tau(y_i - \mathbf{b}_i^\top\beta) + \lambda\,\beta^\top S\beta\Big] = \tfrac{1}{${2 * nObs}}\big[${live(dev, { digits: 4 })} + ${live(pen, { digits: 3 })}\big] = ${live((dev + pen) / (2 * nObs), { digits: 5, strong: true })} \end{aligned}`
          : tex`\begin{aligned} g(\mu) &= \alpha + f(x), \qquad f = \textstyle\sum_j \beta_j B_j(x) \\ J(\beta) &= \tfrac{1}{2n}\big[D(\beta) + \lambda\,\beta^\top S\beta\big] = \tfrac{1}{${2 * nObs}}\big[${live(dev, { digits: 4 })} + ${live(pen, { digits: 3 })}\big] = ${live((dev + pen) / (2 * nObs), { digits: 5, strong: true })} \end{aligned}`}
      </Equation>
    ),
    [showExp, level, tauLo, tauHi, subjectTau, dev, pen, nObs],
  )

  const criterionName = method === 'gcv' ? 'GCV' : 'REML'
  const methodName = METHOD_LABEL[fitMethod] ?? fitMethod
  const readouts: Record<string, ReactNode> = {}
  if (showExp)
    readouts[`band at iteration ${step}`] = (
      <>
        <Readout label="τ_lo" value={tauLo} color={colorOf(tauLo)} />
        <Readout label="τ_hi" value={tauHi} color={colorOf(tauHi)} />
        <Readout label="points inside" value={band ? pct(band.inside) : '—'} />
        <Readout label="nominal level" value={pct(level)} />
        <Readout label="population inside the true band" value={pct(truthBand?.inside ?? NaN)} />
        {band?.crossed && <Readout label="edges" value="cross" />}
      </>
    )
  readouts[`${showExp ? `edge τ = ${subjectTau}` : 'fit'}, ${methodName}, iteration ${subjectState?.t ?? 0}`] = (
    <>
      <Readout label="J" value={f3(subjectState?.objective ?? NaN)} />
      <Readout label="J − J*" value={subject && subjectState ? f3(subjectState.objective - subject.best) : '—'} />
      <Readout label="‖∇J‖" value={f3(subjectState?.gradNorm ?? NaN)} />
      <Readout label="EDF" value={f3(subjectModel?.edf ?? NaN)} />
      <Readout label="λ" value={f3(subjectLambda)} />
      {!fixed && <Readout label={criterionName} value={f3(subject?.criterion ?? NaN)} />}
      {!showExp && subjectModel && <Readout label="deviance explained" value={pct(subjectModel.devianceExplained)} />}
    </>
  )
  if (showExp)
    readouts.note = (
      <span className="text-muted-foreground">
        An expectile band is not a quantile interval: its edges balance squared shortfalls, not counts, so the share
        inside need not equal the level.
      </span>
    )
  if (stale || props.error)
    readouts.status = (
      <>
        {stale && <Readout label="computing" value={<span aria-busy="true">in a worker…</span>} />}
        {props.error && <Readout label="error" value={props.error} />}
      </>
    )

  const caption = `${dataset.meta!.description} ${
    showExp
      ? `Top left: the data, the band between the τ_lo- and τ_hi-expectile curves at the played iteration (filled), and the true expectiles of the law of y dashed in ink. Each edge is its own model: it minimises J_τ, the asymmetric squared loss plus the penalty, at the λ that LAWS (iteratively reweighted least asymmetric squares, λ re-selected at each pass) settled on. P-IRLS is LAWS itself; gradient descent, SGD, Adam and L-BFGS minimise J_τ directly, by autodiff, from a flat start, so every method ends at the same band. Changing the level retrains both.`
      : `Top left: the data, the fitted mean g⁻¹(η̂) at the played iteration with g⁻¹(η̂ ± 2 se), and the true mean dashed in ink; bottom right: the same fit on the link scale. Expectiles are defined on the response scale and are fitted with a Gaussian smoother that ignores the family and link: switch on “expectile band of y” to fit them here.`
  } Top right: the ${criterionName} criterion and the EDF against log₁₀ λ (for ${showExp ? 'the chosen edge, at its asymmetric weights at the optimum' : 'this family and link'}); drag the λ line on either to fix λ there. Bottom: the basis of ${showExp ? 'the chosen edge' : 'the fit'} at the played iteration (each B-spline times its coefficient, their sum the smooth), and J − J* of ${showExp ? 'both models' : 'the fit'} by iteration. A larger k and degree make the basis more flexible; a larger λ shrinks the fit towards the penalty's null space, the polynomials of degree below the penalty order ${order}, so the EDF falls to ${order}; degree ${degree} sets how smooth each basis function is.`

  return (
    <Figure
      title="Smoothing hyperparameters and expectile bands"
      purpose="λ, the basis size, the spline degree and the penalty order set how flexible a P-spline is; two expectile GAMs at τ_lo and τ_hi make a band that widens and narrows with the noise, and the share of points inside it is not the level."
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
      readouts={readouts}
      caption={caption}
    >
      <ExpectileCharts
        glm={glm}
        showExp={showExp}
        dataKey={dataKey}
        x={x}
        y={y}
        truthFixed={truthFixed}
        truthBand={truthBand}
        band={band}
        tauLo={tauLo}
        tauHi={tauHi}
        familyCurves={familyCurves}
        basis={basis}
        rawBasis={rawBasis}
        traces={traces}
        markers={markers}
        count={count}
        path={path}
        pathStale={props.pathStale}
        handleAt={handleStore}
        setLogLambda={props.setLogLambda}
        criterionName={criterionName}
        stale={stale}
        colorOf={colorOf}
      />
    </Figure>
  )
})

type ChartsProps = {
  glm: boolean
  showExp: boolean
  dataKey: string
  x: number[]
  y: number[]
  truthFixed: { mean: number[]; eta: number[] }
  truthBand: { lo: number[]; hi: number[]; inside: number } | null
  band: { lo: number[]; hi: number[]; inside: number } | null
  tauLo: number
  tauHi: number
  familyCurves: {
    eta: number[]
    lowerEta: number[]
    upperEta: number[]
    mu: number[]
    lowerMu: number[]
    upperMu: number[]
  } | null
  basis: { lines: number[][]; sum: number[] | null; size: number } | null
  rawBasis: boolean
  traces: { tau: number | null; x: number[]; y: number[] }[]
  markers: { x: number[]; y: number[] }[]
  count: number
  path: SmoothingPath | null
  pathStale: boolean
  handleAt: Store<number>
  setLogLambda: (x: number) => void
  criterionName: string
  stale: boolean
  colorOf: (t: number) => string
}

/** The charts, memoised on their data so a control that changes none of it does not rebuild them. */
const ExpectileCharts = memo(function ExpectileCharts(props: ChartsProps) {
  const { glm, showExp, dataKey, x, y, truthFixed, truthBand, band, tauLo, tauHi, familyCurves } = props
  const { basis, rawBasis, traces, markers, stale, colorOf } = props
  // Axes held per dataset and view; parameters move the curves, not the axes.
  const xAxis = useAxis({ label: 'x', range: [0, 1] })
  const yAxis = useAxis({ label: 'y', hold: 'initial', key: `${dataKey}|${showExp}` })
  const lamAxis = useAxis({ label: 'log₁₀ λ', range: [-6, 6] })
  const critAxis = useAxis({
    label: props.criterionName,
    hold: 'initial',
    key: `${dataKey}|${props.criterionName}|${showExp}`,
  })
  const edfAxis = useAxis({ label: 'EDF', range: [0, undefined], hold: 'union', key: dataKey })
  const basisX = useAxis({ label: 'x', range: [0, 1] })
  const basisY = useAxis({ label: rawBasis ? 'Bⱼ(x)' : 'βⱼBⱼ(x)', hold: 'union', key: `${dataKey}|${rawBasis}` })
  const etaAxis = useAxis({ label: 'η (link scale)', hold: 'union', key: dataKey })
  const iterAxis = useAxis({ label: 'iteration', range: [0, Math.max(1, props.count - 1)] })
  // Fixed decades, so the traces patch in place (live) as runs land during a drag.
  const traceY = useAxis({ label: 'J − J*', log: true, range: [1e-16, 10] })
  const nameOf = (tau: number | null) => (tau === null ? 'fit' : `τ = ${tau}`)

  const main = (
    <Plot x={xAxis} y={yAxis} title={showExp ? 'expectile band' : 'response scale'}>
      <Points name="data" x={x} y={y} muted size={3} />
      {showExp && band && (
        <Area
          id="band"
          name={`band τ ∈ [${tauLo}, ${tauHi}]`}
          x={GRID_X}
          y={band.hi}
          base={band.lo}
          color={colorOf(0.5)}
          opacity={0.18}
          line={false}
          stale={stale}
          live
        />
      )}
      {showExp && truthBand && <Curve name="true expectiles" x={GRID_X} y={truthBand.lo} emphasis dashed />}
      {showExp && truthBand && <Curve name="true expectiles" x={GRID_X} y={truthBand.hi} emphasis dashed />}
      {showExp && band && (
        <Curve
          id="lo"
          name={`τ_lo = ${tauLo}`}
          x={GRID_X}
          y={band.lo}
          color={colorOf(tauLo)}
          width={2.5}
          stale={stale}
          live
        />
      )}
      {showExp && band && (
        <Curve
          id="hi"
          name={`τ_hi = ${tauHi}`}
          x={GRID_X}
          y={band.hi}
          color={colorOf(tauHi)}
          width={2.5}
          stale={stale}
          live
        />
      )}
      {!showExp && familyCurves && (
        <Area
          name="± 2 se"
          x={GRID_X}
          y={familyCurves.upperMu}
          base={familyCurves.lowerMu}
          slot={0}
          opacity={0.2}
          line={false}
          stale={stale}
          live
        />
      )}
      {!showExp && <Curve name="true mean" x={GRID_X} y={truthFixed.mean} emphasis dashed />}
      {!showExp && familyCurves && (
        <Curve name="fitted mean" x={GRID_X} y={familyCurves.mu} slot={0} stale={stale} live />
      )}
    </Plot>
  )

  const basisPlot = (
    <Plot
      x={basisX}
      y={basisY}
      title={basis ? `${rawBasis ? 'B-spline basis' : 'basis × coefficients'}: ${basis.size} functions` : 'basis'}
      legend={false}
    >
      {basis?.lines.map((c, i) => (
        <Curve key={i} id={`b${i}`} name={`B${i + 1}`} x={GRID_X} y={c} slot={0} thin live stale={stale} />
      ))}
      {basis?.sum && <Curve id="bsum" name="sum: f(x)" x={GRID_X} y={basis.sum} emphasis live stale={stale} />}
    </Plot>
  )

  const tracePlot = (
    <Plot x={iterAxis} y={traceY} title="training trace">
      {traces.map((tr) => (
        <Curve
          key={nameOf(tr.tau)}
          id={`trace-${nameOf(tr.tau)}`}
          name={nameOf(tr.tau)}
          x={tr.x}
          y={tr.y}
          {...(tr.tau === null ? { slot: 0 } : { color: colorOf(tr.tau) })}
          showPoints={tr.x.length <= 60}
          stale={stale}
          live
        />
      ))}
      {markers.map((m, q) => (
        <Points key={q} id={`marker-${q}`} name="this iteration" x={m.x} y={m.y} emphasis live />
      ))}
    </Plot>
  )

  const linkPlot = (
    <Plot x={xAxis} y={etaAxis} title="link scale">
      {familyCurves && (
        <Area
          name="± 2 se"
          x={GRID_X}
          y={familyCurves.upperEta}
          base={familyCurves.lowerEta}
          slot={0}
          opacity={0.2}
          line={false}
          stale={stale}
          live
        />
      )}
      <Curve name="true η" x={GRID_X} y={truthFixed.eta} emphasis dashed />
      {familyCurves && <Curve name="fitted η" x={GRID_X} y={familyCurves.eta} slot={0} stale={stale} live />}
    </Plot>
  )

  return (
    <Dashboard>
      <DashboardRow ratio={1.8} minHeight={320}>
        <DashboardCell ratio={2}>{main}</DashboardCell>
        <DashboardCell ratio={1}>
          <CriterionPanels
            path={props.path}
            pathStale={props.pathStale}
            handleAt={props.handleAt}
            setLogLambda={props.setLogLambda}
            criterionName={props.criterionName}
            lamAxis={lamAxis}
            critAxis={critAxis}
            edfAxis={edfAxis}
          />
        </DashboardCell>
      </DashboardRow>
      <DashboardRow ratio={1} minHeight={220}>
        <DashboardCell>{basisPlot}</DashboardCell>
        <DashboardCell>{tracePlot}</DashboardCell>
        {glm && !showExp && <DashboardCell>{linkPlot}</DashboardCell>}
      </DashboardRow>
    </Dashboard>
  )
})

/** The criterion and the EDF against λ, with the λ line: the only charts a λ drag redraws. */
const CriterionPanels = memo(function CriterionPanels(props: {
  path: SmoothingPath | null
  pathStale: boolean
  handleAt: Store<number>
  setLogLambda: (x: number) => void
  criterionName: string
  lamAxis: AxisModel
  critAxis: AxisModel
  edfAxis: AxisModel
}) {
  const { path, pathStale, lamAxis, critAxis, edfAxis } = props
  const at = useSyncExternalStore(props.handleAt.subscribe, props.handleAt.get, props.handleAt.get)
  return (
    <Plots rows={2} heights={[1, 1]}>
      <Plot x={lamAxis} y={critAxis} title={`${props.criterionName} against λ`} legend={false}>
        {path && <Curve name={props.criterionName} x={PATH_LOG10} y={path.criterion} slot={6} stale={pathStale} live />}
        <Handle kind="x" at={at} label="λ" onDrag={props.setLogLambda} />
      </Plot>
      <Plot x={lamAxis} y={edfAxis} title="effective degrees of freedom" legend={false}>
        {path && <Curve name="EDF" x={PATH_LOG10} y={path.edf} slot={6} stale={pathStale} live />}
        <Handle kind="x" at={at} label="λ" onDrag={props.setLogLambda} />
      </Plot>
    </Plots>
  )
})
