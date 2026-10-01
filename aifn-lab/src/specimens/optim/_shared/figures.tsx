import {
  adagrad,
  adam,
  conjugateGradient,
  gradientDescent,
  momentum,
  nesterov,
  rmsprop,
  type FirstOrderState,
} from 'aifn/optim/first-order'
import { bfgs, lbfgs, newton, trustRegion } from 'aifn/optim/second-order'
import { cmaEs, nelderMead, type NelderMeadState, type CmaEsState } from 'aifn/optim/derivative-free'
import { fista, ista, proxL1 } from 'aifn/optim/proximal'
import { type IterateState, type ObjectiveFn, type StartOptions } from 'aifn/optim'
import { himmelblau, quadraticBowl, rastrigin, rosenbrock, type TestFunction } from 'aifn-applied/data/objectives'
import { child, normals, stream } from 'aifn/foundation/random'
import { tensor, toFlat, type Tensor } from 'aifn/foundation/tensor'
import { trace, type Algorithm, type Trace } from 'aifn/foundation/trace'
import { useMemo, type ReactNode } from 'react'
import { Figure } from '@lab/layout'
import { choice, slider, useComputed, useFigureState, variants, type ParamDefs } from '@lab/state'
import { Curve, Handle, Plot, Points, Raster, Readout, useAxis, type Handle as HandleSpec } from '@lab/viz'
import { TracePanel } from '@lab/views'

// ---------------------------------------------------------------------------------------------------------------------
// Shared: a test function's surface as a heatmap of log₁₀(1 + f), with a path drawn over it.

type Surface = { x: number[]; y: number[]; z: number[][] }

function surfaceOf(fn: TestFunction, n = 90): Surface {
  const [x0, y0] = toFlat(fn.lo)
  const [x1, y1] = toFlat(fn.hi)
  const x = Array.from({ length: n }, (_, i) => x0 + ((x1 - x0) * i) / (n - 1))
  const y = Array.from({ length: n }, (_, i) => y0 + ((y1 - y0) * i) / (n - 1))
  const z = y.map((yi) => x.map((xj) => Math.log10(1 + fn.value(tensor([xj, yi])))))
  return { x, y, z }
}

/** Columns of a [kept, 2] series as xs and ys. */
function pathOf(series: Tensor): { xs: number[]; ys: number[] } {
  const data = toFlat(series)
  const kept = series.shape[0]
  return {
    xs: Array.from({ length: kept }, (_, k) => data[2 * k]),
    ys: Array.from({ length: kept }, (_, k) => data[2 * k + 1]),
  }
}

/**
 * A test function's surface (log₁₀(1 + f)) with the path up to the current step, the current iterate, the minima and
 * any extra layers (a simplex, a population, line-search trials); `start` makes the start point draggable.
 */
function SurfaceWithPath({
  surface,
  path,
  position,
  minima,
  start,
  children,
}: {
  surface: Surface
  path: { xs: number[]; ys: number[] }
  position: number
  minima: Tensor[]
  start?: HandleSpec
  children?: ReactNode
}) {
  const clip = (v: number, a: readonly number[]) => Math.min(Math.max(v, a[0]), a[a.length - 1])
  const minimaXY = useMemo(() => ({ x: minima.map((m) => toFlat(m)[0]), y: minima.map((m) => toFlat(m)[1]) }), [minima])
  const soFar = useMemo(
    () => ({
      x: path.xs.slice(0, position + 1).map((v) => clip(v, surface.x)),
      y: path.ys.slice(0, position + 1).map((v) => clip(v, surface.y)),
    }),
    [path, position, surface],
  )
  const x = useAxis({ label: 'x₀' })
  const y = useAxis({ label: 'x₁', equal: x })
  return (
    <Plot x={x} y={y}>
      <Raster x={surface.x} y={surface.y} z={surface.z} valueLabel="log₁₀(1 + f)" />
      <Points name="minimum" x={minimaXY.x} y={minimaXY.y} emphasis />
      <Curve name="path" x={soFar.x} y={soFar.y} slot={1} showPoints live />
      {children}
      <Points
        name="current"
        x={[clip(path.xs[position], surface.x)]}
        y={[clip(path.ys[position], surface.y)]}
        slot={1}
        size={10}
        live
      />
      {start && <Handle {...start} />}
    </Plot>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 1. Optimisers on Rosenbrock.

const ROSEN = rosenbrock()

type MethodId =
  | 'gradient-descent'
  | 'momentum'
  | 'nesterov'
  | 'adagrad'
  | 'rmsprop'
  | 'adam'
  | 'newton'
  | 'trust-region'
  | 'bfgs'
  | 'lbfgs'
  | 'conjugate-gradient'
  | 'nelder-mead'

const METHODS: { value: MethodId; label: string; lr?: [number, number, number] }[] = [
  { value: 'gradient-descent', label: 'gradient descent', lr: [0.001, 0.0001, 0.004] },
  { value: 'momentum', label: 'heavy-ball momentum', lr: [0.0005, 0.0001, 0.002] },
  { value: 'nesterov', label: 'Nesterov', lr: [0.0005, 0.0001, 0.002] },
  { value: 'adagrad', label: 'AdaGrad', lr: [0.3, 0.01, 1] },
  { value: 'rmsprop', label: 'RMSProp', lr: [0.003, 0.0005, 0.03] },
  { value: 'adam', label: 'Adam', lr: [0.02, 0.001, 0.2] },
  { value: 'newton', label: 'damped Newton' },
  { value: 'trust-region', label: 'trust region (dogleg)' },
  { value: 'bfgs', label: 'BFGS' },
  { value: 'lbfgs', label: 'L-BFGS' },
  { value: 'conjugate-gradient', label: 'conjugate gradient (PR+)' },
  { value: 'nelder-mead', label: 'Nelder–Mead' },
]

function algorithmFor(method: MethodId, fn: TestFunction, lr: number): Algorithm<StartOptions, IterateState> {
  const f = fn.objective
  switch (method) {
    case 'gradient-descent':
      return gradientDescent(f, { stepSize: lr })
    case 'momentum':
      return momentum(f, { stepSize: lr })
    case 'nesterov':
      return nesterov(f, { stepSize: lr })
    case 'adagrad':
      return adagrad(f, { stepSize: lr })
    case 'rmsprop':
      return rmsprop(f, { stepSize: lr })
    case 'adam':
      return adam(f, { stepSize: lr })
    case 'newton':
      return newton(f, { hessian: fn.hessian })
    case 'trust-region':
      return trustRegion(f, { hessian: fn.hessian })
    case 'bfgs':
      return bfgs(f)
    case 'lbfgs':
      return lbfgs(f)
    case 'conjugate-gradient':
      return conjugateGradient(f)
    case 'nelder-mead':
      return nelderMead(fn.value)
  }
}

/** Each method as a variants case: the first-order ones carry their step size. */
const METHOD_CASES = Object.fromEntries(
  METHODS.map((m) => [
    m.value,
    {
      label: m.label,
      params: (m.lr
        ? { lr: slider(m.lr[1], m.lr[2], m.lr[0], { label: 'step size η', step: m.lr[1] }) }
        : {}) as ParamDefs,
    },
  ]),
)

const [ROSEN_X0, ROSEN_Y0] = toFlat(ROSEN.start)

export function RosenbrockSpecimen() {
  const state = useFigureState({
    method: variants(METHOD_CASES, { label: '1 · method', choiceLabel: 'method', initial: 'bfgs' }),
    sx: slider(-2, 2, ROSEN_X0, { onChart: true, step: 0.01, label: 'start x₀' }),
    sy: slider(-1, 3, ROSEN_Y0, { onChart: true, step: 0.01, label: 'start x₁' }),
  })
  const method = state.method.key as MethodId
  const lr = ((state.method.values as Record<string, number>).lr ?? 0) as number
  const { sx, sy } = state
  const surface = useMemo(() => surfaceOf(ROSEN), [])
  // Up to 3000 steps per change, and the trace view's charts redraw in full for every new trace: a dragged start
  // recomputes on release.
  const computed = useComputed(
    () =>
      trace(algorithmFor(method, ROSEN, lr), { x0: [sx, sy] }, 3000, {
        record: {
          'f(x)': (s) => s.value,
          x: (s) => s.x,
          evaluations: (s) => s.evaluations,
        },
        checkpointEvery: 100,
      }),
    [method, lr, sx, sy],
    { mode: 'release' },
  )
  const t = computed.value
  const path = useMemo(() => pathOf(t.series.x), [t])
  return (
    <Figure
      title="Optimisers on the Rosenbrock function"
      purpose="The Rosenbrock valley is curved, narrow and flat along its floor: first-order methods zigzag or crawl along it, while curvature-aware methods (Newton, BFGS, L-BFGS, trust region) follow it in a few dozen steps."
      defaultSize="L"
      state={state}
      readouts={<Readout label="steps" value={`${t.meta.steps}, ${t.meta.stopped}`} />}
      caption="f = 100(x₁ − x₀²)² + (1 − x₀)², from (−1.2, 1) by default; the minimum (1, 1) sits at the end of the valley. Drag the start point; play or scrub the steps to follow the path. Switch the series chart to a log scale to compare rates across methods."
    >
      <TracePanel
        trace={t}
        show={['f(x)']}
        renderState={(_, { position }) => (
          <SurfaceWithPath
            surface={surface}
            path={path}
            position={position}
            minima={ROSEN.minima}
            start={state.handle(['sx', 'sy'], { label: 'start' })}
          />
        )}
      />
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 2. Line-search trials.

const LINE_SEARCHES = [
  { value: 'backtracking', label: 'backtracking (Armijo)' },
  { value: 'strong-wolfe', label: 'strong Wolfe' },
] as const

export function LineSearchSpecimen() {
  const state = useFigureState({
    kind: choice(LINE_SEARCHES, 'backtracking', { label: 'line search' }),
    alpha0: slider(0.001, 0.2, 0.05, { label: 'first trial α₀' }),
  })
  const { kind, alpha0 } = state
  const surface = useMemo(() => surfaceOf(ROSEN), [])
  const t = useMemo(
    () =>
      trace(gradientDescent(ROSEN.objective, { lineSearch: kind, stepSize: alpha0 }), { x0: ROSEN.start }, 200, {
        record: {
          'f(x)': (s) => s.value,
          'step α': (s) => s.stepSize,
          trials: (s) => s.lineSearch?.trials.length ?? 0,
          x: (s) => s.x,
        },
      }),
    [kind, alpha0],
  )
  const path = useMemo(() => pathOf(t.series.x), [t])
  return (
    <Figure
      title="Line-search trials along −∇f"
      purpose="A line search tries step lengths along −∇f until one decreases f enough (Armijo) or also flattens the slope (strong Wolfe), so a fixed first trial α₀ need not be stable."
      defaultSize="L"
      state={state}
      caption="Step through: the left chart is the surface with each step's trials; the right chart is φ(α) = f(x − α∇f) for the step that produced the current state, with the Armijo line f(x) + c₁α∇fᵀp and every trial the search evaluated; the accepted trial is in ink."
    >
      <TracePanel
        trace={t}
        show={['f(x)', 'trials']}
        renderState={(s: FirstOrderState, { position, trace: tr }) => (
          <div className="grid h-full gap-2 md:grid-cols-2">
            <SurfaceWithPath surface={surface} path={path} position={position} minima={ROSEN.minima}>
              <Points
                name="trials"
                x={s.lineSearch?.trials.map((q) => toFlat(q.x)[0]) ?? []}
                y={s.lineSearch?.trials.map((q) => toFlat(q.x)[1]) ?? []}
                slot={2}
                live
              />
            </SurfaceWithPath>
            <LineSearchProfile state={s} previous={position > 0 ? tr.steps[position - 1] : null} />
          </div>
        )}
      />
    </Figure>
  )
}

function LineSearchProfile({ state, previous }: { state: FirstOrderState; previous: FirstOrderState | null }) {
  const profile = useMemo(() => {
    const ls = state.lineSearch
    if (!ls || !previous) return null
    const x = toFlat(previous.x)
    const g = toFlat(previous.grad)
    const alphaMax = Math.max(...ls.trials.map((q) => q.alpha)) * 1.15
    const alphas = Array.from({ length: 120 }, (_, i) => (alphaMax * i) / 119)
    return {
      alphas,
      phi: alphas.map((a) => ROSEN.value(tensor([x[0] - a * g[0], x[1] - a * g[1]]))),
      armijo: alphas.map((a) => previous.value + 1e-4 * a * ls.initialSlope),
      trials: { x: ls.trials.map((q) => q.alpha), y: ls.trials.map((q) => q.value) },
      accepted: { x: [ls.alpha], y: [ls.value] },
    }
  }, [state, previous])
  const x = useAxis({ label: 'α' })
  const y = useAxis({ label: 'f(x − α∇f)' })
  if (!profile) return <div className="p-4 text-xs text-muted-foreground">Step 0: no line search yet.</div>
  return (
    <Plot x={x} y={y}>
      <Curve name="φ(α)" x={profile.alphas} y={profile.phi} slot={0} />
      <Curve name="Armijo line" x={profile.alphas} y={profile.armijo} dashed slot={3} />
      <Points name="trials" x={profile.trials.x} y={profile.trials.y} slot={2} />
      <Points name="accepted" x={profile.accepted.x} y={profile.accepted.y} emphasis />
    </Plot>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 3. L-BFGS against gradient descent.

export function ConvergenceComparisonSpecimen() {
  const state = useFigureState({
    problem: variants(
      {
        'bowl-50': {
          label: 'quadratic bowl, n = 50',
          params: { logKappa: slider(0, 5, 2, { step: 0.5, label: 'log₁₀ condition number κ' }) },
        },
        'rosenbrock-2': { label: 'Rosenbrock, n = 2', params: {} },
        'rosenbrock-10': { label: 'Rosenbrock, n = 10', params: {} },
      },
      { label: '1 · problem', choiceLabel: 'problem' },
    ),
    memory: slider(1, 20, 5, { step: 1, label: 'L-BFGS memory m' }),
    axis: choice(['evaluations', 'steps'], 'evaluations', { label: 'x axis' }),
  })
  const problem = state.problem.key
  const logKappa = state.problem.key === 'bowl-50' ? state.problem.values.logKappa : 0
  const { memory, axis } = state
  const fn = useMemo(() => {
    if (problem === 'rosenbrock-2') return rosenbrock()
    if (problem === 'rosenbrock-10') return rosenbrock({ n: 10 })
    return quadraticBowl({ n: 50, condition: 10 ** logKappa })
  }, [problem, logKappa])
  const runs = useMemo(() => {
    // Gradient descent with the classical step 1/L for the bowl (L = κ), or a stable step on Rosenbrock.
    const gdLr = problem === 'bowl-50' ? 1 / 10 ** logKappa : 0.0008
    const algs: [string, Algorithm<StartOptions, IterateState>, number][] = [
      ['gradient descent', gradientDescent(fn.objective, { stepSize: gdLr }), 0],
      ['Nesterov', nesterov(fn.objective, { stepSize: gdLr }), 1],
      ['conjugate gradient', conjugateGradient(fn.objective), 2],
      [`L-BFGS (m = ${memory})`, lbfgs(fn.objective, { memory }), 3],
      ['BFGS', bfgs(fn.objective), 4],
      ['damped Newton', newton(fn.objective, { hessian: fn.hessian }), 5],
    ]
    return algs.map(([name, alg, slot]) => {
      const tr = trace(alg, { x0: fn.start }, 2000, {
        record: { gap: (s) => Math.max(s.value - fn.minimumValue, 1e-300), evaluations: (s) => s.evaluations },
        stopOnNonFinite: false,
      })
      return { name, slot, tr }
    })
  }, [fn, problem, logKappa, memory])
  const lines = useMemo(
    () =>
      runs.map(({ name, slot, tr }) => ({
        name,
        slot,
        x: axis === 'steps' ? Array.from(tr.index) : toFlat(tr.series.evaluations),
        y: toFlat(tr.series.gap).map((v) => Math.max(v, 1e-16)),
      })),
    [runs, axis],
  )
  const x = useAxis({ label: axis, log: axis === 'evaluations', key: `${problem}|${axis}` })
  const y = useAxis({ label: 'f(x) − f*', log: true, range: [1e-16, undefined], key: problem })
  return (
    <Figure
      title="L-BFGS against gradient descent"
      purpose="On an ill-conditioned problem gradient descent's error shrinks by (κ − 1)/(κ + 1) per step, while conjugate gradient and L-BFGS use curvature from past steps and need far fewer evaluations; Newton solves a quadratic in one step."
      defaultSize="L"
      state={state}
      readouts={runs.map(({ name, tr }) => (
        <Readout key={name} label={name} value={`${tr.meta.steps} steps, ${tr.meta.stopped}`} />
      ))}
      caption="f(x) − f* on a log scale (floored at 10⁻¹⁶). Raise κ on the bowl: gradient descent and Nesterov slow down sharply, conjugate gradient and L-BFGS far less. A larger L-BFGS memory approaches BFGS."
    >
      <Plot x={x} y={y}>
        {lines.map((l) => (
          <Curve key={l.name} name={l.name} x={l.x} y={l.y} slot={l.slot} />
        ))}
      </Plot>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 4. Nelder–Mead on Himmelblau.

const HIMMELBLAU = himmelblau()

export function NelderMeadSpecimen() {
  const state = useFigureState({
    sx: slider(-4.5, 4.5, -1, { onChart: true, step: 0.01, label: 'start x₀' }),
    sy: slider(-4.5, 4.5, -1, { onChart: true, step: 0.01, label: 'start x₁' }),
  })
  const { sx, sy } = state
  const surface = useMemo(() => surfaceOf(HIMMELBLAU), [])
  // The trace view's charts redraw in full for every new trace, so a dragged start recomputes on release.
  const computed = useComputed(
    () =>
      trace(nelderMead(HIMMELBLAU.value, { adaptive: false }), { x0: [sx, sy] }, 200, {
        record: { 'best f': (s) => s.value, 'simplex size': (s) => s.size, x: (s) => s.x },
      }),
    [sx, sy],
    { mode: 'release' },
  )
  const t = computed.value
  const path = useMemo(() => pathOf(t.series.x), [t])
  return (
    <Figure
      title="Nelder–Mead on Himmelblau's function"
      purpose="Nelder–Mead needs only function values: a triangle reflects, expands, contracts or shrinks towards lower f, and which of Himmelblau's four minima it reaches depends on the start."
      defaultSize="L"
      state={state}
      caption="Drag the start point to another quadrant and the simplex settles in another minimum. Step through to watch the simplex (the triangle); the state names each step's operation."
    >
      <TracePanel
        trace={t}
        show={['best f', 'simplex size']}
        renderState={(s: NelderMeadState, { position }) => {
          const v = toFlat(s.simplex)
          return (
            <SurfaceWithPath
              surface={surface}
              path={path}
              position={position}
              minima={HIMMELBLAU.minima}
              start={state.handle(['sx', 'sy'], { label: 'start' })}
            >
              <Curve
                name={`simplex (${s.operation})`}
                x={[v[0], v[2], v[4], v[0]]}
                y={[v[1], v[3], v[5], v[1]]}
                slot={3}
                live
              />
            </SurfaceWithPath>
          )
        }}
      />
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 5. CMA-ES on Rastrigin.

const RASTRIGIN = rastrigin()

export function CmaEsSpecimen() {
  const state = useFigureState({
    sigma: slider(0.1, 4, 2, { label: 'initial σ' }),
    seed: slider(1, 30, 1, { step: 1, label: 'seed' }),
  })
  const { sigma, seed } = state
  const surface = useMemo(() => surfaceOf(RASTRIGIN, 120), [])
  const t = useMemo(
    () =>
      trace(cmaEs(RASTRIGIN.value, { sigma, populationSize: 12 }), { x0: [3, 3] }, 150, {
        record: {
          'f(mean)': (s) => s.value,
          'best f': (s) => s.bestValue,
          'step size σ': (s) => s.sigma,
          x: (s) => s.x,
        },
        stream: stream(seed),
      }),
    [sigma, seed],
  )
  const path = useMemo(() => pathOf(t.series.x), [t])
  return (
    <Figure
      title="CMA-ES on the Rastrigin function"
      purpose="CMA-ES samples a population from N(m, σ²C), moves the mean towards the best half and adapts C and σ; a large initial σ sees past the local minima, a small one is trapped by the nearest."
      defaultSize="L"
      state={state}
      caption="Each generation samples 12 points (dots) around the mean path. Step through from generation 0; with a small initial σ the run can stall in a local minimum near the start (3, 3)."
    >
      <TracePanel
        trace={t}
        show={['best f', 'step size σ']}
        renderState={(s: CmaEsState, { position }) => {
          const p = toFlat(s.population)
          return (
            <SurfaceWithPath surface={surface} path={path} position={position} minima={RASTRIGIN.minima}>
              <Points
                name="population"
                x={p.filter((_, i) => i % 2 === 0)}
                y={p.filter((_, i) => i % 2 === 1)}
                slot={2}
                live
              />
            </SurfaceWithPath>
          )
        }}
      />
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 6. ISTA against FISTA on a lasso problem.

export function ProximalSpecimen() {
  const state = useFigureState({ logLambda: slider(-2, 1.5, -0.5, { label: 'log₁₀ λ' }) })
  const lambda = 10 ** state.logLambda
  const problem = useMemo(() => {
    // Least squares ½‖Ax − b‖² with A 40×20 Gaussian and a sparse truth.
    const s = stream('optim/lasso')
    const A = toFlat(normals(child(s, 'A'), [40, 20]))
    const truth = Array.from({ length: 20 }, (_, j) => (j % 5 === 0 ? 3 - j / 5 : 0))
    const noise = toFlat(normals(child(s, 'noise'), 40, 0, 0.1))
    const b = Array.from({ length: 40 }, (_, i) => truth.reduce((acc, t, j) => acc + A[i * 20 + j] * t, 0) + noise[i])
    const f: ObjectiveFn = (x) => {
      const v = toFlat(x)
      const r = b.map((bi, i) => v.reduce((acc, vj, j) => acc + A[i * 20 + j] * vj, 0) - bi)
      const grad = v.map((_, j) => r.reduce((acc, ri, i) => acc + A[i * 20 + j] * ri, 0))
      return { value: 0.5 * r.reduce((acc, ri) => acc + ri * ri, 0), grad }
    }
    // L = ‖A‖² bounded by the Frobenius norm squared; backtracking refines it.
    return { f, x0: new Array<number>(20).fill(0) }
  }, [])
  const runs = useMemo(() => {
    const g = proxL1(lambda)
    const options = { lr: 1, backtracking: true, tolerance: 0 }
    const record = { F: (s: IterateState) => s.value }
    const a = trace(ista(problem.f, g, options), { x0: problem.x0 }, 400, { record })
    const b = trace(fista(problem.f, g, options), { x0: problem.x0 }, 400, { record })
    const best = Math.min(...toFlat(a.series.F), ...toFlat(b.series.F))
    return { a, b, best }
  }, [problem, lambda])
  const lines = useMemo(
    () =>
      (
        [
          ['ISTA', runs.a, 0],
          ['FISTA', runs.b, 1],
        ] as const
      ).map(([name, tr, slot]) => ({
        name,
        slot,
        x: Array.from(tr.index),
        y: toFlat(tr.series.F).map((v) => Math.max(v - runs.best, 1e-14)),
      })),
    [runs],
  )
  const nonzero = (tr: Trace<IterateState>) => toFlat(tr.steps.at(-1)!.x).filter((v) => v !== 0).length
  const x = useAxis({ label: 'step k' })
  const y = useAxis({ label: 'F(x_k) − F*', log: true, range: [1e-14, undefined] })
  return (
    <Figure
      title="ISTA against FISTA on a lasso"
      purpose="Proximal gradient takes a gradient step on the smooth part and soft-thresholds for the L1 part; Nesterov's momentum (FISTA) improves the worst-case rate from O(1/k) to O(1/k²)."
      state={state}
      readouts={<Readout label="non-zero coefficients (FISTA)" value={`${nonzero(runs.b)} of 20`} />}
      caption="F(x) = ½‖Ax − b‖² + λ‖x‖₁ minus the best value found, on a log scale; A is 40 × 20 Gaussian and the truth has 3 non-zero coefficients. Raise λ and more coefficients are thresholded to exactly zero; once the support is found both converge linearly, and on this well-conditioned problem FISTA's momentum overshoots (the ripples) rather than gaining."
    >
      <Plot x={x} y={y}>
        {lines.map((l) => (
          <Curve key={l.name} name={l.name} x={l.x} y={l.y} slot={l.slot} />
        ))}
      </Plot>
    </Figure>
  )
}
