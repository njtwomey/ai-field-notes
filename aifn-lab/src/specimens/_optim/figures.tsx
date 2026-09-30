import {
  adagrad,
  adam,
  bfgs,
  cmaEs,
  conjugateGradient,
  fista,
  gradientDescent,
  ista,
  lbfgs,
  momentum,
  nelderMead,
  nesterov,
  newton,
  proxL1,
  rmsprop,
  trustRegion,
  type FirstOrderState,
  type IterateState,
  type NelderMeadState,
  type CmaEsState,
  type Objective,
  type StartOptions,
} from 'aifn/optim'
import { himmelblau, quadraticBowl, rastrigin, rosenbrock, type TestFunction } from 'aifn/datasets'
import { normals, stream } from 'aifn/random'
import { tensor, toFlat, type Tensor } from 'aifn/tensor'
import { trace, type Algorithm, type Trace } from 'aifn/trace'
import { useMemo, useState } from 'react'
import { Select, Slider } from '@lab/controls'
import { Figure } from '@lab/layout'
import { Heatmap, Readout, XYChart, type HeatmapOverlay, type XYSeries } from '@lab/viz'
import { TraceView } from '@lab/views'

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

function SurfaceWithPath({
  surface,
  path,
  position,
  extra = [],
  minima,
}: {
  surface: Surface
  path: { xs: number[]; ys: number[] }
  position: number
  extra?: HeatmapOverlay[]
  minima: Tensor[]
}) {
  const minimaOverlay = useMemo<HeatmapOverlay>(
    () => ({
      name: 'minimum',
      type: 'scatter',
      x: minima.map((m) => toFlat(m)[0]),
      y: minima.map((m) => toFlat(m)[1]),
      emphasis: true,
    }),
    [minima],
  )
  const clip = (v: number, a: readonly number[]) => Math.min(Math.max(v, a[0]), a[a.length - 1])
  return (
    <Heatmap
      x={surface.x}
      y={surface.y}
      z={surface.z}
      xLabel="x₀"
      yLabel="x₁"
      valueLabel="log₁₀(1 + f)"
      overlay={[
        minimaOverlay,
        {
          name: 'path',
          type: 'line',
          x: path.xs.slice(0, position + 1).map((v) => clip(v, surface.x)),
          y: path.ys.slice(0, position + 1).map((v) => clip(v, surface.y)),
          showPoints: true,
          slot: 1,
        },
        ...extra,
      ]}
      marker={[clip(path.xs[position], surface.x), clip(path.ys[position], surface.y)]}
    />
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
      return gradientDescent(f, { lr })
    case 'momentum':
      return momentum(f, { lr })
    case 'nesterov':
      return nesterov(f, { lr })
    case 'adagrad':
      return adagrad(f, { lr })
    case 'rmsprop':
      return rmsprop(f, { lr })
    case 'adam':
      return adam(f, { lr })
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

export function RosenbrockSpecimen() {
  const [method, setMethod] = useState<MethodId>('bfgs')
  const spec = METHODS.find((m) => m.value === method)!
  const [lrs, setLrs] = useState<Partial<Record<MethodId, number>>>({})
  const lr = lrs[method] ?? spec.lr?.[0] ?? 0
  const surface = useMemo(() => surfaceOf(ROSEN), [])
  const t = useMemo(
    () =>
      trace(algorithmFor(method, ROSEN, lr), { x0: ROSEN.start }, 3000, {
        record: {
          'f(x)': (s) => s.value,
          x: (s) => s.x,
          evaluations: (s) => s.evaluations,
        },
        checkpointEvery: 100,
      }),
    [method, lr],
  )
  const path = useMemo(() => pathOf(t.series.x), [t])
  return (
    <TraceView
      title="Optimisers on the Rosenbrock function"
      trace={t}
      show={['f(x)']}
      defaultSize="L"
      controls={
        <>
          <Select label="method" value={method} onChange={setMethod} options={METHODS} />
          {spec.lr && (
            <Slider
              label="step size η"
              value={lr}
              min={spec.lr[1]}
              max={spec.lr[2]}
              onChange={(v) => setLrs((o) => ({ ...o, [method]: v }))}
            />
          )}
        </>
      }
      caption="f = 100(x₁ − x₀²)² + (1 − x₀)² from (−1.2, 1); the minimum (1, 1) sits at the end of a curved, flat valley. Scrub the steps to follow the path; switch the series chart to a log scale to compare rates."
      renderState={(_, { position }) => (
        <SurfaceWithPath surface={surface} path={path} position={position} minima={ROSEN.minima} />
      )}
    />
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 2. Line-search trials.

export function LineSearchSpecimen() {
  const [kind, setKind] = useState<'backtracking' | 'strong-wolfe'>('backtracking')
  const [alpha0, setAlpha0] = useState(0.05)
  const surface = useMemo(() => surfaceOf(ROSEN), [])
  const t = useMemo(
    () =>
      trace(gradientDescent(ROSEN.objective, { lineSearch: kind, lr: alpha0 }), { x0: ROSEN.start }, 200, {
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
    <TraceView
      title="Line-search trials along −∇f"
      trace={t}
      show={['f(x)', 'trials']}
      startAtFirst
      defaultSize="L"
      controls={
        <>
          <Select
            label="line search"
            value={kind}
            onChange={setKind}
            options={[
              { value: 'backtracking', label: 'backtracking (Armijo)' },
              { value: 'strong-wolfe', label: 'strong Wolfe' },
            ]}
          />
          <Slider label="first trial α₀" value={alpha0} min={0.001} max={0.2} onChange={setAlpha0} />
        </>
      }
      caption="Each step of gradient descent searches along −∇f. The right chart is φ(α) = f(x − α∇f) for the step that produced the current state, with the Armijo line f(x) + c₁α∇fᵀp and every trial the search evaluated; the accepted trial is the ringed one."
      renderState={(s: FirstOrderState, { position, trace: tr }) => (
        <div className="grid h-full gap-2 md:grid-cols-2">
          <SurfaceWithPath
            surface={surface}
            path={path}
            position={position}
            minima={ROSEN.minima}
            extra={[
              {
                name: 'trials',
                type: 'scatter',
                x: s.lineSearch?.trials.map((q) => toFlat(q.x)[0]) ?? [],
                y: s.lineSearch?.trials.map((q) => toFlat(q.x)[1]) ?? [],
                slot: 2,
              },
            ]}
          />
          <LineSearchProfile state={s} previous={position > 0 ? tr.steps[position - 1] : null} />
        </div>
      )}
    />
  )
}

function LineSearchProfile({ state, previous }: { state: FirstOrderState; previous: FirstOrderState | null }) {
  const series = useMemo<XYSeries[]>(() => {
    const ls = state.lineSearch
    if (!ls || !previous) return []
    const x = toFlat(previous.x)
    const g = toFlat(previous.grad)
    const alphaMax = Math.max(...ls.trials.map((q) => q.alpha)) * 1.15
    const alphas = Array.from({ length: 120 }, (_, i) => (alphaMax * i) / 119)
    const phi = alphas.map((a) => ROSEN.value(tensor([x[0] - a * g[0], x[1] - a * g[1]])))
    const armijo = alphas.map((a) => previous.value + 1e-4 * a * ls.initialSlope)
    return [
      { name: 'φ(α)', type: 'line', x: alphas, y: phi, slot: 0 },
      { name: 'Armijo line', type: 'line', x: alphas, y: armijo, dashed: true, slot: 3 },
      { name: 'trials', type: 'scatter', x: ls.trials.map((q) => q.alpha), y: ls.trials.map((q) => q.value), slot: 2 },
      { name: 'accepted', type: 'scatter', x: [ls.alpha], y: [ls.value], emphasis: true },
    ]
  }, [state, previous])
  if (!series.length) return <div className="p-4 text-xs text-muted-foreground">Step 0: no line search yet.</div>
  return <XYChart series={series} xLabel="α" yLabel="f(x − α∇f)" />
}

// ---------------------------------------------------------------------------------------------------------------------
// 3. L-BFGS against gradient descent.

type Problem = 'rosenbrock-2' | 'rosenbrock-10' | 'bowl-50'

export function ConvergenceComparisonSpecimen() {
  const [problem, setProblem] = useState<Problem>('bowl-50')
  const [logKappa, setLogKappa] = useState(2)
  const [memory, setMemory] = useState(5)
  const [axis, setAxis] = useState<'steps' | 'evaluations'>('evaluations')
  const fn = useMemo(() => {
    if (problem === 'rosenbrock-2') return rosenbrock()
    if (problem === 'rosenbrock-10') return rosenbrock({ n: 10 })
    return quadraticBowl({ n: 50, condition: 10 ** logKappa })
  }, [problem, logKappa])
  const runs = useMemo(() => {
    // Gradient descent with the classical step 1/L for the bowl (L = κ), or a stable step on Rosenbrock.
    const gdLr = problem === 'bowl-50' ? 1 / 10 ** logKappa : 0.0008
    const algs: [string, Algorithm<StartOptions, IterateState>, number][] = [
      ['gradient descent', gradientDescent(fn.objective, { lr: gdLr }), 0],
      ['Nesterov', nesterov(fn.objective, { lr: gdLr }), 1],
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
  const series = useMemo<XYSeries[]>(
    () =>
      runs.map(({ name, slot, tr }) => ({
        name,
        slot,
        type: 'line',
        x: axis === 'steps' ? tr.index : toFlat(tr.series.evaluations),
        y: toFlat(tr.series.gap).map((v) => Math.max(v, 1e-16)),
      })),
    [runs, axis],
  )
  return (
    <Figure
      title="L-BFGS against gradient descent"
      defaultSize="L"
      controls={
        <>
          <Select
            label="problem"
            value={problem}
            onChange={setProblem}
            options={[
              { value: 'bowl-50', label: 'quadratic bowl, n = 50' },
              { value: 'rosenbrock-2', label: 'Rosenbrock, n = 2' },
              { value: 'rosenbrock-10', label: 'Rosenbrock, n = 10' },
            ]}
          />
          {problem === 'bowl-50' && (
            <Slider
              label="log₁₀ condition number κ"
              value={logKappa}
              min={0}
              max={5}
              step={0.5}
              onChange={setLogKappa}
            />
          )}
          <Slider label="L-BFGS memory m" value={memory} min={1} max={20} step={1} onChange={setMemory} />
          <Select label="x axis" value={axis} onChange={setAxis} options={['evaluations', 'steps']} />
        </>
      }
      readouts={runs.map(({ name, tr }) => (
        <Readout key={name} label={name} value={`${tr.meta.steps} steps, ${tr.meta.stopped}`} />
      ))}
      caption="f(x) − f* on a log scale. Gradient descent's rate on the bowl is (κ − 1)/(κ + 1) per step; conjugate gradient and L-BFGS depend on √κ or better, and Newton solves a quadratic in one step."
    >
      <XYChart series={series} yLog xLog={axis === 'evaluations'} xLabel={axis} yLabel="f(x) − f*" />
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 4. Nelder–Mead on Himmelblau.

const HIMMELBLAU = himmelblau()

export function NelderMeadSpecimen() {
  const [x0, setX0] = useState(-1)
  const [y0, setY0] = useState(-1)
  const surface = useMemo(() => surfaceOf(HIMMELBLAU), [])
  const t = useMemo(
    () =>
      trace(nelderMead(HIMMELBLAU.value, { adaptive: false }), { x0: [x0, y0] }, 200, {
        record: { 'best f': (s) => s.value, 'simplex size': (s) => s.size, x: (s) => s.x },
      }),
    [x0, y0],
  )
  const path = useMemo(() => pathOf(t.series.x), [t])
  return (
    <TraceView
      title="Nelder–Mead on Himmelblau's function"
      trace={t}
      show={['best f', 'simplex size']}
      startAtFirst
      defaultSize="L"
      controls={
        <>
          <Slider label="start x₀" value={x0} min={-4.5} max={4.5} onChange={setX0} />
          <Slider label="start x₁" value={y0} min={-4.5} max={4.5} onChange={setY0} />
        </>
      }
      caption="The simplex (triangle) reflects, expands, contracts or shrinks each step; the state names the operation. Which of the four minima it finds depends on the start."
      renderState={(s: NelderMeadState, { position }) => {
        const v = toFlat(s.simplex)
        return (
          <SurfaceWithPath
            surface={surface}
            path={path}
            position={position}
            minima={HIMMELBLAU.minima}
            extra={[
              {
                name: `simplex (${s.operation})`,
                type: 'line',
                x: [v[0], v[2], v[4], v[0]],
                y: [v[1], v[3], v[5], v[1]],
                slot: 3,
              },
            ]}
          />
        )
      }}
    />
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 5. CMA-ES on Rastrigin.

const RASTRIGIN = rastrigin()

export function CmaEsSpecimen() {
  const [sigma, setSigma] = useState(2)
  const [seed, setSeed] = useState(1)
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
    <TraceView
      title="CMA-ES on the Rastrigin function"
      trace={t}
      show={['best f', 'step size σ']}
      startAtFirst
      defaultSize="L"
      controls={
        <>
          <Slider label="initial σ" value={sigma} min={0.1} max={4} onChange={setSigma} />
          <Slider label="seed" value={seed} min={1} max={30} step={1} onChange={setSeed} />
        </>
      }
      caption="Each generation samples 12 points from N(m, σ²C) (dots), moves the mean towards the best half, and adapts C and σ. A large initial σ smooths over the local minima; a small one gets trapped."
      renderState={(s: CmaEsState, { position }) => {
        const p = toFlat(s.population)
        return (
          <SurfaceWithPath
            surface={surface}
            path={path}
            position={position}
            minima={RASTRIGIN.minima}
            extra={[
              {
                name: 'population',
                type: 'scatter',
                x: p.filter((_, i) => i % 2 === 0),
                y: p.filter((_, i) => i % 2 === 1),
                slot: 2,
              },
            ]}
          />
        )
      }}
    />
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 6. ISTA against FISTA on a lasso problem.

export function ProximalSpecimen() {
  const [logLambda, setLogLambda] = useState(-0.5)
  const lambda = 10 ** logLambda
  const problem = useMemo(() => {
    // Least squares ½‖Ax − b‖² with A 40×20 Gaussian and a sparse truth.
    const s = stream('optim/lasso')
    const A = toFlat(normals(s.child('A'), [40, 20]))
    const truth = Array.from({ length: 20 }, (_, j) => (j % 5 === 0 ? 3 - j / 5 : 0))
    const noise = toFlat(normals(s.child('noise'), 40, 0, 0.1))
    const b = Array.from({ length: 40 }, (_, i) => truth.reduce((acc, t, j) => acc + A[i * 20 + j] * t, 0) + noise[i])
    const f: Objective = (x) => {
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
  const series = useMemo<XYSeries[]>(
    () =>
      (
        [
          ['ISTA', runs.a, 0],
          ['FISTA', runs.b, 1],
        ] as const
      ).map(([name, tr, slot]) => ({
        name,
        slot,
        type: 'line',
        x: tr.index,
        y: toFlat(tr.series.F).map((v) => Math.max(v - runs.best, 1e-14)),
      })),
    [runs],
  )
  const nonzero = (tr: Trace<IterateState>) => toFlat(tr.steps.at(-1)!.x).filter((v) => v !== 0).length
  return (
    <Figure
      title="ISTA against FISTA on a lasso"
      controls={<Slider label="log₁₀ λ" value={logLambda} min={-2} max={1.5} onChange={setLogLambda} />}
      readouts={
        <>
          <Readout label="non-zero coefficients (FISTA)" value={nonzero(runs.b)} />
          <Readout label="of" value={20} />
        </>
      }
      caption="F(x) = ½‖Ax − b‖² + λ‖x‖₁ minus its best value found, on a log scale. ISTA's gap falls as O(1/k) and FISTA's as O(1/k²) until linear convergence sets in near the sparse solution."
    >
      <XYChart series={series} yLog xLabel="step k" yLabel="F(x_k) − F*" />
    </Figure>
  )
}
