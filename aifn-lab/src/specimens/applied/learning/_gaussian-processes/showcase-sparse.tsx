import { regression1d } from 'aifn-applied/data/synthetic'
import {
  gpPosterior,
  sparseGpAt,
  type SparseGpFitState,
  type SparseGpGrowState,
  type SparseMethod,
} from 'aifn-applied/learning/gaussian-processes'
import { stream } from 'aifn/foundation/random'
import { fromData, linspace, toFlat, type Tensor } from 'aifn/foundation/tensor'
import type { Trace } from 'aifn/foundation/trace'
import { rbf } from 'aifn/learning/kernels'
import { memo, useCallback, useMemo, useRef, useState } from 'react'
import { Player } from '@lab/controls'
import { Dashboard, DashboardCell, DashboardRow, Equation, Figure, live, tex } from '@lab/layout'
import { call, choice, float, int, row, useComputed, useFigureState, when } from '@lab/state'
import { Area, Curve, Handle, Plot, Plots, Points, Readout, useAxis, type AxisModel } from '@lab/viz'
import { band, f3, usePlayed } from './incremental'

const GP = 'learning/gaussian-processes'
const RANGE: [number, number] = [0, 10]
const GRID = linspace(RANGE[0], RANGE[1], 201)
const GRID_X = toFlat(GRID)
/** Where the inducing inputs are marked, just above the bottom of the data plot. */
const MARK_Y = -2.6
/** The kernel and noise every run starts from (and holds, when growing without re-optimisation). */
const START = { lengthscale: 1, variance: 1, noise: 0.1 }
const TEMPLATE = rbf({ lengthscale: START.lengthscale, variance: START.variance })
/** L-BFGS steps in "optimise all". */
const STEPS = 120
/** Candidate training inputs scored at each greedy step. */
const CANDIDATES = 60
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))

const METHODS = [
  { value: 'vfe' as const, label: 'VFE (Titsias)' },
  { value: 'fitc' as const, label: 'FITC' },
  { value: 'dtc' as const, label: 'DTC' },
  { value: 'sor' as const, label: 'SoR' },
]
const MODES = [
  { value: 'grow' as const, label: 'grow greedily' },
  { value: 'optimise' as const, label: 'optimise all (L-BFGS)' },
]

const SCHEMA = {
  data: row('1 · data', {
    n: int(120, { ge: 20, le: 300, label: 'n' }),
    noise: float(0.2, { ge: 0.02, le: 0.6, label: 'noise sd' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  }),
  approx: row('2 · approximation and fit', {
    method: choice(METHODS, 'vfe', { label: 'method' }),
    mode: choice(MODES, 'grow', { label: 'mode' }),
    m: int(8, { ge: 2, le: 30, label: 'inducing inputs m', when: when('mode', 'optimise') }),
    maxM: int(15, { ge: 2, le: 40, label: 'grow to m', when: when('mode', 'grow') }),
    reoptimise: choice([0, 3, 10], 0, { label: 'L-BFGS steps per addition', when: when('mode', 'grow') }),
  }),
}

/** One played state, as plain numbers. */
type Frame = {
  z: number[]
  /** Whether the last inducing input was added at this step (growth). */
  added: boolean
  logMarginal: number
  exact: number
  lengthscale: number
  variance: number
  noise: number
  logKernel: Float64Array
}

const frameOf = (s: SparseGpFitState | SparseGpGrowState, added: boolean): Frame => ({
  z: Array.from(toFlat(s.inducing)),
  added,
  logMarginal: s.logMarginal,
  exact: s.exact ?? NaN,
  lengthscale: s.hyper.lengthscale,
  variance: s.hyper.variance,
  noise: s.noiseVariance,
  logKernel: s.logKernel,
})

export function SparseGpShowcase() {
  const state = useFigureState(SCHEMA)
  const { n, noise, seed } = state.data
  const method = state.approx.method as SparseMethod
  const mode = state.approx.mode as 'grow' | 'optimise'
  const { m, maxM, reoptimise } = state.approx

  const data = useMemo(
    () => regression1d(stream(`sparse-gp-${seed}`), { n, fn: 'sine', range: RANGE, spacing: 'gapped', noise }),
    [n, noise, seed],
  )
  const X = data.x as Tensor
  const Y = data.y as Tensor
  const points = useMemo(() => ({ x: Array.from(toFlat(X)), y: Array.from(toFlat(Y)) }), [X, Y])

  // ── The run, in a Web Worker: every state of the L-BFGS fit or of the greedy growth ──
  const kernelTask = call('learning/kernels/rbf', { lengthscale: START.lengthscale, variance: START.variance })
  const runKey = JSON.stringify([n, noise, seed, method, mode, mode === 'grow' ? [maxM, reoptimise] : m])
  const run = useComputed(
    () => {
      if (mode === 'optimise') {
        // Z starts evenly spaced over the input range, some of it in the gap.
        const z0 = fromData(Float64Array.from(toFlat(linspace(0.5, 9.5, m))), [m, 1])
        return call<Trace<SparseGpFitState>>(
          'foundation/trace/trace',
          call(`${GP}/sparseGpFitSteps`, kernelTask, X, Y, z0, { method, noiseVariance: START.noise, exact: true }),
          undefined,
          STEPS,
        )
      }
      return call<Trace<SparseGpGrowState>>(
        'foundation/trace/trace',
        call(`${GP}/sparseGpGrowSteps`, kernelTask, X, Y, {
          method,
          noiseVariance: START.noise,
          maxInducing: maxM,
          candidates: CANDIDATES,
          reoptimise,
          exact: true,
        }),
        undefined,
        maxM,
        { stream: call('foundation/random/stream', `sparse-grow-${seed}`) },
      )
    },
    [runKey],
    {
      mode: 'worker',
      initial: null as { key: string; grow: boolean; frames: Frame[] } | null,
      then: (tr: Trace<SparseGpFitState | SparseGpGrowState>) => ({
        key: runKey,
        grow: mode === 'grow',
        frames: tr.steps.map((s) => frameOf(s, mode === 'grow' && (s as SparseGpGrowState).added !== null)),
      }),
      cancelAfter: 400,
    },
  )
  const frames = useMemo(() => run.value?.frames ?? [], [run.value])
  const grow = run.value?.grow ?? mode === 'grow'
  const [pos, setPos] = usePlayed(run.value?.key ?? '', Math.max(1, frames.length))
  const frame = frames[pos] ?? null

  // ── Inducing inputs dragged by hand at the shown step (cleared by any other change), committed on release ──
  const [dragged, setDragged] = useState<{ key: string; z: number[] } | null>(null)
  const dragKey = `${run.value?.key}|${pos}`
  const isDragged = dragged !== null && dragged.key === dragKey
  const z = useMemo(() => (isDragged ? dragged.z : (frame?.z ?? [])), [isDragged, dragged, frame])
  const commitZ = useCallback((zs: number[]) => setDragged({ key: dragKey, z: zs }), [dragKey])

  // ── The exact GP at the shown hyperparameters, and the sparse GP at the shown (or dragged) Z ──
  const exact = useComputed(() => {
    if (!frame) return null
    const post = gpPosterior(rbf({ lengthscale: frame.lengthscale, variance: frame.variance }), X, Y, {
      noiseVariance: frame.noise,
    })
    const p = post.predict(GRID)
    return { logMarginal: post.logMarginal.value, ...band(toFlat(p.mean), toFlat(p.variance)) }
  }, [frame, X, Y])
  const model = useMemo(
    () => (frame && z.length > 0 ? sparseAt(frame, z, X, Y, method) : null),
    [frame, z, X, Y, method],
  )
  const ex = exact.value
  const objective = model?.logMarginal ?? NaN
  const gap = (ex?.logMarginal ?? NaN) - objective

  // ── Curves over the run: the objective, the exact evidence at the same hyperparameters, and the gap ──
  const series = useMemo(() => {
    const xs = frames.map((f, t) => (grow ? f.z.length : t))
    return {
      x: xs,
      objective: frames.map((f) => f.logMarginal),
      exact: frames.map((f) => f.exact),
      gap: frames.map((f) => Math.max(f.exact - f.logMarginal, 1e-3)),
    }
  }, [frames, grow])
  const at = series.x[pos] ?? 0
  const nowMarks = useMemo(
    () => ({
      objective: { x: [at], y: [objective] },
      gap: { x: [at], y: [Math.max(gap, 1e-3)] },
    }),
    [at, objective, gap],
  )

  const xa = useAxis({ label: 'x', range: RANGE })
  const fa = useAxis({ label: 'y', range: [-3, 3] })
  const stepRange = useMemo(
    (): [number, number] => (series.x.length > 1 ? [series.x[0], series.x[series.x.length - 1]] : [0, 1]),
    [series.x],
  )
  const steps = useAxis({ label: grow ? 'inducing inputs m' : 'L-BFGS step', range: stepRange, key: run.value?.key })
  const nats = useAxis({ label: 'nats', hold: 'initial', key: run.value?.key })
  const gapAxis = useAxis({ label: 'exact − approximate (nats)', log: true, range: [1e-3, undefined] })

  const name = method === 'vfe' ? 'ELBO 𝓕' : `${method.toUpperCase()} log marginal`
  const equation = (() => {
    const terms = model?.terms
    const data = terms ? terms.fit + terms.complexity + terms.constant : NaN
    const Q = String.raw`\mathbf Q_{nn} = \mathbf K_{nm}\mathbf K_{mm}^{-1}\mathbf K_{mn}`
    if (method === 'vfe')
      return (
        <Equation>
          {tex`\begin{aligned} &${Q}, \quad \mathcal F = \log\mathcal N(\mathbf y \mid \mathbf 0, \mathbf Q_{nn} + \sigma^2\mathbf I) - \tfrac{1}{2\sigma^2}\operatorname{tr}(\mathbf K_{nn} - \mathbf Q_{nn}) = ${live(data, { digits: 4 })} + (${live(terms?.trace ?? NaN, { digits: 3 })}) \\ &= ${live(objective, { digits: 4, strong: true })}, \qquad \log p(\mathbf y) - \mathcal F = \mathrm{KL}\big[q(\mathbf f, \mathbf u) \,\|\, p(\mathbf f, \mathbf u \mid \mathbf y)\big] = ${live(gap, { digits: 3 })} \end{aligned}`}
        </Equation>
      )
    const cov =
      method === 'fitc'
        ? String.raw`\mathbf Q_{nn} + \operatorname{diag}(\mathbf K_{nn} - \mathbf Q_{nn}) + \sigma^2\mathbf I`
        : String.raw`\mathbf Q_{nn} + \sigma^2\mathbf I`
    return (
      <Equation>
        {tex`\begin{aligned} &${Q}, \quad \log q(\mathbf y) = \log\mathcal N(\mathbf y \mid \mathbf 0, ${cov}) = ${live(objective, { digits: 4, strong: true })}, \\ & \log p(\mathbf y) - \log q(\mathbf y) = ${live(gap, { digits: 3 })} \end{aligned}`}
      </Equation>
    )
  })()

  return (
    <Figure
      title="Sparse GPs, inducing point by inducing point"
      purpose="A sparse GP sees the data only through its m inducing inputs: grown greedily or optimised by L-BFGS, they settle where the data are dense and informative, and the gap to the exact GP's evidence closes."
      state={state}
      defaultSize="XL"
      equation={equation}
      controls={
        <Player
          label="3 · play"
          value={pos}
          onChange={setPos}
          count={Math.max(1, frames.length)}
          duration={grow ? 6 : 10}
          format={(p) => (grow ? `m = ${frames[p]?.z.length ?? '…'}` : `step ${p}`)}
        />
      }
      readouts={{
        'at this step': (
          <>
            <Readout label="m" value={z.length} />
            <Readout label={name} value={f3(objective)} />
            <Readout label="exact log p(y)" value={f3(ex?.logMarginal ?? NaN)} />
            <Readout label={method === 'vfe' ? 'gap = KL' : 'gap'} value={f3(gap)} />
            {isDragged && <Readout label="Z" value="dragged by hand" />}
          </>
        ),
        hyperparameters: (
          <>
            <Readout label="ℓ" value={f3(frame?.lengthscale ?? NaN)} />
            <Readout label="σ_f²" value={f3(frame?.variance ?? NaN)} />
            <Readout label="σ²" value={f3(frame?.noise ?? NaN)} />
            {(run.stale || !run.value) && !run.error && (
              <Readout label="run" value={<span aria-busy="true">computing in a worker…</span>} />
            )}
            {run.error && <Readout label="run failed" value={run.error} />}
          </>
        ),
      }}
      caption={`${n} noisy observations of sin x on [0, 10], dense on the left, sparse on the right and none in between (aifn’s regression1d, spacing “gapped”). Top: the sparse GP’s mean ± 2 sd (blue band) against the exact GP with the same hyperparameters (ink, dashed), and the inducing inputs Z as markers along the bottom, the one just added ringed in ink. “Grow greedily” starts from one inducing input and adds, at each step, the one of ${CANDIDATES} candidate training inputs that most raises the objective (each candidate scored exactly at O(nm²)); with L-BFGS steps per addition the hyperparameters and Z are then re-optimised. “Optimise all” starts Z evenly spaced (some in the gap) and plays L-BFGS on the kernel's log hyperparameters, log σ² and Z together, with gradients by autodiff. Bottom left: the objective (VFE's ELBO 𝓕, otherwise the approximation's log marginal likelihood) and the exact log marginal likelihood at the same hyperparameters (ink, dashed). Bottom right: their gap on a log scale; for VFE it is KL(q ‖ p) ≥ 0 and it never grows as Z grows. FITC's and DTC's objectives are not bounds and can exceed the exact evidence (the gap is then drawn at its floor). Drag an inducing input's marker to move it at the shown step (the readouts and the equation follow on release); the play or any other change returns to the run.`}
    >
      <Dashboard>
        <DashboardRow ratio={1.25} minHeight={300}>
          <DashboardCell>
            <DataPlot
              xa={xa}
              fa={fa}
              frame={frame}
              z={z}
              X={X}
              Y={Y}
              points={points}
              method={method}
              exact={ex}
              grow={grow}
              onCommit={commitZ}
            />
          </DashboardCell>
        </DashboardRow>
        <DashboardRow minHeight={220}>
          <DashboardCell>
            <Plots cols={2}>
              <Plot x={steps} y={nats}>
                <Curve name={name} x={series.x} y={series.objective} slot={0} showPoints={grow} />
                <Curve name="exact log p(y)" x={series.x} y={series.exact} emphasis dashed />
                <Points name="shown" x={nowMarks.objective.x} y={nowMarks.objective.y} emphasis live />
              </Plot>
              <Plot x={steps} y={gapAxis}>
                <Curve name="gap" x={series.x} y={series.gap} slot={2} showPoints={grow} />
                <Points name="shown" x={nowMarks.gap.x} y={nowMarks.gap.y} emphasis live />
              </Plot>
            </Plots>
          </DashboardCell>
        </DashboardRow>
      </Dashboard>
    </Figure>
  )
}

/** The sparse GP at a frame's hyperparameters and inducing inputs z. */
function sparseAt(frame: Frame, z: number[], X: Tensor, Y: Tensor, method: SparseMethod) {
  const inducing = fromData(Float64Array.from(z), [z.length, 1])
  return sparseGpAt(TEMPLATE, X, Y, { logKernel: frame.logKernel, noiseVariance: frame.noise, inducing }, { method })
}

type DataPlotProps = {
  xa: AxisModel
  fa: AxisModel
  frame: Frame | null
  /** The shown inducing inputs (the run's, or the last ones dragged and released). */
  z: number[]
  X: Tensor
  Y: Tensor
  points: { x: number[]; y: number[] }
  method: SparseMethod
  exact: { mean: number[]; upper: number[]; lower: number[] } | null
  grow: boolean
  onCommit: (z: number[]) => void
}

/**
 * The data plot, apart from the figure: a drag moves Z in this component's own state, so only this plot re-renders
 * while the pointer moves (the sparse band by patch); the figure's readouts and curves follow on release.
 */
const DataPlot = memo(function DataPlot({
  xa,
  fa,
  frame,
  z,
  X,
  Y,
  points,
  method,
  exact,
  grow,
  onCommit,
}: DataPlotProps) {
  // Positions while a drag is under way, for the z they started from (a new z from the figure drops them).
  const [moving, setMoving] = useState<{ from: number[]; z: number[] } | null>(null)
  const zs = moving && moving.from === z ? moving.z : z
  const sparse = useComputed(() => {
    if (!frame || zs.length === 0) return null
    const p = sparseAt(frame, zs, X, Y, method).predict(GRID)
    return band(toFlat(p.mean), toFlat(p.variance))
  }, [frame, zs, X, Y, method])
  const sp = sparse.value
  const marks = useMemo(() => {
    const newest = frame?.added ? [zs[zs.length - 1]] : []
    return { all: { x: zs, y: zs.map(() => MARK_Y) }, newest: { x: newest, y: newest.map(() => MARK_Y) } }
  }, [zs, frame])
  const latest = useRef<number[] | null>(null)
  const move = (i: number, u: number) => {
    latest.current = zs.map((w, j) => (j === i ? clamp(u, RANGE[0], RANGE[1]) : w))
    setMoving({ from: z, z: latest.current })
  }
  const release = () => {
    if (latest.current) onCommit(latest.current)
    latest.current = null
  }
  return (
    <Plot x={xa} y={fa}>
      <Points name="data" x={points.x} y={points.y} muted thin />
      {exact && (
        <>
          <Curve name="exact GP mean" x={GRID_X} y={exact.mean} emphasis dashed />
          <Curve name="exact ± 2 sd" x={GRID_X} y={exact.upper} emphasis dashed thin />
          <Curve name="exact ± 2 sd" x={GRID_X} y={exact.lower} emphasis dashed thin />
        </>
      )}
      {sp && (
        <>
          <Area
            name="sparse ± 2 sd"
            x={GRID_X}
            y={sp.upper}
            base={sp.lower}
            slot={0}
            opacity={0.2}
            line={false}
            stale={sparse.stale}
            live
          />
          <Curve name="sparse mean" x={GRID_X} y={sp.mean} slot={0} stale={sparse.stale} live />
        </>
      )}
      <Points name="inducing inputs Z" x={marks.all.x} y={marks.all.y} slot={1} live />
      {grow && (
        <Points
          name="just added"
          x={marks.newest.x}
          y={marks.newest.y}
          size={16}
          colors={marks.newest.x.map(() => 'rgba(0,0,0,0)')}
          live
        />
      )}
      {zs.map((v, i) => (
        <Handle key={i} kind="point" at={[v, MARK_Y]} onDrag={([u]) => move(i, u)} onRelease={release} />
      ))}
    </Plot>
  )
})
