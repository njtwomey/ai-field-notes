import { regression1d } from 'aifn-applied/data/synthetic'
import type { RegressionTruth } from 'aifn-applied/data'
import { rvmModel, rvmPosterior, rvmProblem, type RvmState } from 'aifn-applied/learning/gaussian-processes'
import { stream } from 'aifn/foundation/random'
import { linspace, reshape, toFlat, type Tensor } from 'aifn/foundation/tensor'
import type { Trace } from 'aifn/foundation/trace'
import { matern12, matern32, matern52, rbf } from 'aifn/learning/kernels'
import { useMemo, useState } from 'react'
import { Player, StatusText } from '@lab/controls'
import { Dashboard, DashboardCell, DashboardRow, Equation, Figure, live, tex } from '@lab/layout'
import { call, choice, float, int, row, useComputed, useFigureState } from '@lab/state'
import { Annotation, Area, Bars, Curve, Handle, Plot, Plots, Points, Readout, useAxis } from '@lab/viz'
import { band, f3, usePlayed } from './incremental'

const GP = 'learning/gaussian-processes'
const KERNELS = {
  rbf: { label: 'RBF', make: rbf },
  matern52: { label: 'Matérn 5/2', make: matern52 },
  matern32: { label: 'Matérn 3/2', make: matern32 },
  matern12: { label: 'Matérn 1/2 (Laplacian)', make: matern12 },
} as const
type KernelKey = keyof typeof KERNELS
const DATASETS = {
  sinc: { label: 'sinc', fn: 'sinc' as const, range: [-3, 3] as [number, number], spacing: 'random' as const },
  gapped: {
    label: 'gapped sine',
    fn: 'sine' as const,
    range: [0, 6.5] as [number, number],
    spacing: 'gapped' as const,
  },
}
type DatasetKey = keyof typeof DATASETS
const ALGORITHMS = [
  { value: 'fast' as const, label: 'fast sequential (Tipping & Faul)' },
  { value: 'reestimation' as const, label: 're-estimation (Tipping 2001)' },
]
const MAX_STEPS = 400
/** log₁₀ α is drawn on [LOG_LO, LOG_HI]; a pruned (infinite) α reaches the top. */
const LOG_LO = -3
const LOG_HI = 9
const VERB: Record<RvmState['action'], string> = {
  add: 'added',
  're-estimate': 're-estimated',
  delete: 'deleted',
  noise: 'β re-estimated',
  update: 'all re-estimated',
  none: 'start',
}

const SCHEMA = {
  data: row('1 · data', {
    dataset: choice(
      (Object.keys(DATASETS) as DatasetKey[]).map((k) => ({ value: k, label: DATASETS[k].label })),
      'sinc',
      { label: 'data' },
    ),
    n: int(100, { ge: 10, le: 250, label: 'n' }),
    noise: float(0.1, { ge: 0.01, le: 0.5, label: 'noise sd' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  }),
  model: row('2 · basis and fit', {
    kernel: choice(
      (Object.keys(KERNELS) as KernelKey[]).map((k) => ({ value: k, label: KERNELS[k].label })),
      'rbf',
      { label: 'kernel' },
    ),
    width: float(0.6, { ge: 0.05, le: 3, label: 'width ℓ' }),
    algorithm: choice(ALGORITHMS, 'fast', { label: 'algorithm' }),
  }),
}

export function RvmShowcase() {
  const state = useFigureState(SCHEMA)
  const { n, noise, seed } = state.data
  const { width } = state.model
  const datasetKey = state.data.dataset as DatasetKey
  const kernelKey = state.model.kernel as KernelKey
  const algorithm = state.model.algorithm as 'fast' | 'reestimation'
  const spec = DATASETS[datasetKey]

  const data = useMemo(
    () => regression1d(stream(`rvm-${seed}`), { n, fn: spec.fn, range: spec.range, spacing: spec.spacing, noise }),
    [n, noise, seed, spec],
  )
  const X = data.x as Tensor
  const Y = data.y as Tensor
  const xs = useMemo(() => Array.from(toFlat(X)), [X])
  const ys = useMemo(() => Array.from(toFlat(Y)), [Y])
  const grid = useMemo(() => linspace(spec.range[0], spec.range[1], 241), [spec])
  const gridX = useMemo(() => Array.from(toFlat(grid)), [grid])
  const truth = useMemo(
    () => Array.from(toFlat((data.meta.truth as RegressionTruth).mean(reshape(grid, [gridX.length, 1]) as Tensor))),
    [data, grid, gridX.length],
  )
  const problem = useMemo(
    () => rvmProblem(KERNELS[kernelKey].make({ lengthscale: width }), X, Y),
    [kernelKey, width, X, Y],
  )

  // ── The fit, in a Web Worker: every state from step 0 ──
  const runKey = JSON.stringify([datasetKey, n, noise, seed, kernelKey, width, algorithm])
  const run = useComputed(
    () =>
      call<Trace<RvmState>>(
        'foundation/trace/trace',
        call(
          `${GP}/${algorithm === 'fast' ? 'rvmFastSteps' : 'rvmReestimationSteps'}`,
          call(`${GP}/rvmProblem`, call(`learning/kernels/${kernelKey}`, { lengthscale: width }), X, Y),
          {},
        ),
        undefined,
        MAX_STEPS,
      ),
    [runKey],
    {
      mode: 'worker',
      initial: null as { key: string; states: RvmState[] } | null,
      then: (tr) => ({ key: runKey, states: tr.steps }),
      cancelAfter: 400,
    },
  )
  // A run for other settings (still on its way) is not drawn over this problem's data.
  const states = useMemo(() => (run.value?.key === runKey ? run.value.states : []), [run.value, runKey])
  const [pos, setPos] = usePlayed(run.value?.key ?? '', Math.max(1, states.length))
  const s = states[pos] ?? null

  // ── The model at the shown step: predictive band, relevance vectors, and S, Q for every candidate ──
  const shown = useComputed(() => {
    if (!s) return null
    const model = rvmModel(problem, s)
    const post = rvmPosterior(problem, s.active, s.alpha, s.beta)
    const p = model.predict(grid, { noise: false })
    const pn = model.predict(grid, { noise: true })
    return { model, post, ...band(toFlat(p.mean), toFlat(pn.variance)) }
  }, [s, problem, grid])
  const view = shown.value

  // ── The probe: a draggable x that picks the nearest candidate basis function ──
  const [probeAt, setProbeAt] = useState<{ key: DatasetKey; x: number } | null>(null)
  const probeX = probeAt?.key === datasetKey ? probeAt.x : (spec.range[0] + spec.range[1]) / 2
  const offset = problem.bias ? 1 : 0
  const probeRow = useMemo(
    () => xs.reduce((b, v, i) => (Math.abs(v - probeX) < Math.abs(xs[b] - probeX) ? i : b), 0),
    [xs, probeX],
  )
  const probe = useMemo(() => {
    if (!s || !view) return null
    const i = probeRow + offset
    const a = s.alpha[i]
    const S = view.post.S[i]
    const Q = view.post.Q[i]
    const sI = a === Infinity ? S : (a * S) / (a - S)
    const qI = a === Infinity ? Q : (a * Q) / (a - S)
    const theta = qI * qI - sI
    return { i, alpha: a, s: sI, q: qI, theta, best: theta > 0 ? (sI * sI) / theta : Infinity }
  }, [s, view, probeRow, offset])

  // ── Layers ──
  const marks = useMemo(() => {
    if (!s) return null
    const rv = s.active.map(problem.centre).filter((r): r is number => r !== null)
    const acted = s.index !== null ? problem.centre(s.index) : null
    const pruned = s.pruned.map(problem.centre).filter((r): r is number => r !== null)
    const logA = (i: number) => clampLog(s.alpha[i])
    const bars = xs.map((_, r) => r + offset)
    return {
      rings: { x: rv.map((r) => xs[r]), y: rv.map((r) => ys[r]), clear: rv.map(() => 'rgba(0,0,0,0)') },
      acted: acted === null ? null : ([xs[acted], ys[acted]] as [number, number]),
      pruned: { x: pruned.map((r) => xs[r]), y: pruned.map((r) => ys[r]) },
      active: {
        x: bars.filter((i) => s.alpha[i] !== Infinity).map((i) => xs[i - offset]),
        y: bars.filter((i) => s.alpha[i] !== Infinity).map(logA),
      },
      inactive: {
        x: bars.filter((i) => s.alpha[i] === Infinity).map((i) => xs[i - offset]),
        y: bars.filter((i) => s.alpha[i] === Infinity).map(() => LOG_HI),
      },
      actedBar: acted === null ? null : { x: [xs[acted]], y: [s.index !== null ? logA(s.index) : LOG_HI] },
    }
  }, [s, problem, xs, ys, offset])
  const series = useMemo(
    () => ({
      x: states.map((_, t) => t),
      logMarginal: states.map((st) => st.logMarginal),
      count: states.map((st) => st.active.filter((i) => problem.centre(i) !== null).length),
    }),
    [states, problem],
  )
  const now = useMemo(
    () => ({ x: [pos], ml: [series.logMarginal[pos] ?? NaN], count: [series.count[pos] ?? NaN] }),
    [pos, series],
  )
  const barWidth = (0.5 * (spec.range[1] - spec.range[0])) / Math.max(n, 20)

  const xa = useAxis({ label: 'x', range: spec.range, key: datasetKey })
  const fa = useAxis({ label: 'y', range: datasetKey === 'sinc' ? [-0.8, 1.6] : [-2, 2], key: datasetKey })
  const la = useAxis({ label: 'log₁₀ αᵢ (top: ∞)', range: [LOG_LO, LOG_HI] })
  const stepAxis = useAxis({ label: 'step', range: [0, Math.max(1, states.length - 1)], key: run.value?.key })
  const mlAxis = useAxis({ label: 'log p(y | α, β)', hold: 'initial', key: run.value?.key })
  const countAxis = useAxis({ label: 'RVs', range: [0, undefined], key: run.value?.key })

  const action = s ? verb(s) : '…'
  const actedName =
    s?.index === null || s?.index === undefined ? '' : problem.centre(s.index) === null ? ' (bias)' : ` φ${s.index}`
  const equation = (
    <Equation>
      {tex`\mathcal L(\alpha_i) = \tfrac12\Big[\ln\alpha_i - \ln(\alpha_i + s_i) + \frac{q_i^2}{\alpha_i + s_i}\Big] + \text{const}, \quad s_i = ${live(probe?.s ?? NaN, { digits: 3 })}, \; q_i = ${live(probe?.q ?? NaN, { digits: 3 })}, \; q_i^2 - s_i = ${live(probe?.theta ?? NaN, { digits: 3, strong: true })} \;\Rightarrow\; \alpha_i^\star = ${probe && probe.theta > 0 ? live(probe.best, { digits: 3 }) : String.raw`\infty`}`}
    </Equation>
  )

  // The layers that do not move with the probe, kept as the same elements while it is dragged.
  const mainLayers = useMemo(
    () => (
      <>
        {view && (
          <Area
            name="predictive ± 2 sd"
            x={gridX}
            y={view.upper}
            base={view.lower}
            slot={0}
            opacity={0.2}
            line={false}
            stale={shown.stale}
          />
        )}
        <Curve name="truth" x={gridX} y={truth} emphasis dashed />
        {view && <Curve name="predictive mean" x={gridX} y={view.mean} slot={0} stale={shown.stale} />}
        <Points name="data" x={xs} y={ys} muted thin />
        {marks && (
          <>
            <Points name="relevance vectors" x={marks.rings.x} y={marks.rings.y} size={13} colors={marks.rings.clear} />
            {marks.pruned.x.length > 0 && (
              <Points name="pruned this step" x={marks.pruned.x} y={marks.pruned.y} slot={3} shape={4} />
            )}
            {marks.acted && <Annotation at={marks.acted} text={VERB[s!.action]} slot={1} />}
          </>
        )}
      </>
    ),
    [view, shown.stale, gridX, truth, xs, ys, marks, s],
  )
  const evidencePlots = useMemo(
    () => (
      <Plots rows={2} tight>
        <Plot x={stepAxis} y={mlAxis}>
          <Curve name="log evidence" x={series.x} y={series.logMarginal} slot={0} />
          <Points name="shown" x={now.x} y={now.ml} emphasis live />
        </Plot>
        <Plot x={stepAxis} y={countAxis}>
          <Curve name="relevance vectors" x={series.x} y={series.count} slot={2} />
          <Points name="shown" x={now.x} y={now.count} emphasis live />
        </Plot>
      </Plots>
    ),
    [stepAxis, mlAxis, countAxis, series, now],
  )
  const barLayers = useMemo(
    () => (
      <>
        {marks && (
          <>
            <Bars
              name="out (α = ∞)"
              x={marks.inactive.x}
              y={marks.inactive.y}
              base={LOG_HI - 0.8}
              width={barWidth}
              muted
            />
            <Bars name="in the model" x={marks.active.x} y={marks.active.y} base={LOG_LO} width={barWidth} slot={0} />
            {marks.actedBar && (
              <Bars name="acted on" x={marks.actedBar.x} y={marks.actedBar.y} base={LOG_LO} width={barWidth} emphasis />
            )}
          </>
        )}
      </>
    ),
    [marks, barWidth],
  )

  return (
    <Figure
      title="Relevance vector machine, basis by basis"
      purpose="The RVM puts a kernel basis function on every training input and gives each weight its own prior precision αᵢ; maximising the evidence sends most αᵢ to infinity, and the fast algorithm builds the sparse model one basis function at a time."
      state={state}
      defaultSize="XL"
      equation={equation}
      controls={
        <Player
          label="3 · play"
          value={pos}
          onChange={setPos}
          count={Math.max(1, states.length)}
          format={(p) => {
            const st = states[p]
            if (!st) return `step ${p}`
            const c = st.index === null ? '' : problem.centre(st.index) === null ? ' bias' : ` φ${st.index}`
            return `step ${p}: ${verb(st)}${c}`
          }}
        />
      }
      readouts={{
        'at this step': (
          <>
            <Readout label="relevance vectors" value={series.count[pos] ?? '…'} />
            <Readout label="log p(y | α, β)" value={f3(s?.logMarginal ?? NaN)} />
            <Readout label="β" value={f3(s?.beta ?? NaN)} />
            <Readout label="σ = 1/√β" value={f3(s ? 1 / Math.sqrt(s.beta) : NaN)} />
            <Readout label="action" value={`${action}${actedName}`} />
            <Readout label="gain" value={f3(s?.gain ?? NaN)} />
            {(run.stale || !run.value) && !run.error && (
              <Readout label="fit" value={<span aria-busy="true">computing in a worker…</span>} />
            )}
            {run.error && <StatusText tone="error">fit failed: {run.error}</StatusText>}
          </>
        ),
        'at the probe': (
          <>
            <Readout label="basis" value={probe ? `φ${probe.i} at x = ${f3(xs[probeRow])}` : '…'} />
            <Readout label="αᵢ" value={probe ? (probe.alpha === Infinity ? '∞ (out)' : f3(probe.alpha)) : '…'} />
            <Readout
              label="q² − s"
              value={probe ? `${f3(probe.theta)} (${probe.theta > 0 ? 'belongs' : 'pruned'})` : '…'}
            />
          </>
        ),
      }}
      caption={`${n} noisy observations of ${datasetKey === 'sinc' ? 'sinc x = sin(πx)/(πx), the classic RVM test (Tipping, 2001)' : 'sin x, dense on the left, sparse on the right and none in between'} (aifn’s regression1d). The RVM (aifn’s rvmFastSteps or rvmReestimationSteps, run in a worker) has a bias and one ${KERNELS[kernelKey].label} basis function of width ℓ on each training input, a precision αᵢ per weight and a noise precision β. Top: the predictive mean ± 2 sd (blue; the band includes the noise 1/β), the truth (ink, dashed), the relevance vectors ringed, and the basis function acted on at this step marked with its action. The fast algorithm starts from the one basis function most aligned with y and at each step adds, re-estimates or deletes the single basis function (or re-estimates β) that most raises the evidence, so the evidence never falls; re-estimation starts with every basis function and updates all the αᵢ at once, pruning those that pass 10⁹. Bottom: the log evidence and the number of relevance vectors per step, and log₁₀ αᵢ for every candidate (the short marks along the top are infinite: pruned or never added). Far from the relevance vectors (in the gap of the second dataset) the predictive sd shrinks to the noise level, the RVM's known overconfidence. Drag the probe (the vertical line) to pick a candidate: the equation shows its sparsity and quality factors sᵢ and qᵢ at this step, and it belongs in the model exactly when qᵢ² > sᵢ.`}
    >
      <Dashboard>
        <DashboardRow ratio={1.3} minHeight={300}>
          <DashboardCell>
            <Plot x={xa} y={fa}>
              {mainLayers}
              <Handle kind="x" at={probeX} label="probe" onDrag={(x) => setProbeAt({ key: datasetKey, x })} />
            </Plot>
          </DashboardCell>
        </DashboardRow>
        <DashboardRow minHeight={220}>
          <DashboardCell ratio={1.2}>{evidencePlots}</DashboardCell>
          <DashboardCell>
            <Plot x={xa} y={la}>
              {barLayers}
              <Handle kind="x" at={probeX} onDrag={(x) => setProbeAt({ key: datasetKey, x })} />
            </Plot>
          </DashboardCell>
        </DashboardRow>
      </Dashboard>
    </Figure>
  )
}

/** What a step did, in words; a step that changed nothing after the start is the converged one. */
const verb = (st: RvmState) => (st.action === 'none' && st.t > 0 ? 'converged' : VERB[st.action])

const clampLog = (a: number) => (a === Infinity ? LOG_HI : Math.max(LOG_LO, Math.min(LOG_HI, Math.log10(a))))
