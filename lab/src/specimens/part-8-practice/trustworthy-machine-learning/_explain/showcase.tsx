/**
 * Explaining a model (`aifn/learning/explain`): a random forest, a single tree or a small MLP trained on the attribution
 * task of `aifn-methods/data` (main effects on x₁ and x₂, an interaction x₃x₄, two irrelevant features). Click a point
 * to explain it by TreeSHAP, KernelSHAP, LIME, integrated gradients and SmoothGrad; then KernelSHAP's sampled estimate
 * against exact Shapley values as the coalitions grow, and the global view of permutation importance and partial
 * dependence with ICE curves. The MLP trains in the worker (`fullBatchComparison`); trees fit on the page.
 */
import { useMemo, useState } from 'react'
import { attributionTask } from 'aifn-methods/data/synthetic'
import { decisionTree, randomForest } from 'aifn-methods/learning/trees-and-ensembles'
import { comparisonModel, type ComparisonNetwork, type ComparisonSnapshot } from 'aifn-methods/neural/full-batch'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, sum, toFlat, unwrap, type Tensor } from 'aifn-compute/foundation/tensor'
import {
  ensembleTreeShap,
  exactShapley,
  integratedGradients,
  interventionalValue,
  kernelShap,
  lime,
  partialDependence,
  permutationImportance,
  smoothGrad,
  treeEnsembleOutput,
  type ShapTree,
} from 'aifn-compute/learning/explain'
import { Player, StatusText } from 'aifn-render/controls'
import { ControlRow, Figure } from 'aifn-render/layout'
import {
  call,
  choice,
  int,
  pinField,
  row,
  slider,
  useComputed,
  useFigureState,
  usePinned,
  useStreamed,
  type Task,
} from 'aifn-render/state'
import { Button } from 'aifn-render/ui/button'
import { Annotation, Bars, Curve, formatNumber, Plot, Plots, Points, Readout, useAxis } from 'aifn-render/viz'

const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(digits))) : '—')
const FEATURES = ['x₁', 'x₂', 'x₃', 'x₄', 'x₅', 'x₆']
const D = 6
const BACKGROUND = 30

type ModelKind = 'forest' | 'tree' | 'mlp'
type Setup = {
  model: ModelKind
  n: number
  noise: number
  seed: number
  trees: number
  depth: number
  width: number
  steps: number
}

/** The model as plain functions: a batch output (the explained quantity) and, for the MLP, a differentiable one. */
type Explained = {
  kind: ModelKind
  /** Output per row of X [m, 6]: P(y = 1) for trees, the logit for the MLP. */
  f: (X: Tensor) => Float64Array
  /** The MLP's logit of one row, for gradients. */
  g?: (x: Tensor) => ReturnType<typeof sum>
  trees?: ShapTree[]
  quantity: string
}

const METHODS = [
  { key: 'tree', label: 'TreeSHAP (path-dependent)', slot: 2 },
  { key: 'kernel', label: 'KernelSHAP (exact, interventional)', slot: 3 },
  { key: 'lime', label: 'LIME slope × (x − x̄)', slot: 4 },
  { key: 'ig', label: 'integrated gradients', slot: 5 },
  { key: 'smooth', label: 'SmoothGrad × (x − x̄)', slot: 6 },
] as const

export function ExplainShowcase() {
  const state = useFigureState({
    data: row('1 · data', {
      n: int(400, { ge: 50, le: 5000, suggestions: [200, 400, 1000], label: 'rows n' }),
      noise: slider(0, 2, 0.5, { step: 0.05, label: 'label noise sd' }),
      seed: slider(0, 20, 0, { step: 1, label: 'seed' }),
    }),
    model: row('2 · model', {
      model: choice(
        [
          { value: 'forest', label: 'random forest' },
          { value: 'tree', label: 'decision tree' },
          { value: 'mlp', label: 'MLP (6 → h → 1, tanh)' },
        ],
        'forest',
        { label: 'model' },
      ),
      trees: int(30, { ge: 1, le: 300, suggestions: [10, 30, 100], label: 'trees', when: (v) => v.model === 'forest' }),
      depth: int(5, { ge: 1, le: 12, suggestions: [3, 5, 8], label: 'max depth', when: (v) => v.model !== 'mlp' }),
      width: int(16, {
        ge: 2,
        le: 64,
        suggestions: [8, 16, 32],
        label: 'hidden units h',
        when: (v) => v.model === 'mlp',
      }),
      steps: int(400, {
        ge: 1,
        le: 5000,
        suggestions: [200, 400, 1000],
        label: 'Adam steps',
        when: (v) => v.model === 'mlp',
      }),
    }),
    pin: pinField(),
  })
  const current: Setup = {
    model: state.model.model as ModelKind,
    n: state.data.n,
    noise: state.data.noise,
    seed: state.data.seed,
    trees: state.model.trees,
    depth: state.model.depth,
    width: state.model.width,
    steps: state.model.steps,
  }
  const [trained, setTrained] = useState<Setup | null>(null)
  const stale = trained !== null && JSON.stringify(trained) !== JSON.stringify(current)
  const shown = trained ?? current
  const data = useMemo(
    () => attributionTask(stream(`explain-${shown.seed}`), { n: shown.n, noise: shown.noise }),
    [shown.seed, shown.n, shown.noise],
  )
  const test = useMemo(
    () => attributionTask(stream(`explain-test-${shown.seed}`), { n: 1000, noise: shown.noise }),
    [shown.seed, shown.noise],
  )
  const network: ComparisonNetwork = { inputs: D, width: shown.width, depth: 1, activation: 'tanh' }
  const task = useMemo((): Task<ComparisonSnapshot> | null => {
    if (!trained || trained.model !== 'mlp') return null
    return call<ComparisonSnapshot>(
      'applied/neural/full-batch/fullBatchComparison',
      call('data/synthetic/attributionTask', call('foundation/random/stream', `explain-${trained.seed}`), {
        n: trained.n,
        noise: trained.noise,
      }),
      {
        task: 'classification',
        network: { width: trained.width, depth: 1, activation: 'tanh' },
        optimisers: ['adam'],
        iterations: trained.steps,
        adamStep: 0.02,
        batchSize: 64,
        l2: 1e-3,
        seed: trained.seed,
        checkpoints: 40,
      },
    )
  }, [trained])
  const run = useStreamed(task)
  const checkpoints = run.value?.runs[0]?.checkpoints ?? []
  const [picked, setPicked] = useState<{ task: typeof task; index: number } | null>(null)
  const cpIndex = Math.min(picked?.task === task ? picked.index : 0, Math.max(0, checkpoints.length - 1))

  // The trained model as functions of the inputs.
  const explained = useMemo((): Explained | null => {
    if (!trained) return null
    if (trained.model === 'mlp') {
      const cp = checkpoints[cpIndex]
      if (!cp) return null
      const { model, unravel } = comparisonModel(network)
      const params = unravel(cp.theta)
      return {
        kind: 'mlp',
        f: (X) => Float64Array.from(toFlat(unwrap(model.apply(params, X)) as Tensor)),
        g: (x) => sum(model.apply(params, x)),
        quantity: 'logit of P(y = 1)',
      }
    }
    const fitted =
      trained.model === 'forest'
        ? randomForest({ trees: trained.trees, maxDepth: trained.depth }).fit(data as never, {
            stream: stream(`forest-${trained.seed}`),
          }).trees
        : [decisionTree({ maxDepth: trained.depth }).fit(data as never).tree]
    const trees = fitted as unknown as ShapTree[]
    return { kind: trained.model, f: (X) => treeEnsembleOutput(trees, X, { output: 1 }), trees, quantity: 'P(y = 1)' }
  }, [trained, checkpoints, cpIndex]) // eslint-disable-line react-hooks/exhaustive-deps

  const X = useMemo(() => Float64Array.from(toFlat(data.x)), [data])
  const y = useMemo(() => Float64Array.from(toFlat(data.y!)), [data])
  const n = data.x.shape[0]
  const pin = usePinned(state.pin, (v) => state.set('pin', v), { valid: (i) => i < n })
  const focus = pin.pinned ?? 0
  const background = useMemo(() => fromData(Float64Array.from(X.slice(0, BACKGROUND * D)), [BACKGROUND, D]), [X])
  const mean = useMemo(
    () =>
      Array.from({ length: D }, (_, j) => {
        let s = 0
        for (let i = 0; i < BACKGROUND; i++) s += X[i * D + j] / BACKGROUND
        return s
      }),
    [X],
  )

  const sd = useMemo(
    () =>
      Array.from({ length: D }, (_, j) => {
        let m = 0
        for (let i = 0; i < n; i++) m += X[i * D + j] / n
        let s2 = 0
        for (let i = 0; i < n; i++) s2 += (X[i * D + j] - m) ** 2 / n
        return Math.sqrt(s2) || 1
      }),
    [X, n],
  )
  const attributions = useComputed(() => {
    if (!explained) return null
    const x = Array.from(X.slice(focus * D, (focus + 1) * D))
    const f = (T: Tensor) => explained.f(T)
    const kernel = kernelShap(f, x, background)
    const local = lime(f, x, stream(`lime-${focus}`), { scale: sd, samples: 1000 })
    const out: Record<string, number[] | null> = {
      tree: explained.trees ? Array.from(ensembleTreeShap(explained.trees, x, { output: 1 }).values) : null,
      kernel: Array.from(kernel.values),
      // LIME's slope per standard deviation, times the row's offset from the background mean in standard deviations.
      lime: Array.from(local.coefficients, (c, j) => (c * (x[j] - mean[j])) / sd[j]),
      ig: null,
      smooth: null,
    }
    let delta = NaN
    if (explained.g) {
      const ig = integratedGradients(explained.g, x, { baseline: mean, steps: 64 })
      out.ig = Array.from(ig.values)
      delta = ig.delta
      const sg = smoothGrad(explained.g, x, stream(`smooth-${focus}`), { samples: 64, noise: 0.3 })
      out.smooth = Array.from(sg.values, (v, j) => v * (x[j] - mean[j]))
    }
    return { out, base: kernel.base, output: kernel.output, limeScore: local.score, delta }
  }, [explained, focus, background, mean, sd])

  const at = attributions.value
  const methods = METHODS.filter((m) => at?.out[m.key])
  const shift = (k: number) =>
    FEATURES.map((_, j) => j + 1 + (k - (methods.length - 1) / 2) * (0.8 / Math.max(1, methods.length)))
  const top = Math.max(0.1, ...methods.flatMap((m) => at!.out[m.key]!.map(Math.abs))) * 1.1
  const ax = useAxis({ label: 'x₁', range: [-3.5, 3.5] })
  const ay = useAxis({ label: 'x₂', range: [-3.5, 3.5], equal: ax })
  const fAxis = useAxis({
    label: 'feature',
    range: [0.4, D + 0.6],
    integer: true,
    format: (v) => FEATURES[v - 1] ?? '',
  })
  const attrAxis = useAxis({
    label: `attribution to ${explained?.quantity ?? 'the output'}`,
    range: [-top, top],
    key: `${focus}-${JSON.stringify(trained)}-${cpIndex}`,
  })
  const classes = useMemo(() => Array.from(y), [y])

  const select = (p: [number, number]) => {
    let best = 0
    let bd = Infinity
    for (let i = 0; i < n; i++) {
      const dd = (X[i * D] - p[0]) ** 2 + (X[i * D + 1] - p[1]) ** 2
      if (dd < bd) {
        bd = dd
        best = i
      }
    }
    pin.toggle(best)
  }
  const progress = run.value ? run.value.done / Math.max(1, run.value.total) : 0
  const xFocus = useMemo(() => Array.from(X.slice(focus * D, (focus + 1) * D)), [X, focus])
  return (
    <>
      <Figure
        title="Explain one prediction five ways"
        purpose="Shapley values split a prediction minus the average prediction among the features fairly (TreeSHAP and KernelSHAP compute them for two value functions); LIME fits a local line; integrated gradients and SmoothGrad read the network's gradients. On this task they agree on the main effects, split the interaction between x₃ and x₄, and give the irrelevant features nothing."
        state={state}
        defaultSize="XL"
        controls={
          <ControlRow label="3 · train">
            <div className="flex flex-wrap items-center gap-3">
              {run.running ? (
                <Button size="sm" variant="destructive" aria-label="Stop" onClick={run.stop}>
                  Stop
                </Button>
              ) : (
                <Button
                  size="sm"
                  variant={!trained || stale ? 'default' : 'outline'}
                  aria-label="Train"
                  onClick={() => setTrained({ ...current })}
                >
                  {trained ? 'Retrain' : 'Train'}
                </Button>
              )}
              {shown.model === 'mlp' && (
                <div className="h-1.5 w-40 overflow-hidden rounded bg-muted" aria-busy={run.running}>
                  <div className="h-full bg-primary" style={{ width: `${100 * Math.min(1, progress)}%` }} />
                </div>
              )}
              <StatusText tone={trained && run.error ? 'error' : !trained || stale ? 'attention' : 'muted'}>
                {!trained
                  ? 'Not trained yet: choose a model, then press Train.'
                  : run.error
                    ? `failed: ${run.error}`
                    : stale
                      ? 'Settings changed since this model was trained: press Retrain.'
                      : trained.model === 'mlp'
                        ? run.stopped
                          ? `stopped at step ${checkpoints.at(-1)?.iteration ?? 0}`
                          : `step ${checkpoints.at(-1)?.iteration ?? 0} / ${trained.steps}${run.running ? '…' : ''}`
                        : `fitted ${trained.model === 'forest' ? `${trained.trees} trees` : 'one tree'} of depth ≤ ${trained.depth}`}
              </StatusText>
            </div>
            {shown.model === 'mlp' && (
              <Player
                label="checkpoint"
                value={cpIndex}
                onChange={(i) => setPicked({ task, index: i })}
                count={Math.max(1, checkpoints.length)}
                format={(p) => `step ${checkpoints[p]?.iteration ?? 0}`}
              />
            )}
          </ControlRow>
        }
        readouts={{
          [`row ${focus}${pin.pinned === null ? ' (click a point to pin another)' : ''}`]: (
            <>
              <Readout label="label y" value={String(y[focus])} />
              <Readout label={`model ${explained?.quantity ?? ''}`} value={at ? fmt(at.output) : '—'} />
              <Readout label="average over the background" value={at ? fmt(at.base) : '—'} />
              <Readout label="LIME weighted R²" value={at ? fmt(at.limeScore) : '—'} />
              {at && Number.isFinite(at.delta) && <Readout label="IG completeness gap" value={fmt(at.delta, 2)} />}
            </>
          ),
          [`features of row ${focus}`]: (
            <>
              {FEATURES.map((name, j) => (
                <Readout key={name} label={name} value={fmt(xFocus[j])} />
              ))}
            </>
          ),
        }}
        caption={`Data: aifn attributionTask (seeded): ${n} rows of six standard normal features with y = 1 when 2x₁ − 1.5x₂ + 1.5x₃x₄ + noise > 0. Press Train to fit the chosen model (trees on the page, the MLP in the worker by Adam; its player steps through checkpoints from the untrained network). Left: the rows on (x₁, x₂), coloured by label; click a point to explain it (again or Escape to unpin), the ink point is the row explained. Right: each method's attribution per feature. TreeSHAP (trees only) uses the trees' own covers to average out unknown features; KernelSHAP uses the first ${BACKGROUND} rows as background, enumerating all 62 coalitions, so it is exact; both sum to the output minus the average. LIME fits a line to 1000 Gaussian perturbations weighted by distance; its bar is the line's slope times the row's offset from the background mean, so it compares with the others. Integrated gradients and SmoothGrad × (x − x̄) (MLP only) start from the background mean; integrated gradients also sum to f(x) − f(x̄).`}
      >
        <Plots cols={2} widths={[40, 60]}>
          <Plot x={ax} y={ay} onPlotClick={select}>
            <Points
              name="rows"
              x={Array.from({ length: n }, (_, i) => X[i * D])}
              y={Array.from({ length: n }, (_, i) => X[i * D + 1])}
              group={classes}
              groupNames={['y = 0', 'y = 1']}
              thin
            />
            <Points name="explained row" x={[xFocus[0]]} y={[xFocus[1]]} emphasis size={12} />
          </Plot>
          <Plot x={fAxis} y={attrAxis}>
            {at &&
              methods.map((m, k) => (
                <Bars
                  key={m.key}
                  name={m.label}
                  x={shift(k)}
                  y={at.out[m.key]!}
                  width={0.8 / Math.max(1, methods.length)}
                  slot={m.slot}
                  stale={attributions.stale}
                />
              ))}
            <Annotation y={0} />
          </Plot>
        </Plots>
      </Figure>
      <ConvergenceSpecimen explained={explained} focus={focus} background={background} x={xFocus} data={test} />
    </>
  )
}

// ── KernelSHAP against exact Shapley values; permutation importance and partial dependence ───────────────────────────

function ConvergenceSpecimen({
  explained,
  focus,
  background,
  x,
  data,
}: {
  explained: Explained | null
  focus: number
  background: Tensor
  x: number[]
  data: { x: Tensor; y?: Tensor }
}) {
  const state = useFigureState({
    sampling: row('1 · sampling', {
      repeats: int(10, { ge: 2, le: 200, suggestions: [5, 10, 50], label: 'repeats per budget' }),
    }),
    global: row('2 · global view', {
      feature: choice(
        FEATURES.map((f, j) => ({ value: String(j), label: f })),
        '0',
        { label: 'partial dependence on' },
      ),
    }),
  })
  const { repeats } = state.sampling
  const feature = Number(state.global.feature)
  const budgets = [6, 8, 10, 14, 20, 30, 44, 60]
  const conv = useComputed(
    () => {
      if (!explained) return null
      const f = (T: Tensor) => explained.f(T)
      const exact = exactShapley(interventionalValue(f, x, background), D).values
      const err = budgets.map((b) => {
        const errs = Array.from({ length: repeats }, (_, r) => {
          const k = kernelShap(f, x, background, { samples: b, stream: stream(`ks-${focus}-${b}-${r}`) })
          return Math.max(...k.values.map((v, i) => Math.abs(v - exact[i])))
        }).sort((a, c) => a - c)
        return { median: errs[Math.floor(errs.length / 2)], hi: errs[Math.floor(0.9 * (errs.length - 1))] }
      })
      const scale = Math.max(...Array.from(exact, Math.abs))
      return { err, scale }
    },
    [explained, focus, background, repeats],
    { mode: 'release' },
  )
  const global = useComputed(() => {
    if (!explained) return null
    const f = (T: Tensor) => explained.f(T)
    const yv = Float64Array.from(toFlat(data.y!))
    const threshold = explained.kind === 'mlp' ? 0 : 0.5
    const accuracy = (t: Float64Array, p: Float64Array) =>
      p.reduce((s, v, i) => s + ((v > threshold ? 1 : 0) === t[i] ? 1 : 0), 0) / t.length
    const imp = permutationImportance(f, data.x, yv, accuracy, { repeats: 5, stream: stream('permutation') })
    const pd = partialDependence(f, data.x, feature, { resolution: 40 })
    const ice = toFlat(pd.individual)
    const g = pd.grid.length
    const lines = Array.from({ length: 40 }, (_, i) => Array.from({ length: g }, (_, k) => ice[i * g + k]))
    return { imp, grid: Array.from(pd.grid), average: Array.from(pd.average), lines }
  }, [explained, data, feature])
  const c = conv.value
  const gl = global.value
  const bAxis = useAxis({ label: 'coalitions sampled (exact at 62)', range: [0, 65] })
  const eAxis = useAxis({ label: 'largest |φ̂ᵢ − φᵢ|', hold: 'union', key: `${focus}` })
  const fAxis = useAxis({
    label: 'feature',
    range: [0.4, D + 0.6],
    integer: true,
    format: (v) => FEATURES[v - 1] ?? '',
  })
  const iAxis = useAxis({ label: 'accuracy drop when shuffled', hold: 'union' })
  const gAxis = useAxis({ label: FEATURES[feature], range: [-2, 2], key: feature })
  const pAxis = useAxis({ label: explained?.quantity ?? 'output', hold: 'union', key: `${feature}-${explained?.kind}` })
  return (
    <Figure
      title="KernelSHAP converges to the exact Shapley values; the global view"
      purpose="KernelSHAP estimates Shapley values by a kernel-weighted regression over sampled coalitions, so its error falls as the coalitions grow and vanishes when all 62 are enumerated; summed over the data, permutation importance and partial dependence show which features matter overall and how."
      state={state}
      defaultSize="L"
      readouts={{
        [`row ${focus}`]: (
          <>
            <Readout label="largest |exact Shapley value|" value={c ? fmt(c.scale) : '—'} />
            <Readout label="median error at 10 coalitions" value={c ? fmt(c.err[2].median) : '—'} />
            <Readout label="median error at 30 coalitions" value={c ? fmt(c.err[5].median) : '—'} />
          </>
        ),
        importance: (
          <>
            {FEATURES.map((name, j) => (
              <Readout
                key={name}
                label={name}
                value={gl ? `${fmt(gl.imp.mean[j], 2)} ± ${fmt(gl.imp.std[j], 1)}` : '—'}
              />
            ))}
          </>
        ),
      }}
      caption={`Uses the model trained above and the row pinned there. Left: KernelSHAP's largest error against the exact Shapley values (enumeration of all 2⁶ coalitions), median (line) and 90th percentile (dashed) over the given number of seeded repeats per budget; coalitions are drawn by size from the Shapley kernel with their complements. Middle: permutation importance, the drop in accuracy on 1000 fresh rows when one feature's column is shuffled (mean of five shuffles); the irrelevant x₅ and x₆ stay near zero, and the interaction pair shows up although neither has a main effect. Right: forty ICE curves (thin), each one row's prediction as the chosen feature sweeps its 5–95% range, and their mean, the partial dependence (ink). The interaction features have a flat partial dependence but fanning ICE curves: their effect depends on the other feature's sign. Recomputed on release.`}
    >
      <Plots cols={3}>
        <Plot x={bAxis} y={eAxis}>
          {c && (
            <Curve
              name="median error"
              x={budgets}
              y={c.err.map((e) => e.median)}
              slot={3}
              showPoints
              stale={conv.stale}
            />
          )}
          {c && (
            <Curve name="90th percentile" x={budgets} y={c.err.map((e) => e.hi)} slot={3} dashed stale={conv.stale} />
          )}
          <Annotation x={62} dashed text="exact" />
        </Plot>
        <Plot x={fAxis} y={iAxis}>
          {gl && (
            <Bars
              name="accuracy drop"
              x={FEATURES.map((_, j) => j + 1)}
              y={Array.from(gl.imp.mean)}
              slot={2}
              stale={global.stale}
            />
          )}
        </Plot>
        <Plot x={gAxis} y={pAxis}>
          {gl && gl.lines.map((l, i) => <Curve key={i} name="ICE" x={gl.grid} y={l} muted thin silent />)}
          {gl && (
            <Curve name="partial dependence" x={gl.grid} y={gl.average} emphasis width={2.5} stale={global.stale} />
          )}
        </Plot>
      </Plots>
    </Figure>
  )
}
