/**
 * Which training points matter (`aifn/learning/explain` through `aifn-methods/learning/explanation`'s
 * `dataValuationStudy`, in the worker): two Gaussian blobs with a share of labels flipped, one L2 logistic regression,
 * and five data attributions ranking the training points by how suspect they are. The gain curves ask how many of the
 * flipped points each ranking finds among the first it inspects.
 */
import { useMemo, useState } from 'react'
import { blobs, flippedMask, withLabelNoise } from 'aifn-methods/data/synthetic'
import { VALUATION_METHODS, type ValuationMethod, type ValuationSnapshot } from 'aifn-methods/learning/explanation'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { gainCurve, rocCurve } from 'aifn-compute/learning/metrics'
import { StatusText } from 'aifn-render/controls'
import { ControlRow, Figure } from 'aifn-render/layout'
import { call, choice, float, int, row, slider, useFigureState, useStreamed, type Task } from 'aifn-render/state'
import { Button } from 'aifn-render/ui/button'
import { Curve, Plot, Plots, Points, Readout, useAxis } from 'aifn-render/viz'
import { fmt } from './mlp'

const LABELS: Record<ValuationMethod, string> = {
  influence: 'influence on validation loss',
  'self-influence': 'self-influence',
  tracin: 'TracIn self-influence',
  'knn-shapley': 'KNN-Shapley',
  'data-shapley': 'data Shapley (TMC)',
}
type Setup = {
  n: number
  rate: number
  separation: number
  seed: number
  K: number
  permutations: number
  tolerance: number
}

export function ValuationShowcase() {
  const state = useFigureState({
    data: row('1 · data', {
      n: int(120, { ge: 20, le: 1000, suggestions: [60, 120, 300], label: 'training points n' }),
      rate: slider(0, 0.4, 0.1, { step: 0.01, label: 'label flip rate' }),
      separation: slider(1, 6, 3, { step: 0.1, label: 'blob separation (sd)' }),
      seed: slider(0, 20, 0, { step: 1, label: 'seed' }),
    }),
    methods: row('2 · methods', {
      K: int(5, { ge: 1, le: 50, suggestions: [1, 5, 15], label: 'KNN-Shapley K' }),
      permutations: int(40, { ge: 1, le: 2000, suggestions: [20, 40, 200], label: 'TMC permutations' }),
      tolerance: float(0.01, { ge: 0, le: 0.5, suggestions: [0, 0.01, 0.05], label: 'TMC truncation tolerance' }),
    }),
    view: row('3 · inspect', {
      method: choice(
        VALUATION_METHODS.map((m) => ({ value: m, label: LABELS[m] })),
        'influence',
        { label: 'ranking shown' },
      ),
      budget: slider(1, 60, 12, { step: 1, label: 'points inspected' }),
    }),
  })
  const current: Setup = { ...state.data, ...state.methods }
  const [trained, setTrained] = useState<Setup | null>(null)
  const stale = trained !== null && JSON.stringify(trained) !== JSON.stringify(current)
  const shown = trained ?? current
  const blobOptions = (t: Setup, n: number) => ({ centers: 2, n, separation: t.separation, sd: 1, layout: 'polygon' })
  const train = useMemo(
    () =>
      withLabelNoise(
        stream(`val-flip-${shown.seed}`),
        blobs(stream(`val-${shown.seed}`), blobOptions(shown, shown.n) as never),
        {
          rate: shown.rate,
        },
      ),
    [shown.seed, shown.n, shown.rate, shown.separation], // eslint-disable-line react-hooks/exhaustive-deps
  )
  const flipped = useMemo(() => Array.from(flippedMask(train)), [train])
  const task = useMemo((): Task<ValuationSnapshot> | null => {
    if (!trained) return null
    const S = (k: string) => call('foundation/random/stream', k)
    return call<ValuationSnapshot>(
      'applied/learning/explanation/dataValuationStudy',
      call(
        'data/synthetic/withLabelNoise',
        S(`val-flip-${trained.seed}`),
        call('data/synthetic/blobs', S(`val-${trained.seed}`), blobOptions(trained, trained.n)),
        {
          rate: trained.rate,
        },
      ),
      call('data/synthetic/blobs', S(`val-valid-${trained.seed}`), blobOptions(trained, 200)),
      call('applied/learning/generalised/glm/logisticRegression', { l2: 1 }),
      { K: trained.K, permutations: trained.permutations, tolerance: trained.tolerance, seed: trained.seed },
    )
  }, [trained]) // eslint-disable-line react-hooks/exhaustive-deps
  const run = useStreamed(task)
  const r = run.value
  const X = useMemo(() => Float64Array.from(toFlat(train.x)), [train])
  const y = useMemo(() => Array.from(toFlat(train.y!)), [train])
  const n = y.length
  const method = state.view.method as ValuationMethod
  const budget = Math.min(state.view.budget, n)
  const flips = flipped.reduce((a, b) => a + b, 0)
  const ranked = useMemo(() => {
    const s = r?.suspicion[method]
    if (!s) return null
    return Array.from({ length: n }, (_, i) => i).sort((a, b) => s[b] - s[a])
  }, [r, method, n])
  const top = ranked ? ranked.slice(0, budget) : []
  const found = top.reduce((a, i) => a + flipped[i], 0)
  const curves = useMemo(() => {
    if (!r || flips === 0) return null
    return VALUATION_METHODS.map((m) => {
      const s = r.suspicion[m]
      if (!s) return null
      const g = gainCurve(flipped, s, { positive: 1 })
      return {
        x: Array.from(toFlat(g.x)),
        y: Array.from(toFlat(g.y)),
        auroc: rocCurve(flipped, s, { positive: 1 }).area,
      }
    })
  }, [r, flipped, flips])
  const box = useMemo(() => {
    const xs = Array.from({ length: n }, (_, i) => X[2 * i])
    const ys = Array.from({ length: n }, (_, i) => X[2 * i + 1])
    return {
      x: [Math.min(...xs) - 0.5, Math.max(...xs) + 0.5] as [number, number],
      y: [Math.min(...ys) - 0.5, Math.max(...ys) + 0.5] as [number, number],
    }
  }, [X, n])
  const ax = useAxis({ label: 'x₁', range: box.x, key: `${shown.seed}-${shown.separation}-${shown.n}` })
  const ay = useAxis({ label: 'x₂', range: box.y, key: `${shown.seed}-${shown.separation}-${shown.n}`, equal: ax })
  const qAxis = useAxis({ label: 'fraction of training points inspected', range: [0, 1] })
  const gAxis = useAxis({ label: 'fraction of flipped labels found', range: [0, 1] })
  // The fitted boundary θ₁x₁ + θ₂x₂ + b = 0 across the plot.
  const boundary = useMemo(() => {
    if (!r) return null
    const [a, b, c] = r.theta
    // Sample the line and keep the part inside the plotted box.
    const xs = Array.from({ length: 201 }, (_, k) => box.x[0] + ((box.x[1] - box.x[0]) * k) / 200)
    const pts =
      Math.abs(b) > 1e-9 ? xs.map((v) => [v, -(a * v + c) / b]).filter(([, v]) => v >= box.y[0] && v <= box.y[1]) : []
    return pts.length > 1 ? { x: pts.map((p) => p[0]), y: pts.map((p) => p[1]) } : null
  }, [r, box])
  const progress = r
    ? r.stage === 'done'
      ? 1
      : (VALUATION_METHODS.indexOf(r.stage as ValuationMethod) + 1 + r.done / Math.max(1, r.total)) / 6
    : 0
  return (
    <Figure
      title="Finding mislabelled points by data attribution"
      purpose="A mislabelled training point pulls the model towards the wrong class: it harms the validation loss (influence), the fit bends to it (self-influence, TracIn), its neighbours disagree with it (KNN-Shapley), and adding it lowers the validation accuracy (data Shapley). Ranking by each finds the flipped labels well before a random search would."
      defaultSize="XL"
      state={state}
      controls={
        <ControlRow label="4 · run">
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
                {trained ? 'Rerun' : 'Train'}
              </Button>
            )}
            <div className="h-1.5 w-40 overflow-hidden rounded bg-muted" aria-busy={run.running}>
              <div className="h-full bg-primary" style={{ width: `${100 * Math.min(1, progress)}%` }} />
            </div>
            <StatusText tone={trained && run.error ? 'error' : !trained || stale ? 'attention' : 'muted'}>
              {!trained
                ? 'Not run yet: press Train to fit the model and value every training point.'
                : run.error
                  ? `failed: ${run.error}`
                  : stale
                    ? 'Settings changed since this run: press Rerun.'
                    : r
                      ? `${r.stage === 'done' ? 'done' : `computing ${r.stage}…`} · TMC permutations ${r.done} / ${r.total} · ${r.calls} model fits`
                      : 'starting…'}
            </StatusText>
          </div>
        </ControlRow>
      }
      readouts={{
        [`${LABELS[method]}: the first ${budget} points inspected`]: (
          <>
            <Readout label="flipped labels found" value={ranked ? `${found} of ${flips}` : '—'} />
            <Readout label="expected by chance" value={fmt((budget * flips) / Math.max(1, n), 2)} />
            <Readout label="validation accuracy of the fit" value={r ? fmt(r.accuracy) : '—'} />
          </>
        ),
        'AUROC of each ranking for the flipped labels': (
          <>
            {VALUATION_METHODS.map((m, k) => (
              <Readout key={m} label={LABELS[m]} value={curves?.[k] ? fmt(curves[k]!.auroc) : '—'} />
            ))}
          </>
        ),
      }}
      caption={`Data: aifn blobs (seeded): ${n} training points in two Gaussian blobs ${shown.separation} sd apart, with each label flipped with probability ${shown.rate} (${flips} flipped, marked by ink triangles), and 200 clean validation points. Press Train to run the study in the worker: an L2 logistic regression (ink line) is fitted; influence functions (exact Hessian) predict the change of the validation loss when each point is removed; self-influence ∇ℓᵀH⁻¹∇ℓ and TracIn (the sum over gradient-descent checkpoints of the step size times ‖∇ℓ‖²) measure how much each point bends the fit; KNN-Shapley values each point exactly for a K-nearest-neighbour classifier on the validation set; and TMC data Shapley averages each point's marginal gain in validation accuracy over random orders, refitting the logistic regression on every prefix (its curve sharpens as permutations arrive). Left: the ${budget} most suspect points under the chosen ranking are ringed. Right: the gain curve of each ranking, the share of flipped labels found against the share of points inspected; the diagonal is random inspection.`}
    >
      <Plots cols={2} widths={[50, 50]}>
        <Plot x={ax} y={ay}>
          {top.length > 0 && (
            <Points
              name="inspected"
              x={top.map((i) => X[2 * i])}
              y={top.map((i) => X[2 * i + 1])}
              slot={VALUATION_METHODS.indexOf(method) + 2}
              size={16}
            />
          )}
          <Points
            name="training points"
            x={Array.from({ length: n }, (_, i) => X[2 * i])}
            y={Array.from({ length: n }, (_, i) => X[2 * i + 1])}
            group={y}
            groupNames={['observed y = 0', 'observed y = 1']}
          />
          <Points
            name="flipped label"
            x={flipped.flatMap((f, i) => (f ? [X[2 * i]] : []))}
            y={flipped.flatMap((f, i) => (f ? [X[2 * i + 1]] : []))}
            emphasis
            shape={2}
            size={7}
          />
          {boundary && <Curve name="fitted boundary" x={boundary.x} y={boundary.y} emphasis />}
        </Plot>
        <Plot x={qAxis} y={gAxis}>
          <Curve name="random" x={[0, 1]} y={[0, 1]} muted dashed />
          {curves &&
            VALUATION_METHODS.map((m, k) =>
              curves[k] ? <Curve key={m} name={LABELS[m]} x={curves[k]!.x} y={curves[k]!.y} slot={k + 2} /> : null,
            )}
        </Plot>
      </Plots>
    </Figure>
  )
}
