/**
 * Rule explanations (`aifn/learning/explain`): a random forest on the two moons explained locally by an anchor (a box
 * around a pinned point inside which the forest keeps its prediction with precision ≥ τ), and globally by surrogate
 * trees of growing depth and a decision list, with their fidelity to the forest on fresh data.
 */
import { useMemo, useState } from 'react'
import { moons } from 'aifn-applied/data/synthetic'
import { decisionTree, randomForest } from 'aifn-applied/learning/trees-and-ensembles'
import { stream } from 'aifn/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn/foundation/tensor'
import { dataset } from 'aifn/learning/estimators'
import {
  anchor,
  applyRuleList,
  fidelity,
  ruleList,
  treeEnsembleOutput,
  type Predicate,
  type ShapTree,
} from 'aifn/learning/explain'
import { StatusText } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { choice, int, pinField, row, slider, useComputed, useFigureState, usePinned } from '@lab/state'
import { Button } from '@lab/ui/button'
import { Annotation, Contours, Curve, Plot, Plots, Points, Raster, Readout, Segments, useAxis } from '@lab/viz'
import { fmt } from './mlp'

const XR: [number, number] = [-1.6, 2.6]
const YR: [number, number] = [-1.1, 1.6]
const GX = Array.from({ length: 71 }, (_, i) => XR[0] + ((XR[1] - XR[0]) * i) / 70)
const GY = Array.from({ length: 51 }, (_, i) => YR[0] + ((YR[1] - YR[0]) * i) / 50)
const GRID = fromData(Float64Array.from(GY.flatMap((y) => GX.map((x) => [x, y])).flat()), [GX.length * GY.length, 2])
const NAMES = ['x₁', 'x₂']
const DEPTHS = Array.from({ length: 10 }, (_, k) => k + 1)

type Setup = { n: number; noise: number; trees: number; depth: number; seed: number }

const asGrid = (v: ArrayLike<number>) =>
  GY.map((_, r) => Array.from({ length: GX.length }, (_, c) => v[r * GX.length + c]))

/** A rule as text: "a < x₁ ≤ b ∧ …". */
function ruleText(rule: readonly Predicate[]): string {
  if (rule.length === 0) return '(always)'
  return rule
    .map((p) => {
      const name = NAMES[p.feature] ?? `x${p.feature + 1}`
      if (p.lower === -Infinity) return `${name} ≤ ${fmt(p.upper, 2)}`
      if (p.upper === Infinity) return `${name} > ${fmt(p.lower, 2)}`
      return `${fmt(p.lower, 2)} < ${name} ≤ ${fmt(p.upper, 2)}`
    })
    .join(' ∧ ')
}

/** The box of a rule within the plot's range, as four segments. */
const boxOf = (rule: readonly Predicate[]) => {
  const b = { x0: XR[0], x1: XR[1], y0: YR[0], y1: YR[1] }
  for (const p of rule) {
    if (p.feature === 0) {
      b.x0 = Math.max(b.x0, p.lower)
      b.x1 = Math.min(b.x1, p.upper)
    } else {
      b.y0 = Math.max(b.y0, p.lower)
      b.y1 = Math.min(b.y1, p.upper)
    }
  }
  const c: [number, number][] = [
    [b.x0, b.y0],
    [b.x1, b.y0],
    [b.x1, b.y1],
    [b.x0, b.y1],
  ]
  return c.map((from, k) => ({ from, to: c[(k + 1) % 4] }))
}

export function RulesShowcase() {
  const state = useFigureState({
    data: row('1 · data and forest', {
      n: int(500, { ge: 50, le: 5000, suggestions: [300, 500, 1000], label: 'points n' }),
      noise: slider(0, 0.4, 0.2, { step: 0.01, label: 'moons noise' }),
      trees: int(30, { ge: 1, le: 200, suggestions: [10, 30, 100], label: 'trees' }),
      depth: int(8, { ge: 1, le: 16, suggestions: [4, 8, 12], label: 'max depth' }),
      seed: slider(0, 20, 0, { step: 1, label: 'seed' }),
    }),
    anchor: row('3 · anchor', {
      threshold: slider(0.7, 0.99, 0.95, { step: 0.01, label: 'precision τ' }),
      bins: choice([3, 4, 6, 8, 12], 4, { label: 'quantile bins per feature' }),
      beam: int(2, { ge: 1, le: 10, suggestions: [1, 2, 4], label: 'beam width' }),
    }),
    pin: pinField(),
  })
  const current: Setup = { ...state.data }
  const [trained, setTrained] = useState<Setup | null>(null)
  const stale = trained !== null && JSON.stringify(trained) !== JSON.stringify(current)
  const shown = trained ?? current
  const data = useMemo(
    () => moons(stream(`rules-${shown.seed}`), { n: shown.n, noise: shown.noise }),
    [shown.seed, shown.n, shown.noise],
  )
  const fresh = useMemo(
    () => moons(stream(`rules-test-${shown.seed}`), { n: 2000, noise: shown.noise }),
    [shown.seed, shown.noise],
  )
  const forest = useMemo(() => {
    if (!trained) return null
    const fit = randomForest({ trees: trained.trees, maxDepth: trained.depth }).fit(data as never, {
      stream: stream(`rules-forest-${trained.seed}`),
    })
    return fit.trees as unknown as ShapTree[]
  }, [trained, data])
  const proba = (Z: Tensor) => (forest ? treeEnsembleOutput(forest, Z, { output: 1 }) : new Float64Array(Z.shape[0]))
  const predict = (Z: Tensor) => Float64Array.from(proba(Z), (p) => (p > 0.5 ? 1 : 0))
  const field = useMemo(() => (forest ? asGrid(proba(GRID)) : null), [forest]) // eslint-disable-line react-hooks/exhaustive-deps
  const X = useMemo(() => Float64Array.from(toFlat(data.x)), [data])
  const y = useMemo(() => Array.from(toFlat(data.y!)), [data])
  const n = y.length
  const pin = usePinned(state.pin, (v) => state.set('pin', v), { valid: (i) => i < n })
  // Default: the row nearest the middle of the upper moon, well inside class 0.
  const fallback = useMemo(() => {
    let best = 0
    let bd = Infinity
    for (let i = 0; i < n; i++) {
      const dd = (X[2 * i] + 0.2) ** 2 + (X[2 * i + 1] - 0.95) ** 2
      if (dd < bd) {
        bd = dd
        best = i
      }
    }
    return best
  }, [X, n])
  const focus = pin.pinned ?? fallback
  const query = [X[2 * focus], X[2 * focus + 1]]
  const result = useComputed(
    () =>
      forest
        ? anchor(predict, query, data.x, stream(`anchor-${focus}`), {
            threshold: state.anchor.threshold,
            bins: state.anchor.bins,
            beam: state.anchor.beam,
          })
        : null,
    [forest, focus, data, state.anchor.threshold, state.anchor.bins, state.anchor.beam],
    { mode: 'release' },
  )
  const a = result.value
  const ax = useAxis({ label: 'x₁', range: XR, nice: false })
  const ay = useAxis({ label: 'x₂', range: YR, nice: false, equal: ax })
  const select = (p: [number, number]) => {
    let best = 0
    let bd = Infinity
    for (let i = 0; i < n; i++) {
      const dd = (X[2 * i] - p[0]) ** 2 + (X[2 * i + 1] - p[1]) ** 2
      if (dd < bd) {
        bd = dd
        best = i
      }
    }
    pin.toggle(best)
  }
  return (
    <>
      <Figure
        title="An anchor: a box where the prediction holds"
        purpose="An anchor is a rule true at x under which the model keeps x's prediction with precision at least τ, whatever the other features do; among such rules it is the one covering most of the data, found by beam search with a bandit (KL-LUCB) that spends model calls on the candidates whose precision is uncertain."
        state={state}
        defaultSize="L"
        controls={
          <ControlRow label="2 · fit">
            <div className="flex flex-wrap items-center gap-3">
              <Button
                size="sm"
                variant={!trained || stale ? 'default' : 'outline'}
                aria-label="Train"
                onClick={() => setTrained({ ...current })}
              >
                {trained ? 'Refit' : 'Train'}
              </Button>
              <StatusText tone={!trained || stale ? 'attention' : 'muted'}>
                {!trained
                  ? 'Not fitted yet: press Train to fit the forest.'
                  : stale
                    ? 'Settings changed since the forest was fitted: press Refit.'
                    : `random forest of ${trained.trees} trees, depth ≤ ${trained.depth}`}
              </StatusText>
            </div>
          </ControlRow>
        }
        readouts={{
          [`row ${focus}${pin.pinned === null ? ' (click a point to pin another)' : ''}`]: (
            <>
              <Readout label="forest's class" value={a ? String(a.label) : '—'} />
              <Readout label="anchor" value={a ? ruleText(a.rule) : '—'} />
              <Readout label="precision" value={a ? fmt(a.precision) : '—'} />
              <Readout label="coverage" value={a ? fmt(a.coverage) : '—'} />
              <Readout label="meets τ" value={a ? (a.valid ? 'yes' : 'no (best precision shown)') : '—'} />
              <Readout label="model calls" value={a ? String(a.evaluations) : '—'} />
            </>
          ),
          'beam per size: rule (precision, coverage)': (
            <>
              {(a?.beams ?? []).map((beam, k) => (
                <Readout
                  key={k}
                  label={`size ${k + 1}`}
                  value={beam
                    .map((c) => `${ruleText(c.rule)} (${fmt(c.precision, 2)}, ${fmt(c.coverage, 2)})`)
                    .join('; ')}
                />
              ))}
            </>
          ),
        }}
        caption={`Data: aifn moons (seeded), ${n} points. Press Train to fit a random forest; the background is its P(y = 1) with the 0.5 boundary in ink. Click a point to explain it (again or Escape to unpin). Each feature is cut at its quantiles into the chosen number of bins; a predicate says "x's bin of this feature", and an anchor is a set of them, drawn as the ink box. Its precision is estimated by drawing a data row and replacing every feature the rule constrains, when it breaks the rule, by a value from a row in x's bin; its coverage is the fraction of the data inside the box. Lower τ and the box grows; more bins make the predicates narrower and the anchor more precise but smaller. Near the boundary no anchor reaches τ and the one with the best lower bound is shown.`}
      >
        <Plot x={ax} y={ay} onPlotClick={select}>
          {field && <Raster x={GX} y={GY} z={field} range={[0, 1]} valueLabel="P(y = 1)" fillOpacity={0.35} boundary />}
          <Points
            name="data"
            x={Array.from({ length: n }, (_, i) => X[2 * i])}
            y={Array.from({ length: n }, (_, i) => X[2 * i + 1])}
            group={y}
            groupNames={['y = 0', 'y = 1']}
            thin
          />
          {a && <Segments segments={boxOf(a.rule)} emphasis width={2.5} />}
          <Points name="explained x" x={[query[0]]} y={[query[1]]} emphasis size={13} />
        </Plot>
      </Figure>
      <SurrogateFigure forest={forest} data={data} fresh={fresh} proba={proba} field={field} />
    </>
  )
}

// ── Global surrogates ────────────────────────────────────────────────────────────────────────────────────────────────

function SurrogateFigure({
  forest,
  data,
  fresh,
  proba,
  field,
}: {
  forest: ShapTree[] | null
  data: { x: Tensor }
  fresh: { x: Tensor; y?: Tensor }
  proba: (Z: Tensor) => Float64Array
  field: number[][] | null
}) {
  const state = useFigureState({
    show: row('1 · surrogate', {
      depth: slider(1, 10, 3, { step: 1, label: 'tree depth shown' }),
      rules: int(6, { ge: 1, le: 20, suggestions: [3, 6, 10], label: 'rules in the list' }),
    }),
  })
  const { depth, rules: maxRules } = state.show
  const study = useComputed(
    () => {
      if (!forest) return null
      const box = Float64Array.from(proba(data.x), (p) => (p > 0.5 ? 1 : 0))
      const target = Float64Array.from(proba(fresh.x), (p) => (p > 0.5 ? 1 : 0))
      const truth = Float64Array.from(toFlat(fresh.y!))
      const labelled = dataset(data.x, fromData(Int32Array.from(box), [box.length]))
      const trees = DEPTHS.map(
        (dd) => decisionTree({ maxDepth: dd }).fit(labelled as never).tree as unknown as ShapTree,
      )
      const decide = (t: ShapTree, Z: Tensor) =>
        Float64Array.from(treeEnsembleOutput([t], Z, { output: 1 }), (p) => (p > 0.5 ? 1 : 0))
      const fid = trees.map((t) => fidelity(target, decide(t, fresh.x)))
      const acc = trees.map((t) => fidelity(truth, decide(t, fresh.x)))
      const list = ruleList(data.x, box, { maxRules, maxConditions: 3, bins: 8 })
      const listFidelity = fidelity(target, applyRuleList(list, fresh.x).labels)
      return {
        trees,
        fid,
        acc,
        boxAccuracy: fidelity(truth, target),
        list,
        listFidelity,
        listAccuracy: fidelity(truth, applyRuleList(list, fresh.x).labels),
      }
    },
    [forest, data, fresh, maxRules],
    { mode: 'release' },
  )
  const s = study.value
  const surrogateField = useMemo(
    () => (s ? asGrid(treeEnsembleOutput([s.trees[depth - 1]], GRID, { output: 1 })) : null),
    [s, depth],
  )
  const ax = useAxis({ label: 'x₁', range: XR, nice: false })
  const ay = useAxis({ label: 'x₂', range: YR, nice: false, equal: ax })
  const dAxis = useAxis({ label: 'surrogate tree depth', range: [0.5, 10.5], integer: true })
  const fAxis = useAxis({ label: 'agreement on fresh data', range: [0.5, 1] })
  return (
    <Figure
      title="Surrogate trees and a decision list: fidelity against depth"
      purpose="A global surrogate is an interpretable model fitted to the black box's predictions; its fidelity, the share of fresh points where it agrees with the black box, says how far its rules can be trusted as a description of the model, and it rises with the surrogate's size."
      state={state}
      defaultSize="XL"
      readouts={{
        fidelity: (
          <>
            <Readout label={`tree of depth ${depth}`} value={s ? fmt(s.fid[depth - 1]) : '—'} />
            <Readout
              label={`decision list (${s?.list.rules.length ?? 0} rules)`}
              value={s ? fmt(s.listFidelity) : '—'}
            />
            <Readout label="forest's accuracy" value={s ? fmt(s.boxAccuracy) : '—'} />
          </>
        ),
        'decision list (first rule that holds)': (
          <>
            {(s?.list.rules ?? []).map((r, k) => (
              <Readout
                key={k}
                label={`${k + 1}`}
                value={`if ${ruleText(r.rule)} then ${r.label} (${r.support} rows, ${fmt(r.precision, 2)})`}
              />
            ))}
            {s && <Readout label="else" value={String(s.list.defaultLabel)} />}
          </>
        ),
      }}
      caption="Uses the forest fitted above (press Train there first). The surrogates are fitted to the forest's labels on the training points, never to the true labels; fidelity and accuracy are measured on 2000 fresh moons. Left: the chosen depth's surrogate tree, P(class 1) per leaf, with the forest's 0.5 boundary drawn over it in ink; its axis-aligned cells approximate the forest's curved boundary. Right: fidelity and accuracy against the true labels of trees of depth 1–10, and the decision list's fidelity (dotted line); the list is learned by sequential covering, each rule the purest conjunction of up to three thresholds on the rows not yet covered."
    >
      <Plots cols={2} widths={[55, 45]}>
        <Plot x={ax} y={ay}>
          {surrogateField && (
            <Raster
              x={GX}
              y={GY}
              z={surrogateField}
              range={[0, 1]}
              valueLabel="surrogate P(class 1)"
              fillOpacity={0.6}
              colorBar={false}
            />
          )}
          {field && <Contours x={GX} y={GY} z={field} levels={[0.5]} labels={false} />}
        </Plot>
        <Plot x={dAxis} y={fAxis}>
          {s && <Curve name="fidelity to the forest" x={DEPTHS} y={s.fid} slot={2} showPoints stale={study.stale} />}
          {s && <Curve name="accuracy on true labels" x={DEPTHS} y={s.acc} slot={3} dashed stale={study.stale} />}
          {s && <Annotation y={s.listFidelity} dashed text="decision list" />}
          <Annotation x={depth} dashed />
        </Plot>
      </Plots>
    </Figure>
  )
}
