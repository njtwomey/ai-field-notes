import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  float,
  Handle,
  int,
  Player,
  Plot,
  Points,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { BLOB_NAMES, BoundaryMap } from '../_shared/BoundaryMap'
import type { ConfidenceThresholdRule } from 'aifn-methods/neural/semi-supervised'
import {
  BLOBS_BOX,
  BLOBS_GRID,
  blobsProblem,
  byClass,
  classAccuracy,
  classifier,
  consistencyRun,
  field,
  percent,
  type RunSetup,
} from '../_shared/consistency-runs'
import { FixedHeight, usePlayerPosition, useSlicedRun } from '../_shared/figure-helpers'

const EVERY = 5
const PANEL = 250
const RULES = [
  { value: 'fixed', label: 'fixed threshold (FixMatch)' },
  { value: 'flexmatch', label: 'FlexMatch' },
  { value: 'freematch', label: 'FreeMatch' },
  { value: 'softmatch', label: 'SoftMatch' },
] as const
const SIZES = [
  { value: 'balanced', label: 'balanced: 80, 80, 80' },
  { value: 'imbalanced', label: 'imbalanced: 150, 60, 30' },
] as const
const CLASS_SIZES = { balanced: [80, 80, 80], imbalanced: [150, 60, 30] } as const

/**
 * FixMatch with a fixed threshold against FlexMatch, FreeMatch and SoftMatch on three Gaussian blobs with two labels
 * per class: the per-class thresholds and the pseudo-labels each class receives over training.
 */
export function ThresholdTrajectories() {
  const state = useFigureState({
    rule: choice(RULES, 'flexmatch', { label: 'threshold rule' }),
    sizes: choice(SIZES, 'balanced', { label: 'class sizes' }),
    tau: float(0.95, { ge: 0.5, le: 0.999, label: 'base threshold τ', suggestions: [0.8, 0.9, 0.95], onChart: true }),
    momentum: float(0.99, {
      ge: 0,
      le: 0.9999,
      label: 'EMA momentum (FreeMatch, SoftMatch)',
      suggestions: [0.9, 0.99, 0.999],
    }),
    steps: int(300, { ge: 10, le: 2000, label: 'training steps', suggestions: [100, 300, 600] }),
    rate: float(0.01, { gt: 0, le: 0.1, label: 'Adam step size', suggestions: [0.003, 0.01, 0.03] }),
    seed: int(2, { ge: 1, le: 999, label: 'data seed' }),
  })
  const { rule, sizes, tau, momentum, steps, rate, seed } = state
  const problem = useMemo(
    () => blobsProblem(CLASS_SIZES[sizes as keyof typeof CLASS_SIZES] as unknown as [number, number, number], seed),
    [sizes, seed],
  )
  const setup = useMemo<RunSetup>(
    () => ({
      problem,
      method: { kind: 'fixmatch', threshold: tau, thresholds: rule as ConfidenceThresholdRule, momentum },
      steps,
      rate,
      weight: 1,
      weak: 0.1,
      strong: 0.5,
      seed,
      every: EVERY,
    }),
    [problem, tau, rule, momentum, steps, rate, seed],
  )
  const run = useSlicedRun(setup, consistencyRun)
  const [pos, setPos] = usePlayerPosition(setup)
  const count = run.checkpoints.length
  const at = Math.min(pos, Math.max(0, count - 1))
  const cp = run.checkpoints[at]
  const net = useMemo(() => classifier(3), [])

  const view = useMemo(() => {
    if (!cp) return null
    const has = cp.step > 0
    const used = Int32Array.from(cp.state.pseudo, (c, i) => (has && cp.state.mask[i] > 0.5 ? c : -1))
    const unused = Int32Array.from(cp.state.pseudo, (_, i) => (has && cp.state.mask[i] <= 0.5 ? 0 : -1))
    return {
      has,
      used,
      unused,
      values: field(net, cp.state.params, BLOBS_GRID, 3),
      accuracy: classAccuracy(net, cp.state.params, problem),
    }
  }, [cp, net, problem])

  const history = useMemo(() => {
    const kept = run.checkpoints.filter((c) => c.step > 0)
    const step = kept.map((c) => c.step)
    const thresholds = [0, 1, 2].map((k) => kept.map((c) => c.state.thresholds[k] ?? NaN))
    const counts = [0, 1, 2].map((k) =>
      kept.map((c) => {
        let n = 0
        c.state.pseudo.forEach((p, i) => (n += p === k ? c.state.mask[i] : 0))
        return n
      }),
    )
    return { step, thresholds, counts }
  }, [run.checkpoints])
  const topCount = Math.max(10, ...history.counts.flat())
  const stepNow = cp?.step ?? 0

  const sa = useAxis({ label: 'step', range: [0, steps] })
  const ta = useAxis({ label: 'threshold', range: [0, 1] })
  const sb = useAxis({ label: 'step', range: [0, steps] })
  const ca = useAxis({ label: 'pseudo-labels used (weighted)', range: [0, 1.05 * topCount] })
  const acc = view?.accuracy

  return (
    <Figure
      title="Fixed and adaptive confidence thresholds"
      state={state}
      defaultSize="XL"
      caption={`Three Gaussian blobs (sd 0.8, centres on a triangle of side 3) with 2 labelled points per class (large marks); two hidden layers of 32 tanh units, trained full batch with Adam. Each step is FixMatch: Gaussian jitter of sd 0.1 for the weak view and 0.5 for the strong view, cross-entropy on the strong view against the weak view's arg max, weighted by the threshold rule. Left: the predicted class at the player's checkpoint, with the unlabelled points used in the last step coloured by their pseudo-label and the rest grey. Middle: each class's threshold over training (SoftMatch's single curve is the mean confidence μ below which its weight falls off); drag the dashed line to set the base threshold τ. Right: the pseudo-labels each class receives, counted with their weights. On the default balanced data, the fixed threshold leaves class A almost unlabelled and the network ends up worse than with the labels alone; FlexMatch and FreeMatch lower class A's threshold and recover it. With the imbalanced sizes FlexMatch reads the small classes as poorly learnt, lowers their thresholds, and they absorb points of the large class.`}
      controls={
        <Player
          value={at}
          onChange={setPos}
          count={Math.max(1, count)}
          label="checkpoint"
          format={(a) => `step ${run.checkpoints[a]?.step ?? 0}`}
        />
      }
      readouts={
        <>
          <Readout label="step" value={cp ? `${cp.step}${run.done ? '' : ', training…'}` : '…'} />
          <Readout label="accuracy on unlabelled points" value={percent(cp?.accuracy)} />
          {BLOB_NAMES.map((n, k) => (
            <Readout key={n} label={`${n} accuracy`} value={percent(acc?.[k])} />
          ))}
          <Readout label="pseudo-labels correct" value={percent(cp?.state.pseudoAccuracy)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <FixedHeight height={PANEL}>
          <BoundaryMap
            problem={problem}
            grid={BLOBS_GRID}
            box={BLOBS_BOX}
            values={view?.values ?? null}
            hideUnlabelled={Boolean(view?.has)}
            title="pseudo-labels on the plane"
          >
            {view?.has && (
              <>
                <Points name="not used" {...byClass(problem.unlabelled, view.unused, 0)} muted thin />
                {[0, 1, 2].map((c) => (
                  <Points
                    key={c}
                    name={`pseudo-label: ${BLOB_NAMES[c]}`}
                    {...byClass(problem.unlabelled, view.used, c)}
                    slot={c}
                    thin
                  />
                ))}
              </>
            )}
          </BoundaryMap>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={sa} y={ta} title="thresholds">
            {[0, 1, 2].map((k) => (
              <Curve key={k} name={BLOB_NAMES[k]} x={history.step} y={history.thresholds[k]} slot={k} />
            ))}
            <Curve name="base threshold τ" x={[0, steps]} y={[tau, tau]} emphasis dashed silent />
            <Curve name="checkpoint" x={[stepNow, stepNow]} y={[0, 1]} muted dashed silent />
            <Handle
              kind="y"
              at={tau}
              label="τ"
              onDrag={(y) => state.set('tau', Math.round(Math.min(0.999, Math.max(0.5, y)) * 1000) / 1000)}
            />
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={sb} y={ca} title="pseudo-labels per class">
            {[0, 1, 2].map((k) => (
              <Curve key={k} name={BLOB_NAMES[k]} x={history.step} y={history.counts[k]} slot={k} />
            ))}
            <Curve name="checkpoint" x={[stepNow, stepNow]} y={[0, 1.05 * topCount]} muted dashed silent />
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
