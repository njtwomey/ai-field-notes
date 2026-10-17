import { useMemo } from 'react'
import {
  Bars,
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  toggle,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { permutation, stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { CLOUD_SHAPES } from 'aifn-methods/data/synthetic'
import { FixedHeight, useSlicedRun } from '../_shared/figure-helpers'
import { cloudModel, exampleCloud, pointNetRun, TEST, TRAIN, WIDTH, type Checkpoint, type Setup } from './pointnet-run'

const PANEL = 260
/** Points in the inspected cloud: twice the training clouds' 32, to show the network runs on any number of points. */
const SHOWN = 64
const COUNTS = Array.from({ length: SHOWN + 1 }, (_, k) => k)
type Order = 'others' | 'random' | 'critical'

/** The order in which points are removed: non-critical first, a random order, or critical first. */
function removalOrder(order: Order, critical: readonly number[], seed: number): number[] {
  const random = Array.from(toFlat(permutation(stream(`pointnet/removal/${seed}`), SHOWN)))
  if (order === 'random') return random
  const isCritical = new Set(critical)
  const crit = random.filter((i) => isCritical.has(i))
  const rest = random.filter((i) => !isCritical.has(i))
  return order === 'others' ? [...rest, ...crit] : [...crit, ...rest]
}

/** A PointNet trained on 2-D outlines; its critical points, and what happens to its output as points are removed. */
export function PointNetCriticalPoints() {
  const state = useFigureState({
    rotate: toggle(false, { label: 'shapes at random angles' }),
    steps: int(100, { ge: 20, le: 2000, suggestions: [50, 100, 400], label: 'Adam steps' }),
    rate: float(0.01, { gt: 0, le: 0.2, suggestions: [0.003, 0.01, 0.03], label: 'learning rate' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
    example: int(0, { ge: 0, le: 99, label: 'test cloud (shape cycles)' }),
    order: choice(
      [
        { value: 'others', label: 'non-critical points first' },
        { value: 'random', label: 'random order' },
        { value: 'critical', label: 'critical points first' },
      ] as const,
      'others',
      { label: 'removal order' },
    ),
    removed: int(0, { ge: 0, le: SHOWN - 1, label: 'points removed' }),
  })
  const { rotate, steps, rate, seed, example, order, removed } = state
  const setup = useMemo<Setup>(() => ({ rotate, steps, rate, seed }), [rotate, steps, rate, seed])
  const run = useSlicedRun(setup, pointNetRun)
  const last: Checkpoint | undefined = run.checkpoints.at(-1)
  const { cloud, label } = useMemo(() => exampleCloud(rotate, example, SHOWN), [rotate, example])

  const view = useMemo(() => {
    if (!last) return null
    const model = cloudModel(last.params, cloud)
    const sequence = removalOrder(order, model.critical, example)
    const keptAfter = (k: number) => {
      const gone = new Set(sequence.slice(0, k))
      return cloud.map((_, i) => i).filter((i) => !gone.has(i))
    }
    const gFull = model.pooled(cloud.map((_, i) => i))
    const curve = COUNTS.slice(0, SHOWN).map((k) => model.probabilities(model.pooled(keptAfter(k)))[label])
    const kept = keptAfter(removed)
    const gKept = model.pooled(kept)
    return {
      critical: new Set(model.critical),
      kept: new Set(kept),
      full: model.probabilities(gFull),
      now: model.probabilities(gKept),
      change: Math.max(...gFull.map((v, c) => Math.abs(v - gKept[c]))),
      curve,
      criticalGone: model.critical.filter((i) => !kept.includes(i)).length,
    }
  }, [last, cloud, label, order, example, removed])

  const pick = (keep: (i: number) => boolean) => cloud.map((_, i) => i).filter(keep)
  const keptOrdinary = view ? pick((i) => view.kept.has(i) && !view.critical.has(i)) : []
  const keptCritical = view ? pick((i) => view.kept.has(i) && view.critical.has(i)) : []
  const gone = view ? pick((i) => !view.kept.has(i)) : []
  const layer = (idx: number[]) => ({ x: idx.map((i) => cloud[i][0]), y: idx.map((i) => cloud[i][1]) })

  const cx = useAxis({ label: '', range: [-1.45, 1.45], nice: false, zoom: false })
  const cy = useAxis({ label: '', range: [-1.45, 1.45], nice: false, zoom: false, equal: cx })
  const kx = useAxis({ label: 'points removed', range: [0, SHOWN], nice: false, integer: true })
  const py = useAxis({ label: `P(${CLOUD_SHAPES[label]})`, range: [0, 1.02], nice: false })
  const bx = useAxis({ label: '', categories: [...CLOUD_SHAPES] })
  const by = useAxis({ label: 'probability', range: [0, 1.02], nice: false })
  const predicted = view ? view.now.indexOf(Math.max(...view.now)) : -1

  return (
    <Figure
      title="The critical points decide a max-pooled output"
      state={state}
      defaultSize="L"
      hoverReadout={false}
      caption={`A PointNet with a shared per-point network of width ${WIDTH}, a max pool and a classifier is trained by full-batch Adam on ${TRAIN} noisy outlines of 32 points (circles, squares and triangles) and tested on ${TEST} more. It is then applied to a fresh cloud of ${SHOWN} points. Left: the cloud. Large black points are the critical points, the points that win the maximum in at least one of the ${WIDTH} pooled features; there are at most ${WIDTH} of them. Small grey marks are removed points. Middle: the probability of the true shape as points are removed in the chosen order; drag the vertical handle to set how many are removed. Right: class probabilities of the reduced cloud (bars) and of the full cloud (dots). Removing non-critical points leaves the pooled feature, and so the output, exactly unchanged until the first critical point goes. Removing critical points first changes the output at once. Turn on random angles to train on rotated shapes: with ${TRAIN} training clouds the network does not learn rotation invariance and test accuracy falls.`}
      readouts={{
        training: (
          <>
            <Readout label="train accuracy" value={last ? `${Math.round(100 * last.train)}%` : '…'} />
            <Readout label="test accuracy" value={last ? `${Math.round(100 * last.test)}%` : '…'} />
            <Readout label="step" value={last ? `${last.step}${run.done ? '' : ', training…'}` : '…'} />
          </>
        ),
        'this cloud': (
          <>
            <Readout label="critical points" value={view ? `${view.critical.size} of ${SHOWN}` : '…'} />
            <Readout label="critical points removed" value={view ? String(view.criticalGone) : '…'} />
            <Readout label="max change in pooled feature" value={view ? formatNumber(view.change) : '…'} />
            <Readout
              label="prediction"
              value={view ? `${CLOUD_SHAPES[predicted]} (${formatNumber(view.now[predicted])})` : '…'}
            />
          </>
        ),
      }}
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <FixedHeight height={PANEL}>
          <Plot x={cx} y={cy} bare title={`${CLOUD_SHAPES[label]}, ${SHOWN - removed} points left`}>
            {gone.length > 0 && <Points id="gone" name="removed" {...layer(gone)} muted shape={3} size={6} />}
            {keptOrdinary.length > 0 && <Points id="kept" name="point" {...layer(keptOrdinary)} slot={0} size={7} />}
            {keptCritical.length > 0 && (
              <Points id="critical" name="critical point" {...layer(keptCritical)} emphasis size={11} />
            )}
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={kx} y={py} title="true-class probability">
            {view && <Curve name={`P(${CLOUD_SHAPES[label]})`} x={COUNTS.slice(0, SHOWN)} y={view.curve} slot={0} />}
            <Handle {...state.handle('removed', { label: `${removed} removed` })} />
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={bx} y={by} title="class probabilities">
            {view && <Bars name="reduced cloud" x={[0, 1, 2]} y={view.now} slot={0} />}
            {view && <Points name="full cloud" x={[0, 1, 2]} y={view.full} emphasis size={8} />}
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
