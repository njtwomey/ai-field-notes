import { useMemo, useState } from 'react'
import {
  Figure,
  float,
  formatNumber,
  int,
  Player,
  Plot,
  Points,
  Raster,
  Readout,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { logisticRegression } from 'aifn-methods/learning/generalised/glm'
import { child, stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { coTrainingSteps, selfTrainingSteps } from 'aifn-methods/learning/semi-supervised'
import { semiSupervisedSplit, twoViewDataset } from 'aifn-methods/data/synthetic'
import { centredField, predictionGrid } from '../_shared/models'

const N = 200
const NG = 31
const BOX: [number, number] = [-1, 1]
const GRID = predictionGrid(BOX, BOX, NG, NG)
const BASE = logisticRegression({ l2: 1 })
const VIEW_NAMES = ['view a', 'view b'] as const

/**
 * Co-training on two views whose marginals are uniform: each view's classifier labels its most confident pool points
 * for the shared labelled set, round by round, with a knob for how dependent the views are given the class.
 */
export function CoTrainingViews() {
  const fig = useFigureState({
    dependence: float(0, { min: 0, max: 1, step: 0.05, label: 'dependence of the views' }),
    perClass: int(2, { min: 1, max: 20, label: 'labels per class', suggestions: [1, 2, 5, 10] }),
    perRound: int(2, { min: 1, max: 10, label: 'labels added per class per round', suggestions: [1, 2, 5] }),
    rounds: int(30, { min: 1, max: 45, label: 'rounds', suggestions: [10, 30, 45] }),
    seed: int(2, { min: 1, max: 999, label: 'data seed' }),
  })
  const { dependence, perClass, perRound, rounds, seed } = fig

  const data = useMemo(() => {
    const d = twoViewDataset(stream(seed), { n: N, dependence })
    const test = twoViewDataset(child(stream(seed), 'test'), { n: 1000, dependence })
    const y = toFlat(d.y)
    const split = semiSupervisedSplit(child(stream(seed), 'split'), y, perClass)
    return { d, test, y, testY: toFlat(test.y), split, a: toFlat(d.a), b: toFlat(d.b) }
  }, [seed, dependence, perClass])

  const run = useComputed(
    () => {
      const { d, test, y, testY, split } = data
      const co = trace(
        coTrainingSteps(
          {
            views: [d.a, d.b],
            labels: split.labels,
            classes: 2,
            truth: y,
            test: { views: [test.a, test.b], truth: testY },
            evaluate: [GRID.points, GRID.points],
          },
          { base: BASE, perRound: [perRound, perRound], pool: 50 },
        ),
        undefined,
        rounds,
        { keep: 'all' },
      ).steps
      // Self-training of view b alone with the same budget: it labels 2 × perRound points per class per round.
      const self = trace(
        selfTrainingSteps(
          { x: d.b, labels: split.labels, classes: 2, truth: y, evaluate: test.b, evaluateTruth: testY },
          { base: BASE, threshold: 0, perClass: 2 * perRound, keep: 'accumulate' },
        ),
        undefined,
        rounds,
        { keep: 'all' },
      ).steps.map((s) => s.evaluateAccuracy)
      return { co, self }
    },
    [data, perRound, rounds],
    { mode: 'release' },
  )

  const settings = `${seed}/${dependence}/${perClass}/${perRound}/${rounds}`
  const [pos, setPos] = useState({ settings, at: 0 })
  const { co, self } = run.value
  const at = pos.settings === settings ? Math.min(pos.at, co.length - 1) : 0
  const s = co[at]

  const xa = useAxis({ label: 'a₁ (class coordinate)', range: [-1.05, 1.05] })
  const ya = useAxis({ label: 'a₂', range: [-1.05, 1.05], equal: xa })
  const xb = useAxis({ label: 'b₁', range: [-1.05, 1.05] })
  const yb = useAxis({ label: 'b₂ (class coordinate)', range: [-1.05, 1.05], equal: xb })
  if (!s) return null

  const coords = [data.a, data.b]
  const idx = data.y.map((_, i) => i)
  const panel = (v: 0 | 1) => {
    const c = coords[v]
    const px = (rows: number[]) => rows.map((i) => c[2 * i])
    const py = (rows: number[]) => rows.map((i) => c[2 * i + 1])
    const own = v === 0 ? 2 : 3
    const other = v === 0 ? 3 : 2
    const unl = idx.filter((i) => s.source[i] === 0)
    const original = idx.filter((i) => s.source[i] === 1)
    const wrong = idx.filter((i) => s.source[i] >= 2 && s.assigned[i] !== data.y[i])
    const layers = [0, 1].flatMap((k) => [
      {
        key: `r${k}`,
        name: `class ${k}, from the other view`,
        rows: idx.filter((i) => s.source[i] === other && s.assigned[i] === k),
        thin: false,
      },
      {
        key: `o${k}`,
        name: `class ${k}, own pick`,
        rows: idx.filter((i) => s.source[i] === own && s.assigned[i] === k),
        thin: true,
      },
    ])
    return (
      <>
        <Raster
          x={GRID.gx}
          y={GRID.gy}
          z={centredField(s.evaluateProbs[v], NG, NG)}
          scale="diverging"
          range={[-0.5, 0.5]}
          valueLabel={`P(class 1 | ${VIEW_NAMES[v]}) − ½`}
          boundary
          fillOpacity={0.45}
          colorBar={v === 1}
          stale={run.stale}
        />
        {unl.length > 0 && <Points name="unlabelled" x={px(unl)} y={py(unl)} muted thin />}
        {layers.map(
          (l) =>
            l.rows.length > 0 && (
              <Points
                key={l.key}
                name={l.name}
                x={px(l.rows)}
                y={py(l.rows)}
                slot={l.key.endsWith('0') ? 0 : 1}
                thin={l.thin}
              />
            ),
        )}
        {wrong.length > 0 && <Points name="wrong label" x={px(wrong)} y={py(wrong)} tone="destructive" />}
        <Points name="original labels" x={px(original)} y={py(original)} emphasis size={9} />
      </>
    )
  }

  return (
    <Figure
      title="Co-training on two views"
      state={fig}
      caption="Each point has two views: view a (left), whose class depends on a₁, and view b (right), whose class depends on b₂. Each view's density is uniform on the square, so the unlabelled points of one view alone say nothing about its boundary. The shading is each view's logistic classifier, P(class 1) − ½, with its boundary in ink. In every round each classifier labels its most confident pool points of each class, and those labels join the shared set: large coloured points in a panel came from the other view's classifier, small ones from the panel's own; red marks a wrong label. Step through the rounds with the player. With dependence 0 the views are independent given the class, and the points one classifier receives lie anywhere in its square, including near its boundary. With dependence 1 view b is a rotated copy of view a, co-training reduces to self-training, and more of the added labels are wrong."
      controls={<Player value={at} onChange={(v) => setPos({ settings, at: v })} count={co.length} label="round" />}
      readouts={
        <>
          <Readout label="round" value={s.t} />
          <Readout label="test accuracy, view a" value={formatNumber(s.testAccuracy[0])} />
          <Readout label="test accuracy, view b" value={formatNumber(s.testAccuracy[1])} />
          <Readout label="both views (product)" value={formatNumber(s.testAccuracy[2])} />
          <Readout label="view b self-trained alone" value={formatNumber(self[Math.min(at, self.length - 1)])} />
          <Readout label="wrong labels added" value={`${s.addedWrong} of ${s.added}`} />
          <Readout label="best one view can do" value={formatNumber(1 - data.d.truth.viewBayesError)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Plot x={xa} y={ya}>
          {panel(0)}
        </Plot>
        <Plot x={xb} y={yb}>
          {panel(1)}
        </Plot>
      </div>
    </Figure>
  )
}
