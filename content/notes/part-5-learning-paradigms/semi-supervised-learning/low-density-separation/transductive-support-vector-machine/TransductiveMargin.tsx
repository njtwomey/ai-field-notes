import { useMemo, useState } from 'react'
import {
  Contours,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Player,
  Plot,
  Points,
  Raster,
  Readout,
  useAxis,
  useComputed,
  useFigureState,
  variants,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import {
  symmetricHinge,
  symmetricSigmoid,
  transductiveSvmDecision,
  transductiveSvmFit,
  transductiveSvmStages,
} from 'aifn-methods/learning/semi-supervised'
import {
  centredMoons,
  FixedHeight,
  gridField,
  labelledByAnchors,
  predictionGrid,
  randomFeatureMap,
  twoClusters,
} from '../_shared/figure-helpers'

const PANEL = 340
const NX = 41
const NY = 31
const DOUBLINGS = 10
const CLIP = 1.5
const MOON_FEATURES = randomFeatureMap(40, 0.5, 1)
const LINEAR = (rows: readonly (readonly number[])[]) => rows.map((r) => [r[0], r[1]])
const T = toFlat(linspace(-2.5, 2.5, 201))
const HINGE = T.map((t) => symmetricHinge(t))
const SIGMOID = T.map((t) => symmetricSigmoid(t))

type DataKey = 'clusters' | 'moons'
type Anchors = [number, number][][]

const SETUP: Record<DataKey, { x: [number, number]; y: [number, number]; anchors: Anchors; features: typeof LINEAR }> =
  {
    clusters: {
      x: [-2.7, 2.7],
      y: [-1.9, 1.9],
      anchors: [
        [
          [-1.0, 0.9],
          [-0.5, 0.8],
        ],
        [
          [1.0, -0.9],
          [0.5, -0.8],
        ],
      ],
      features: LINEAR,
    },
    moons: {
      x: [-1.8, 1.8],
      y: [-1.3, 1.3],
      anchors: [
        [
          [0.5, -0.25],
          [-0.5, 0.75],
        ],
        [
          [-0.5, 0.25],
          [0.5, -0.75],
        ],
      ],
      features: MOON_FEATURES,
    },
  }

/**
 * The ∇TSVM continuation on two data sets with two labelled points per class: the decision function and margins at
 * each stage as the unlabelled cost C* doubles, and where the unlabelled outputs sit on the symmetric loss.
 */
export function TransductiveMargin() {
  const state = useFigureState({
    data: variants(
      {
        clusters: {
          label: 'two clusters (linear SVM)',
          params: {
            spread: float(0.45, { gt: 0, le: 1.5, label: 'cluster sd', suggestions: [0.3, 0.45, 0.7, 1] }),
          },
        },
        moons: {
          label: 'two moons (random-feature SVM)',
          params: {
            spread: float(0.1, { ge: 0, le: 0.5, label: 'noise sd', suggestions: [0.05, 0.1, 0.2, 0.3] }),
          },
        },
      },
      { choiceLabel: 'data set' },
    ),
    C: float(10, { gt: 0, label: 'labelled cost C', suggestions: [1, 10, 100] }),
    Cstar: float(10, { ge: 0, label: 'final unlabelled cost C*', suggestions: [0.1, 1, 10, 100] }),
    seed: int(1, { ge: 1, le: 999, label: 'data seed' }),
  })
  const key = state.data.key as DataKey
  const { spread } = state.data.values
  const setup = SETUP[key]
  const { C, Cstar, seed } = state

  const [anchors, setAnchors] = useState<Record<DataKey, Anchors>>({
    clusters: SETUP.clusters.anchors,
    moons: SETUP.moons.anchors,
  })

  const data = useMemo(
    () => (key === 'moons' ? centredMoons(seed, spread) : twoClusters(seed, spread)),
    [key, spread, seed],
  )
  const features = useMemo(() => setup.features(data.rows), [setup, data])
  const grid = useMemo(() => {
    const g = predictionGrid(setup.x, setup.y, NX, NY)
    return { ...g, features: setup.features(g.rows) }
  }, [setup])
  const split = useMemo(() => labelledByAnchors(data, anchors[key]), [data, anchors, key])

  const run = useComputed(
    () => {
      const problem = {
        labelled: split.labelled.map((i) => features[i]),
        labels: split.classes.map((c) => (c === 1 ? 1 : -1)),
        unlabelled: split.unlabelled.map((i) => features[i]),
      }
      const supervised = transductiveSvmFit(problem, { C, Cstar: 0, balance: null })
      const stages = transductiveSvmStages(problem, { C, Cstar, doublings: DOUBLINGS })
      const accuracy = stages.map((s) => {
        let right = 0
        split.unlabelled.forEach((i, j) => {
          if ((s.unlabelledOutputs[j] > 0 ? 1 : 0) === data.y[i]) right++
        })
        return right / split.unlabelled.length
      })
      const supervisedF = transductiveSvmDecision(
        split.unlabelled.map((i) => features[i]),
        supervised,
      )
      let supervisedRight = 0
      split.unlabelled.forEach((i, j) => {
        if ((supervisedF[j] > 0 ? 1 : 0) === data.y[i]) supervisedRight++
      })
      return {
        supervised,
        supervisedAccuracy: supervisedRight / split.unlabelled.length,
        supervisedField: gridField(transductiveSvmDecision(grid.features, supervised), NX, NY),
        stages,
        accuracy,
        inMargin: stages.map((s) => Array.from(s.unlabelledOutputs).filter((f) => Math.abs(f) < 1).length),
      }
    },
    [features, split, data, grid, C, Cstar],
    { mode: 'release' },
  )
  const result = run.value

  const settings = `${key}/${spread}/${seed}/${C}/${Cstar}/${split.labelled.join(',')}`
  const [pos, setPos] = useState({ settings, at: 0 })
  const at = pos.settings === settings ? pos.at : 0
  const stage = result?.stages[Math.min(at, DOUBLINGS)]

  const field = useMemo(
    () => (stage ? gridField(transductiveSvmDecision(grid.features, stage), NX, NY) : null),
    [stage, grid],
  )
  const shaded = useMemo(() => field?.map((row) => row.map((v) => Math.max(-CLIP, Math.min(CLIP, v)))) ?? null, [field])

  const xa = useAxis({ label: 'x₁', range: setup.x })
  const ya = useAxis({ label: 'x₂', range: setup.y, equal: xa })
  const ta = useAxis({ label: 'output f(x) of an unlabelled point', range: [-2.5, 2.5] })
  const la = useAxis({ label: 'loss', range: [0, 1.05] })

  const byClass = [0, 1].map((c) => split.unlabelled.map((i, j) => ({ i, j })).filter(({ i }) => data.y[i] === c))

  return (
    <Figure
      title="The transductive SVM pushes the margin off the unlabelled points"
      state={state}
      defaultSize="L"
      caption="Left: two hundred points, two per class labelled (large markers; drag any of them, and it snaps to the nearest point of its class). The shading is the decision value f(x), clipped to ±1.5, with the boundary f = 0 in ink and the margins f = ±1 dashed; the grey dashed line is the boundary of the supervised SVM (squared hinge, free bias) on the four labels alone. The player steps through the continuation of ∇TSVM: stage k minimises the objective with C* multiplied by 2^(k − 10), starting from stage k − 1. The balancing constraint fixes the mean output on the unlabelled points to the mean labelled label, 0 here. Right: the symmetric hinge max(0, 1 − |f|) (grey) and the smooth loss exp(−3f²) that ∇TSVM minimises (ink), with every unlabelled point placed at its output f. As C* grows the points leave the bump: the margin empties and the boundary turns into the gap between the clusters."
      controls={
        <Player
          value={at}
          onChange={(v) => setPos({ settings, at: v })}
          count={DOUBLINGS + 1}
          label="stage"
          format={(k) => `stage ${k}: C* = ${formatNumber(Cstar * 2 ** (k - DOUBLINGS))}`}
        />
      }
      readouts={
        <>
          <Readout label="C* at this stage" value={stage ? formatNumber(stage.Cstar) : '…'} />
          <Readout
            label="unlabelled accuracy (supervised SVM)"
            value={result ? `${formatNumber(result.accuracy[at])} (${formatNumber(result.supervisedAccuracy)})` : '…'}
          />
          <Readout
            label="unlabelled points with |f| < 1"
            value={result ? `${result.inMargin[at]} of ${split.unlabelled.length}` : '…'}
          />
          <Readout label="objective" value={stage ? formatNumber(stage.objective) : '…'} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-[3fr_2fr]">
        <FixedHeight height={PANEL}>
          <Plot x={xa} y={ya}>
            {shaded && field && (
              <>
                <Raster
                  x={grid.gx}
                  y={grid.gy}
                  z={shaded}
                  scale="diverging"
                  range={[-CLIP, CLIP]}
                  valueLabel="f(x)"
                  boundary
                  fillOpacity={0.5}
                  stale={run.stale}
                />
                <Contours
                  name="margin f = ±1"
                  x={grid.gx}
                  y={grid.gy}
                  z={field}
                  levels={[-1, 1]}
                  labels={false}
                  dashed
                />
              </>
            )}
            {result && (
              <Contours
                name="supervised SVM boundary"
                x={grid.gx}
                y={grid.gy}
                z={result.supervisedField}
                levels={[0]}
                labels={false}
                dashed
                muted
              />
            )}
            {byClass.map(
              (rows, c) =>
                rows.length > 0 && (
                  <Points
                    key={c}
                    name={`class ${c}, unlabelled`}
                    x={rows.map(({ i }) => data.rows[i][0])}
                    y={rows.map(({ i }) => data.rows[i][1])}
                    slot={c}
                    thin
                  />
                ),
            )}
            {split.labelled.map((i, j) => (
              <Points
                key={`l${j}`}
                name={`class ${split.classes[j]}, labelled`}
                x={[data.rows[i][0]]}
                y={[data.rows[i][1]]}
                slot={split.classes[j]}
                emphasis
                size={12}
              />
            ))}
            {anchors[key].flatMap((list, c) =>
              list.map((_, a) => {
                const i = split.byAnchor[c][a]
                if (i < 0) return null
                return (
                  <Handle
                    key={`h${c}-${a}`}
                    kind="point"
                    at={[data.rows[i][0], data.rows[i][1]]}
                    slot={c}
                    label={`label ${c}`}
                    onDrag={(p) =>
                      setAnchors((all) => {
                        const next = all[key].map((l) => l.map((q) => [q[0], q[1]] as [number, number]))
                        next[c][a] = [p[0], p[1]]
                        return { ...all, [key]: next }
                      })
                    }
                  />
                )
              }),
            )}
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={ta} y={la}>
            <Curve name="symmetric hinge" x={T} y={HINGE} muted />
            <Curve name="exp(−3f²)" x={T} y={SIGMOID} emphasis />
            {stage &&
              byClass.map(
                (rows, c) =>
                  rows.length > 0 && (
                    <Points
                      key={c}
                      name={`class ${c}, unlabelled`}
                      x={rows.map(({ j }) => stage.unlabelledOutputs[j])}
                      y={rows.map(({ j }) => symmetricSigmoid(stage.unlabelledOutputs[j]))}
                      slot={c}
                      thin
                    />
                  ),
              )}
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
