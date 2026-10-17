import { useMemo, useState } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Raster,
  Readout,
  Segments,
  slider,
  useAxis,
  useComputed,
  useFigureState,
  variants,
} from 'aifn-render'
import { entropyMinimisationPath, entropyMinimisationProbabilities } from 'aifn-methods/learning/semi-supervised'
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
const L2 = 0.01
const LAMBDA_MAX = 3
const LAMBDA_STEP = 0.1
const LAMBDAS = Array.from({ length: Math.round(LAMBDA_MAX / LAMBDA_STEP) + 1 }, (_, i) => i * LAMBDA_STEP)
// Moons: logistic regression on 40 random Fourier features of lengthscale 0.5 (a smooth nonlinear classifier).
const MOON_FEATURES = randomFeatureMap(40, 0.5, 1)
const LINEAR = (rows: readonly (readonly number[])[]) => rows.map((r) => [r[0], r[1]])

type DataKey = 'moons' | 'clusters'
type Anchors = [number, number][][]

const SETUP: Record<DataKey, { x: [number, number]; y: [number, number]; anchors: Anchors; features: typeof LINEAR }> =
  {
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
  }

const LN2 = Math.log(2)

/**
 * Entropy minimisation on two data sets with two labelled points per class: the boundary at the chosen weight λ, and
 * the mean entropy of the unlabelled predictions and the unlabelled accuracy along the path of λ.
 */
export function EntropyBoundary() {
  const state = useFigureState({
    data: variants(
      {
        moons: {
          label: 'two moons (random-feature logistic regression)',
          params: {
            noise: float(0.1, { ge: 0, le: 0.5, label: 'noise sd', suggestions: [0.05, 0.1, 0.2, 0.3] }),
          },
        },
        clusters: {
          label: 'two clusters (linear logistic regression)',
          params: {
            noise: float(0.55, { gt: 0, le: 1.5, label: 'cluster sd', suggestions: [0.3, 0.55, 0.8, 1.1] }),
          },
        },
      },
      { choiceLabel: 'data set' },
    ),
    lambda: slider(0, LAMBDA_MAX, 1, { step: LAMBDA_STEP, label: 'entropy weight λ' }),
    seed: int(3, { ge: 1, le: 999, label: 'data seed' }),
  })
  const key = state.data.key as DataKey
  const { noise } = state.data.values
  const setup = SETUP[key]
  const lambdaIndex = Math.min(LAMBDAS.length - 1, Math.max(0, Math.round(state.lambda / LAMBDA_STEP)))

  const [anchors, setAnchors] = useState<Record<DataKey, Anchors>>({
    moons: SETUP.moons.anchors,
    clusters: SETUP.clusters.anchors,
  })

  const data = useMemo(
    () => (key === 'moons' ? centredMoons(state.seed, noise) : twoClusters(state.seed, noise)),
    [key, noise, state.seed],
  )
  const features = useMemo(() => setup.features(data.rows), [setup, data])
  const grid = useMemo(() => {
    const g = predictionGrid(setup.x, setup.y, NX, NY)
    return { ...g, features: setup.features(g.rows) }
  }, [setup])
  const split = useMemo(() => labelledByAnchors(data, anchors[key]), [data, anchors, key])

  const path = useComputed(
    () => {
      const fits = entropyMinimisationPath(
        {
          labelled: split.labelled.map((i) => features[i]),
          labels: split.classes,
          unlabelled: split.unlabelled.map((i) => features[i]),
          classes: 2,
        },
        { lambdas: LAMBDAS, l2: L2, maxSteps: 400 },
      )
      const unlabelledFeatures = split.unlabelled.map((i) => features[i])
      const accuracy = fits.map((fit) => {
        const p = entropyMinimisationProbabilities(unlabelledFeatures, fit)
        let right = 0
        split.unlabelled.forEach((i, j) => {
          if ((p[2 * j + 1] > 0.5 ? 1 : 0) === data.y[i]) right++
        })
        return right / split.unlabelled.length
      })
      return { fits, accuracy, entropyBits: fits.map((f) => f.entropy / LN2) }
    },
    [features, split, data],
    { mode: 'release' },
  )
  const run = path.value
  const fit = run?.fits[lambdaIndex]
  const p1 = useMemo(() => {
    if (!fit) return null
    const p = entropyMinimisationProbabilities(grid.features, fit)
    return gridField(
      Float64Array.from({ length: NX * NY }, (_, i) => p[2 * i + 1]),
      NX,
      NY,
      (v) => v - 0.5,
    )
  }, [fit, grid])

  const xa = useAxis({ label: 'x₁', range: setup.x })
  const ya = useAxis({ label: 'x₂', range: setup.y, equal: xa })
  const la = useAxis({ label: 'entropy weight λ', range: [0, LAMBDA_MAX] })
  const va = useAxis({ label: 'value', range: [0, 1.02] })

  const byClass = [0, 1].map((c) => split.unlabelled.filter((i) => data.y[i] === c))
  const lambda = LAMBDAS[lambdaIndex]

  return (
    <Figure
      title="Entropy minimisation moves the boundary into the gap"
      state={state}
      defaultSize="L"
      caption="Left: two hundred points, two per class labelled (large markers); small markers show the true class of the unlabelled points, and the shading is the model's P(class 1) − ½ with the boundary in ink. Drag any labelled point: it snaps to the nearest point of its class. The slider sets the weight λ of the mean entropy of the unlabelled predictions; λ = 0 is plain L2-penalised logistic regression on the four labels. Each λ is fitted by L-BFGS from the solution at the previous λ, increasing λ in steps of 0.1. Right: along that path, the mean entropy of the unlabelled predictions in bits (falling) and the accuracy on the unlabelled points. With the default labels the boundary leaves the moons' tips and settles in the gap. Move a labelled point of each class onto the far ends of the moons and the path stalls in a boundary that cuts each moon: the entropy is low there too, because only a few points lie near it."
      readouts={
        <>
          <Readout label="λ" value={formatNumber(lambda)} />
          <Readout label="unlabelled accuracy" value={run ? formatNumber(run.accuracy[lambdaIndex]) : '…'} />
          <Readout label="mean entropy (bits)" value={run ? formatNumber(run.entropyBits[lambdaIndex]) : '…'} />
          <Readout label="labelled cross-entropy (nats)" value={fit ? formatNumber(fit.crossEntropy) : '…'} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-[3fr_2fr]">
        <FixedHeight height={PANEL}>
          <Plot x={xa} y={ya}>
            {p1 && (
              <Raster
                x={grid.gx}
                y={grid.gy}
                z={p1}
                scale="diverging"
                range={[-0.5, 0.5]}
                valueLabel="P(class 1) − ½"
                boundary
                fillOpacity={0.5}
                stale={path.stale}
              />
            )}
            {byClass.map(
              (rows, c) =>
                rows.length > 0 && (
                  <Points
                    key={c}
                    name={`class ${c}, unlabelled`}
                    x={rows.map((i) => data.rows[i][0])}
                    y={rows.map((i) => data.rows[i][1])}
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
          <Plot x={la} y={va}>
            {run && (
              <>
                <Curve name="mean entropy (bits)" x={LAMBDAS} y={run.entropyBits} slot={2} stale={path.stale} />
                <Curve name="unlabelled accuracy" x={LAMBDAS} y={run.accuracy} emphasis stale={path.stale} />
                <Segments name="chosen λ" segments={[{ from: [lambda, 0], to: [lambda, 1.02] }]} muted />
              </>
            )}
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
