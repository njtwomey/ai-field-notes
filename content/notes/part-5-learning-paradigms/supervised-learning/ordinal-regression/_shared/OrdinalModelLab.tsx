import { useMemo, useState } from 'react'
import { choice, Figure, formatNumber, Readout, slider, useFigureState } from 'aifn-render'
import { fitGp2d, GP_CAP } from './gp2d'
import { MAP_CAPTION, negativeShare } from './mapText'
import { OrdinalDataControls } from './OrdinalDataControls'
import { OrdinalMap } from './OrdinalMap'
import {
  MODEL_LABELS,
  TEST_PER_CLASS,
  dataset,
  expectedClass,
  ordinalMetrics,
  type DataSpec,
  type ModelId,
  type Point,
} from './ordinal'
import { useFit } from './useFit'
import { useGrid, useOrdinalData } from './useOrdinalData'

/** Every model the lab can fit: the shared models, and the GP, whose fit also depends on its own two settings. */
export type LabModel = ModelId | 'gp'

const LABELS: Record<LabModel, string> = { ...MODEL_LABELS, gp: 'Gaussian process' }

/** The families of the overview's comparison table, in its order. */
const FAMILIES: { value: string; label: string; models: LabModel[] }[] = [
  { value: 'baseline', label: 'baselines', models: ['multiclass', 'regression'] },
  { value: 'threshold', label: 'latent threshold', models: ['cumulative-logit', 'cumulative-probit', 'gp'] },
  { value: 'sequential', label: 'sequential and adjacent', models: ['continuation-ratio', 'adjacent-category'] },
  { value: 'loss', label: 'threshold losses', models: ['immediate-threshold', 'all-threshold', 'svor'] },
  { value: 'decomposition', label: 'decomposition', models: ['binary-decomposition', 'li-lin'] },
  { value: 'deep', label: 'neural heads', models: ['orcnn', 'coral', 'corn', 'sord'] },
  { value: 'structured', label: 'structured', models: ['storm', 'storm-poly2', 'storm-poly3', 'storm-nystrom'] },
]

const ALL_LAB_MODELS = FAMILIES.flatMap((f) => f.models)

type Props = {
  /** The model selected at first. */
  model?: LabModel
  /** Overrides of the shared data defaults, e.g. `{ shape: 'spiral' }`. */
  data?: Partial<DataSpec>
  /** The models offered. Defaults to all of them, grouped by family. */
  models?: LabModel[]
  title?: string
}

/**
 * One frame for every ordinal model in the category: the shared data controls, a model chosen by family, the same
 * views (fill, contours, class probabilities at a draggable query point) and the same held-out metrics. Only the
 * selected model is fitted.
 */
export function OrdinalModelLab({
  model: initialModel = 'cumulative-logit',
  data: initialData,
  models = ALL_LAB_MODELS,
  title = 'Ordinal models on shared data',
}: Props) {
  // One list in the overview's family order; each option names its family, which search also matches.
  const options = FAMILIES.flatMap((f) =>
    f.models
      .filter((m) => models.includes(m))
      .map((m) => ({ value: m, label: `${LABELS[m]} (${f.label})`, keywords: f.label })),
  )
  const state = useFigureState({
    model: choice<LabModel>(options, initialModel, { label: 'model' }),
    lengthscale: slider(0.2, 2.5, 0.8, { step: 0.05, label: 'GP lengthscale ℓ', when: (v) => v.model === 'gp' }),
    sigma: slider(0.05, 1, 0.3, { step: 0.05, label: 'GP noise σ', when: (v) => v.model === 'gp' }),
  })
  const { model, lengthscale, sigma } = state
  const { spec, setSpec, resolution, setResolution, fill, setFill } = useOrdinalData(initialData)
  const [query, setQuery] = useState<Point>([0, 1.5])

  // The shared models fit in slices between frames; the GP's Laplace fit on its capped subsample is quick enough to
  // run here. The cumulative logit stands in for the hook while the GP is selected: it is cached and costs nothing.
  const shared = useFit(spec, model === 'gp' ? 'cumulative-logit' : model)
  const gp = useMemo(
    () => (model === 'gp' ? fitGp2d(dataset(spec), spec, lengthscale, sigma) : null),
    [model, spec, lengthscale, sigma],
  )
  const fitted = gp ? gp.fitted : shared.fitted
  const d = gp ? dataset(spec) : shared.data
  const fitting = !gp && shared.fitting

  const grid = useGrid(d.range, resolution)
  const metrics = useMemo(() => ordinalMetrics(d.test.y, d.test.x.map(fitted.predict), d.k), [d, fitted])
  // Models whose class probabilities are differences of separately fitted curves can go negative: the share of the
  // plane where they do.
  const negative = useMemo(
    () => (['binary-decomposition', 'li-lin', 'orcnn'].includes(model) ? negativeShare(fitted, grid) : null),
    [fitted, model, grid],
  )
  const probs = fitted.probs(query)

  return (
    <Figure
      title={title}
      state={state}
      caption={`Each model is fitted to the same training points, an equal number per class, drawn along the chosen curve and cut into K classes by position along it; only the selected model is fitted. The neural models have one hidden layer of 24 units; the GP uses at most ${GP_CAP} points. ${MAP_CAPTION} Metrics are computed on ${TEST_PER_CLASS} held-out points per class from the same curve, whatever the training size.`}
      controls={
        <OrdinalDataControls
          spec={spec}
          setSpec={setSpec}
          resolution={resolution}
          setResolution={setResolution}
          fill={fill}
          setFill={setFill}
        />
      }
      readouts={
        <>
          {fitting && <Readout label="fitting" value="…" />}
          <Readout label="accuracy" value={formatNumber(metrics.accuracy)} />
          <Readout label="MAE" value={formatNumber(metrics.mae)} />
          <Readout label="macro-MAE" value={formatNumber(metrics.macroMae)} />
          <Readout label="QWK" value={formatNumber(metrics.qwk)} />
          <Readout label="predicted class at x" value={fitted.predict(query) + 1} />
          <Readout label="E[y | x]" value={formatNumber(expectedClass(probs))} />
          {fitted.invalidMass && (
            <Readout label="mass on invalid codes at x" value={formatNumber(fitted.invalidMass(query))} />
          )}
          {negative !== null && <Readout label="grid with a negative probability" value={formatNumber(negative)} />}
          {gp && <Readout label="GP log evidence" value={formatNumber(gp.logEvidence)} />}
        </>
      }
    >
      <OrdinalMap fitted={fitted} data={d} resolution={resolution} fill={fill} query={query} setQuery={setQuery} />
    </Figure>
  )
}
