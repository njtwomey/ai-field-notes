import { useMemo } from 'react'
import {
  datasetRegistry,
  describeRecipe,
  modifierRegistry,
  recipe,
  recipeBases,
  recipeOps,
  type ClassificationTruth,
  type RecipeInput,
} from 'aifn-applied/data'
import { flippedMask } from 'aifn-applied/data/synthetic'
import { grid2d } from 'aifn/numerics/geometry'
import { type SpaceValues } from 'aifn/foundation/space'
import { fromData, toFlat } from 'aifn/foundation/tensor'
import { Figure } from '@lab/layout'
import { int, fromSpace, row, toggle, useFigureState, variants, type AnyValues, type ParamDefs } from '@lab/state'
import { Contours, Plot, Plots, Points, Raster, Readout, useAxis } from '@lab/viz'

// Every labelled point-set generator of the registry; its knobs and the modifiers' parameters come from their spaces.
const BASES = recipeBases.filter((k) => {
  const { random, task } = datasetRegistry[k].info
  return random === true && (task === 'classification' || task === 'clustering')
})

const pct = (v: number) => `${Math.round(100 * v)}%`

/** The base generator: one case per registered generator, its knobs from the registry's space. */
const BASE = variants(
  Object.fromEntries(BASES.map((k) => [k, { label: k, params: fromSpace(datasetRegistry[k].info.knobs) }])),
  {
    shared: { seed: int(0, { ge: 0, label: 'seed' }) },
    initial: 'moons',
    label: '1 · base',
    choiceLabel: 'generator',
  },
)

const opName = (op: string) => modifierRegistry[op].info.name.toLowerCase()

/** Which modifiers apply: one revealing toggle each, in the order they are applied. */
const MODS = row(
  '2 · modifiers',
  Object.fromEntries(recipeOps.map((op) => [op, toggle(op === 'withLabelNoise', opName(op))])) as ParamDefs,
)

/** The parameters of each modifier, from the registry's space: a row shown only while that modifier applies. */
const MOD_PARAMS: ParamDefs = Object.fromEntries(
  recipeOps
    .map((op) => [op, fromSpace(modifierRegistry[op].info.params)] as const)
    .filter(([, fields]) => Object.keys(fields).length > 0)
    .map(([op, fields]) => [
      op,
      {
        ...row(`2 · ${opName(op)}`, fields),
        when: (v: AnyValues) => (v.mods as Record<string, boolean> | undefined)?.[op] === true,
      },
    ]),
)

const SCHEMA = {
  base: BASE,
  mods: MODS,
  ...MOD_PARAMS,
  show: row('3 · reveal', { truth: toggle(true, 'Bayes posterior and boundary') }),
} as const

/**
 * The recipe the figure's values describe, as JSON, so the dataset is rebuilt only when a value changes, not when an
 * object is new.
 */
function recipeKey(values: Readonly<Record<string, unknown>>): string {
  const base = values.base as { key: string; values: SpaceValues }
  const { seed, ...knobs } = base.values
  const applied = values.mods as Record<string, boolean>
  return JSON.stringify({
    base: base.key,
    seed: Number(seed),
    knobs,
    modifiers: recipeOps
      .filter((op) => applied[op] === true)
      .map((op) => ({ op, params: (values[op] ?? {}) as SpaceValues })),
  })
}

export function RecipeSpecimen() {
  const state = useFigureState(SCHEMA)
  const showTruth = (state.show as { truth: boolean }).truth
  const specKey = recipeKey(state.values as Readonly<Record<string, unknown>>)
  const spec = JSON.parse(specKey) as RecipeInput
  // oxlint-disable-next-line react/preserve-manual-memoization -- keyed by a string; the compiler cannot see that
  const data = useMemo(() => recipe(JSON.parse(specKey) as RecipeInput), [specKey])
  const [rows, d] = data.x.shape
  const classes = data.meta.labelNames?.length ?? 1 + Math.max(...toFlat(data.y!))
  // The field shows P(y = 1 | x), which is the whole posterior only with two classes.
  const truth = classes === 2 ? (data.meta.truth as ClassificationTruth | undefined) : undefined

  // Points at their complete positions (before missing values), split into the groups the chart marks.
  const view = useMemo(() => {
    const full = toFlat(data.meta.complete ?? data.x)
    const mask = data.meta.missing ? toFlat(data.meta.missing) : undefined
    const y = toFlat(data.y!)
    const flipped = flippedMask(data)
    const px = Array.from({ length: rows }, (_, i) => full[i * d])
    const py = Array.from({ length: rows }, (_, i) => full[i * d + 1])
    const gappy = (i: number) => mask !== undefined && (mask[i * d] === 1 || mask[i * d + 1] === 1)
    const pick = (keep: (i: number) => boolean) => px.map((_, i) => i).filter(keep)
    const shown = pick((i) => !gappy(i))
    const flips = pick((i) => flipped[i] === 1)
    const holes = pick(gappy)
    const pad = (lo: number, hi: number): [number, number] => {
      const m = 0.08 * (hi - lo || 1)
      return [lo - m, hi + m]
    }
    const range = (v: number[]) => pad(Math.min(...v), Math.max(...v))
    const at = (idx: number[]) => ({ x: idx.map((i) => px[i]), y: idx.map((i) => py[i]) })
    return {
      px,
      y,
      shown: { ...at(shown), group: shown.map((i) => y[i]) },
      flips: at(flips),
      holes: at(holes),
      xr: range(px),
      yr: range(py),
      full,
    }
  }, [data, rows, d])

  const field = useMemo(() => grid2d(view.xr, view.yr, 70), [view])
  const posterior = useMemo(() => {
    if (!truth || !showTruth) return undefined
    const pts = toFlat(field.points)
    const [ny, nx] = field.shape
    // The grid in the first two features, zero in the rest; P(ỹ = 1 | x) is the truth's expectation for two classes.
    const probes = new Float64Array(ny * nx * d)
    for (let p = 0; p < ny * nx; p++) {
      probes[p * d] = pts[2 * p]
      probes[p * d + 1] = pts[2 * p + 1]
    }
    const p1 = toFlat(truth.expect(fromData(probes, [ny * nx, d])))
    return Array.from({ length: ny }, (_, i) => Array.from({ length: nx }, (_, j) => p1[i * nx + j]))
  }, [truth, showTruth, field, d])

  const names = useMemo(() => data.meta.labelNames ?? ['class 0', 'class 1'], [data])
  const nuisanceMod = spec.modifiers?.find((m) => m.op === 'withNuisanceFeatures')
  const nuisance = nuisanceMod ? Number(nuisanceMod.params?.count ?? 0) : 0
  const outliers = spec.modifiers?.some((m) => m.op === 'withOutliers') === true
  const axisKey = `${spec.base}:${rows}:${spec.seed}:${outliers}`

  // Side panels: a nuisance feature against x₁, and the missingness mask.
  const pair = useMemo(
    () => (nuisance === 0 ? undefined : view.px.map((_, i) => view.full[i * d + d - nuisance])),
    [nuisance, view, d],
  )
  const mask = useMemo(() => {
    if (!data.meta.missing) return undefined
    const m = toFlat(data.meta.missing)
    return {
      z: Array.from({ length: rows }, (_, i) => m.slice(i * d, (i + 1) * d)),
      x: Array.from({ length: d }, (_, j) => j + 1),
      y: Array.from({ length: rows }, (_, i) => i),
    }
  }, [data, rows, d])

  const counts = Array.from({ length: classes }, (_, j) => view.y.filter((v) => v === j).length)
  const panels = 1 + (pair ? 1 : 0) + (mask ? 1 : 0)
  const fx = useMemo(() => toFlat(field.x), [field])
  const fy = useMemo(() => toFlat(field.y), [field])

  const x = useAxis({ label: data.meta.featureNames[0], hold: 'initial', key: axisKey })
  const y = useAxis({ label: data.meta.featureNames[1], hold: 'initial', key: axisKey, equal: x })
  const yPair = useAxis({ label: data.meta.featureNames[d - 1], hold: 'initial', key: `${axisKey}:${nuisance}` })
  const mx = useAxis({ label: 'feature' })
  const my = useAxis({ label: 'row' })

  return (
    <Figure
      title="Dataset recipe"
      purpose="A dataset is plain data: a registered generator and its knobs, then modifiers in order; where the process has a closed form the Bayes boundary and Bayes error come with it."
      state={state}
      defaultSize="L"
      readouts={{
        classes: (
          <>
            {counts.map((c, j) => (
              <Readout key={j} label={names[j] ?? `class ${j}`} value={`${c} (${pct(c / rows)})`} />
            ))}
          </>
        ),
        recipe: (
          <>
            <Readout label="flipped" value={view.flips.x.length} />
            <Readout label="features" value={d} />
            <Readout label="ignored knobs" value={data.meta.ignored?.length ? data.meta.ignored.join(', ') : 'none'} />
            <Readout
              label="Bayes error"
              value={
                truth
                  ? `${truth.bayesError.toFixed(3)}${truth.bayesErrorMethod === 'monte carlo' ? ` ± ${truth.bayesErrorSe.toFixed(3)} (MC)` : ' (exact)'}`
                  : '—'
              }
            />
          </>
        ),
      }}
      caption={`${describeRecipe(spec)}. ${data.meta.description} Every control comes from the registry's parameter spaces. ${posterior ? 'The field is the Bayes posterior P(y = 1 | x) (pale at ½); the ink line is its 0.5 contour, the Bayes-optimal boundary. ' : ''}Ink diamonds mark flipped labels.${nuisance ? ' The second panel shows x₁ against the last nuisance feature, which carries no information about y.' : ''}${mask ? ' The mask shows missing entries (rows × features).' : ''}`}
    >
      <Plots cols={panels} widths={panels > 1 ? [2, ...new Array<number>(panels - 1).fill(1)] : undefined}>
        <Plot x={x} y={y}>
          {posterior && (
            <Raster
              x={fx}
              y={fy}
              z={posterior}
              scale="diverging"
              range={[0, 1]}
              fillOpacity={0.85}
              valueLabel="P(y = 1 | x)"
            />
          )}
          {posterior && <Contours x={fx} y={fy} z={posterior} levels={[0.5]} />}
          <Points name="points" {...view.shown} groupNames={names} />
          {view.flips.x.length > 0 && <Points name="label flipped" {...view.flips} emphasis />}
          {view.holes.x.length > 0 && <Points name="x₁ or x₂ missing" {...view.holes} slot={4} />}
        </Plot>
        {pair && (
          <Plot x={x} y={yPair}>
            <Points name="points" x={view.px} y={pair} group={view.y} groupNames={names} />
          </Plot>
        )}
        {mask && (
          <Plot x={mx} y={my}>
            <Raster x={mask.x} y={mask.y} z={mask.z} range={[0, 1]} colorBar={false} valueLabel="missing" />
          </Plot>
        )}
      </Plots>
    </Figure>
  )
}
