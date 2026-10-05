/**
 * The tables the subgroup pages search, and the quality measures and model classes they offer. Everything here is
 * aifn: the lab picks a dataset and a measure, and draws.
 */
import type { PlantedPattern, TableData } from 'aifn-methods/data'
import { titanic } from 'aifn-methods/data/real'
import { plantedModelFlip, plantedSubgroups } from 'aifn-methods/data/synthetic'
import { stream } from 'aifn-compute/foundation/random'
import {
  associationModel,
  binomialQuality,
  chiSquareQuality,
  correlationModel,
  coverageQuality,
  liftQuality,
  logisticModel,
  meanShiftQuality,
  regressionModel,
  wraccQuality,
  type Direction,
  type ModelMeasure,
  type QualityMeasure,
} from 'aifn-compute/learning/subgroups'
import { choice, int, setting } from 'aifn-render/state'

const cache = new Map<string, TableData>()
const once = (key: string, make: () => TableData) => {
  let d = cache.get(key)
  if (!d) cache.set(key, (d = make()))
  return d
}

// ── Subgroup discovery ───────────────────────────────────────────────────────────────────────────────────────────

export const SD_PROBLEMS = [
  { value: 'planted-outcome', label: 'planted · outcome (binary)' },
  { value: 'planted-cost', label: 'planted · cost (numeric)' },
  { value: 'titanic', label: 'Titanic · survived' },
] as const
export type SdProblemKey = (typeof SD_PROBLEMS)[number]['value']

export type SdProblem = {
  data: TableData
  target: string
  kind: 'binary' | 'numeric'
  /** The planted patterns that move this target. */
  planted: readonly PlantedPattern[]
  values: number[]
}

export function sdProblem(key: SdProblemKey): SdProblem {
  const data =
    key === 'titanic' ? once('titanic', titanic) : once('planted', () => plantedSubgroups(stream('lab/subgroups')))
  const target = key === 'titanic' ? 'survived' : key === 'planted-cost' ? 'cost' : 'outcome'
  return {
    data,
    target,
    kind: key === 'planted-cost' ? 'numeric' : 'binary',
    planted: data.planted.filter((p) => p.targets.includes(target)),
    values: data.table[target] as number[],
  }
}

export const BINARY_MEASURES = [
  { value: 'wracc', label: 'WRAcc' },
  { value: 'binomial', label: 'binomial test z' },
  { value: 'lift', label: 'lift' },
  { value: 'chiSquare', label: 'χ²' },
  { value: 'coverage', label: 'coverage' },
  { value: 'meanShift', label: 'mean shift z' },
] as const
export type MeasureKey = (typeof BINARY_MEASURES)[number]['value']

/** The measure of `key` over the problem's target (a numeric target always uses the mean shift). */
export function sdMeasure(p: SdProblem, key: MeasureKey, direction: Direction, minSupport: number): QualityMeasure {
  if (p.kind === 'numeric' || key === 'meanShift') return meanShiftQuality(p.values, { direction })
  if (key === 'wracc') return wraccQuality(p.values, { direction })
  if (key === 'binomial') return binomialQuality(p.values, { direction })
  if (key === 'lift') return liftQuality(p.values, { minSupport })
  if (key === 'chiSquare') return chiSquareQuality(p.values, { direction: direction === 'both' ? 'both' : direction })
  return coverageQuality(p.values)
}

/** The measure's formula, for the equation band. */
export const MEASURE_TEX: Record<MeasureKey, string> = {
  wracc: String.raw`q = \tfrac{n}{N}(p - p_0),\ \ \hat q = \tfrac{tp}{N}(1 - p_0)`,
  binomial: String.raw`q = \sqrt{n}\,(p - p_0)/\sqrt{p_0(1-p_0)}`,
  lift: String.raw`q = p/p_0,\ \ \hat q = \min(1, tp/m)/p_0`,
  chiSquare: String.raw`q = \chi^2,\ \ \hat q = \max\{\chi^2(\text{positives}), \chi^2(\text{negatives})\}`,
  coverage: String.raw`q = n/N`,
  meanShift: String.raw`q = \sqrt{n}\,(\mu - \mu_0)/\sigma_0,\ \ \hat q = \max_j \sqrt{j}\,(\bar y_{(1..j)} - \mu_0)/\sigma_0`,
}

// ── Exceptional model mining ─────────────────────────────────────────────────────────────────────────────────────

export const EMM_MODELS = [
  { value: 'correlation', label: 'correlation (x, y)' },
  { value: 'regression', label: 'regression slope (y on x)' },
  { value: 'classification', label: 'logistic classifier (label on x)' },
  { value: 'association', label: 'association (a, b)' },
] as const
export type EmmModelKey = (typeof EMM_MODELS)[number]['value']

export const EMM_MEASURES: Record<EmmModelKey, readonly { value: string; label: string }[]> = {
  correlation: [
    { value: 'fisher-z', label: 'Fisher z test' },
    { value: 'absolute', label: '|ρ_G − ρ_Ḡ|' },
    { value: 'entropy', label: 'entropy × |ρ_G − ρ_Ḡ|' },
  ],
  regression: [
    { value: 'slope-difference', label: 'slope t test' },
    { value: 'entropy', label: 'entropy × |b_G − b_Ḡ|' },
    { value: 'cook', label: 'Cook’s distance' },
  ],
  classification: [
    { value: 'wald', label: 'Wald test of the interaction' },
    { value: 'entropy', label: 'entropy × |b_G − b_Ḡ|' },
  ],
  association: [
    { value: 'entropy', label: 'entropy × |Q_G − Q_Ḡ|' },
    { value: 'absolute', label: '|Q_G − Q_Ḡ|' },
  ],
}

export const EMM_TARGETS: Record<EmmModelKey, readonly [string, string]> = {
  correlation: ['x', 'y'],
  regression: ['x', 'y'],
  classification: ['x', 'label'],
  association: ['a', 'b'],
}

export function emmData(): TableData {
  return once('flip', () => plantedModelFlip(stream('lab/emm')))
}

// The fits' shapes differ by class; the page reads them by `measure.model`.
export type AnyModel = ModelMeasure<unknown>

export function emmMeasure(model: EmmModelKey, measure: string): AnyModel {
  const t = emmData().table
  const [a, b] = EMM_TARGETS[model].map((k) => t[k] as number[])
  if (model === 'correlation')
    return correlationModel(a, b, { measure: measure as 'fisher-z' | 'absolute' | 'entropy' }) as AnyModel
  if (model === 'regression')
    return regressionModel(a, b, { measure: measure as 'slope-difference' | 'entropy' | 'cook' }) as AnyModel
  if (model === 'classification') return logisticModel(a, b, { measure: measure as 'wald' | 'entropy' }) as AnyModel
  return associationModel(a, b, { measure: measure as 'absolute' | 'entropy' }) as AnyModel
}

// ── The search and language rows both pages share ──────────────────────────────────────────────────────────────

const STRATEGIES = [
  { value: 'beam', label: 'beam' },
  { value: 'best-first', label: 'best-first' },
  { value: 'depth-first', label: 'depth-first' },
  { value: 'breadth-first', label: 'breadth-first' },
] as const

export const SEARCH_ROW = {
  strategy: choice(STRATEGIES, 'best-first', { label: 'strategy' }),
  beamWidth: int(5, { min: 1, max: 200, label: 'beam width', when: (v) => v.strategy === 'beam' }),
  depth: int(3, { min: 1, max: 4, label: 'max selectors' }),
  k: int(8, { min: 1, max: 40, label: 'top k' }),
  prune: setting(true, 'branch and bound'),
  redundancy: choice(['none', 'cover', 'description'], 'none', { label: 'redundancy filter' }),
} as const

export const LANGUAGE_ROW = {
  discretisation: choice(['equal-frequency', 'equal-width', 'on-the-fly'], 'equal-frequency', {
    label: 'numeric cuts',
  }),
  bins: int(4, { min: 2, max: 12, label: 'intervals' }),
  negations: setting(false, '≠ selectors'),
  minSupport: int(15, { min: 1, max: 2000, label: 'min support' }),
} as const
