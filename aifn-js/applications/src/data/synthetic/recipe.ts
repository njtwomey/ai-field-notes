/**
 * Dataset recipes: a dataset described by plain, JSON-serialisable parameters (a base generator, its size and class
 * balance, an overlap knob, and modifiers), so a figure's state or URL can hold the spec and rebuild the same data.
 * The dataset records the contract `Recipe` it was made by in `meta.recipe` (base, seed, knobs, modifiers), and the
 * knobs its base could not use in `meta.ignored`.
 */

import { stream, child } from 'aifn/foundation/random'
import {
  withLabelNoise,
  withMissing,
  withNuisanceFeatures,
  withOutliers,
  withTransform,
  type MissingMechanism,
} from './modifiers'
import { blobs, checkerboard, circles, gaussians, moons, rings, shuffleDataset, spirals, xor } from './points'
import { friedman1, linearRegressionData, regression1d, type RegressionFunction } from './regression'
import type { Dataset } from '../types'

/** Plain JSON values. */
export type Json = null | boolean | number | string | Json[] | { [key: string]: Json }

/** Labelled bases. */
export type ClassificationBase =
  'blobs' | 'moons' | 'circles' | 'rings' | 'spirals' | 'xor' | 'checkerboard' | 'gaussians'
/** Real-target bases. */
export type RegressionBase = 'regression1d' | 'linearRegression' | 'friedman1'
export type RecipeBase = ClassificationBase | RegressionBase

/** Every base, labelled ones first. */
export const RECIPE_BASES: readonly RecipeBase[] = [
  'moons',
  'circles',
  'blobs',
  'gaussians',
  'xor',
  'spirals',
  'rings',
  'checkerboard',
  'regression1d',
  'linearRegression',
  'friedman1',
]

/**
 * A dataset spec. Steps run in a fixed order, each on its own substream of `stream(seed)`: the base generator
 * (`base`), then `transform`, `outliers`, `nuisance`, `labelNoise`, `missing` and `shuffle`. Knobs that do not apply
 * to the base (e.g. `separation` for moons) are not used and are listed in `meta.ignored`, so a figure can switch
 * bases without clearing its state and still say which of its controls had no effect.
 */
export interface DatasetRecipe {
  base: RecipeBase
  /** Default 0. */
  seed?: number | string
  /** Total points (the base's default when omitted). */
  n?: number
  /** Per-class counts, in place of `n` with a prevalence or weights. */
  sizes?: number[]
  /** Two-class bases: the share of class 1. */
  prevalence?: number
  /** One weight per class. */
  classWeights?: number[]
  /**
   * The base's noise: noise sd for moons, circles, rings, spirals and regression; the within-class sd for blobs,
   * Gaussian XOR (which it switches on) and gaussians (covariance noise² I when none is given).
   */
  noise?: number
  /** Blobs: centre spacing in blob sds; gaussians: d′, the Mahalanobis distance of the closest means. */
  separation?: number
  /** Any other option of the base generator (e.g. `factor`, `arms`, `turns`, `tiles`, `centers`, `fn`, `d`). */
  options?: { [key: string]: Json }
  /** `withTransform`: x ↦ A x + b. */
  transform?: { matrix: number[][]; offset?: number[] }
  /** `withOutliers`: a fraction of rows (or its options). */
  outliers?: number | { fraction: number; scale?: number }
  /** `withNuisanceFeatures`: a count of Gaussian features (or its options). */
  nuisance?: number | { count: number; kind?: 'gaussian' | 'uniform'; scale?: number }
  /** `withLabelNoise`: a symmetric flip rate, or a noise matrix. */
  labelNoise?: number | number[][]
  /** `withMissing`: a rate (MCAR) or its options. */
  missing?: number | { rate: number; mechanism?: MissingMechanism; strength?: number; observed?: number }
  /** Shuffle the rows at the end (bases group points by class). Default false. */
  shuffle?: boolean
}

const CLASSIFICATION = new Set<string>([
  'blobs',
  'moons',
  'circles',
  'rings',
  'spirals',
  'xor',
  'checkerboard',
  'gaussians',
])

/** True for bases with class labels. */
export function isClassificationBase(base: RecipeBase): base is ClassificationBase {
  return CLASSIFICATION.has(base)
}

function sizeOptions(r: DatasetRecipe) {
  if (r.sizes) return { n: r.sizes }
  return { n: r.n, prevalence: r.prevalence, classWeights: r.classWeights }
}

function defined<T extends object>(o: T): T {
  return Object.fromEntries(Object.entries(o).filter(([, v]) => v !== undefined)) as T
}

function base(r: DatasetRecipe): Dataset {
  const s = child(stream(r.seed ?? 0), 'base')
  const o = (r.options ?? {}) as Record<string, never>
  const size = sizeOptions(r)
  const noise = r.noise
  switch (r.base) {
    case 'moons':
      return moons(s, defined({ ...o, ...size, noise }))
    case 'circles':
      return circles(s, defined({ ...o, ...size, noise }))
    case 'rings':
      return rings(s, defined({ ...o, ...size, noise }))
    case 'spirals':
      return spirals(s, defined({ ...o, ...size, noise }))
    case 'blobs':
      return blobs(s, defined({ ...o, ...size, sd: noise, separation: r.separation }))
    case 'xor': {
      const kind = (o as { kind?: 'uniform' | 'gaussian' }).kind ?? (noise !== undefined ? 'gaussian' : 'uniform')
      return xor(s, defined({ ...o, ...size, kind, sd: noise }))
    }
    case 'checkerboard':
      return checkerboard(s, defined({ ...o, ...size }))
    case 'gaussians': {
      const opts = o as { means?: number[][]; covariances?: number[][][] }
      const d = opts.means?.[0]?.length ?? 2
      const k = opts.means?.length ?? 2
      const sd = noise ?? 1
      const covariances =
        opts.covariances ??
        Array.from({ length: k }, () =>
          Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => (i === j ? sd * sd : 0))),
        )
      return gaussians(s, defined({ ...o, ...size, covariances, separation: r.separation }))
    }
    case 'regression1d':
      return regression1d(s, defined({ ...(o as { fn?: RegressionFunction }), n: r.n, noise }))
    case 'linearRegression':
      return linearRegressionData(s, defined({ ...o, n: r.n, noise }))
    case 'friedman1':
      return friedman1(s, defined({ ...o, n: r.n, noise }))
    default:
      throw new RangeError(`recipe: unknown base ${String((r as { base: unknown }).base)}`)
  }
}

/** The knobs of a recipe that its base does not use. */
export function ignoredKnobs(r: DatasetRecipe): string[] {
  const classes = isClassificationBase(r.base)
  const out: string[] = []
  if (r.noise !== undefined && r.base === 'checkerboard') out.push('noise')
  if (r.separation !== undefined && r.base !== 'blobs' && r.base !== 'gaussians') out.push('separation')
  if (!classes) {
    for (const key of ['sizes', 'prevalence', 'classWeights', 'labelNoise'] as const)
      if (r[key] !== undefined) out.push(key)
  }
  return out
}

/**
 * Build the dataset a recipe describes. The same recipe (after a JSON round trip too) gives the same dataset; each
 * step records itself in `meta.recipe` (with the recipe's seed), the truth follows every step, and knobs the base
 * cannot use are listed in `meta.ignored`.
 */
export function recipe(r: DatasetRecipe): Dataset {
  const root = stream(r.seed ?? 0)
  const made = base(r)
  const ignored = ignoredKnobs(r)
  let d: Dataset = {
    ...made,
    meta: {
      ...made.meta,
      recipe: made.meta.recipe && { ...made.meta.recipe, seed: r.seed ?? 0 },
      ...(ignored.length ? { ignored } : {}),
    },
  }
  if (r.transform) d = withTransform(d, r.transform.matrix, r.transform.offset)
  if (r.outliers !== undefined) {
    const o = typeof r.outliers === 'number' ? { fraction: r.outliers } : r.outliers
    if (o.fraction > 0) d = withOutliers(child(root, 'outliers'), d, o)
  }
  if (r.nuisance !== undefined) {
    const o = typeof r.nuisance === 'number' ? { count: r.nuisance } : r.nuisance
    if (o.count > 0) d = withNuisanceFeatures(child(root, 'nuisance'), d, o)
  }
  if (r.labelNoise !== undefined && isClassificationBase(r.base)) {
    const noisy = typeof r.labelNoise === 'number' ? r.labelNoise > 0 : true
    if (noisy)
      d = withLabelNoise(
        child(root, 'labelNoise'),
        d,
        typeof r.labelNoise === 'number' ? { rate: r.labelNoise } : { matrix: r.labelNoise },
      )
  }
  if (r.missing !== undefined) {
    const o = typeof r.missing === 'number' ? { rate: r.missing } : r.missing
    if (o.rate > 0) d = withMissing(child(root, 'missing'), d, o)
  }
  if (r.shuffle) d = shuffleDataset(child(root, 'shuffle'), d)
  return d
}

/** A recipe as a compact string for a URL query (URI-encoded JSON). */
export function encodeRecipe(r: DatasetRecipe): string {
  return encodeURIComponent(JSON.stringify(r))
}

/** The recipe in a string from `encodeRecipe`, checked. Throws on anything that is not a recipe. */
export function decodeRecipe(text: string): DatasetRecipe {
  return parseRecipe(JSON.parse(decodeURIComponent(text)))
}

const KEYS = new Set([
  'base',
  'seed',
  'n',
  'sizes',
  'prevalence',
  'classWeights',
  'noise',
  'separation',
  'options',
  'transform',
  'outliers',
  'nuisance',
  'labelNoise',
  'missing',
  'shuffle',
])

/** Check that a parsed JSON value is a recipe (known keys, a known base, numbers where numbers go). */
export function parseRecipe(value: unknown): DatasetRecipe {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) throw new TypeError('recipe: not an object')
  const r = value as Record<string, unknown>
  for (const key of Object.keys(r)) if (!KEYS.has(key)) throw new TypeError(`recipe: unknown key ${key}`)
  if (!RECIPE_BASES.includes(r.base as RecipeBase)) throw new TypeError(`recipe: unknown base ${String(r.base)}`)
  for (const key of ['n', 'prevalence', 'noise', 'separation'])
    if (r[key] !== undefined && typeof r[key] !== 'number') throw new TypeError(`recipe: ${key} must be a number`)
  return r as unknown as DatasetRecipe
}

function fmt(v: number): string {
  return String(+v.toPrecision(3))
}

/**
 * One line describing a recipe, e.g. "400 points of moons (noise 0.2), 20% in class 1, 5% labels flipped, seed 7".
 * Knobs that do not apply to the base are left out.
 */
export function describeRecipe(r: DatasetRecipe): string {
  const parts: string[] = []
  const classes = isClassificationBase(r.base)
  const knobs: string[] = []
  if (r.noise !== undefined && r.base !== 'checkerboard')
    knobs.push(
      r.base === 'blobs' || r.base === 'gaussians' || r.base === 'xor' ? `sd ${fmt(r.noise)}` : `noise ${fmt(r.noise)}`,
    )
  if (r.separation !== undefined && (r.base === 'blobs' || r.base === 'gaussians'))
    knobs.push(r.base === 'gaussians' ? `d′ = ${fmt(r.separation)}` : `separation ${fmt(r.separation)} sd`)
  for (const [k, v] of Object.entries(r.options ?? {}))
    if (typeof v === 'number' || typeof v === 'string') knobs.push(`${k} ${typeof v === 'number' ? fmt(v) : v}`)
  const count = r.sizes ? `${r.sizes.join(' + ')} points` : r.n !== undefined ? `${r.n} points` : 'points'
  parts.push(`${count} of ${r.base}${knobs.length ? ` (${knobs.join(', ')})` : ''}`)
  if (classes && !r.sizes) {
    if (r.prevalence !== undefined) parts.push(`${fmt(100 * r.prevalence)}% in class 1`)
    else if (r.classWeights) parts.push(`class weights ${r.classWeights.map(fmt).join(' : ')}`)
  }
  if (r.transform) parts.push('linearly transformed')
  const outliers = typeof r.outliers === 'number' ? r.outliers : r.outliers?.fraction
  if (outliers) parts.push(`${fmt(100 * outliers)}% outliers`)
  const nuisance = typeof r.nuisance === 'number' ? r.nuisance : r.nuisance?.count
  if (nuisance) parts.push(`${nuisance} nuisance feature${nuisance === 1 ? '' : 's'}`)
  if (classes && r.labelNoise !== undefined) {
    if (typeof r.labelNoise !== 'number') parts.push('class-conditional label noise')
    else if (r.labelNoise > 0) parts.push(`${fmt(100 * r.labelNoise)}% labels flipped`)
  }
  const missing = typeof r.missing === 'number' ? { rate: r.missing } : r.missing
  if (missing && missing.rate > 0)
    parts.push(`${fmt(100 * missing.rate)}% missing (${(missing.mechanism ?? 'mcar').toUpperCase()})`)
  if (r.shuffle) parts.push('shuffled')
  parts.push(`seed ${r.seed ?? 0}`)
  return parts.join(', ')
}
