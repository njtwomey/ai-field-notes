/**
 * One dataset and task selector for training pages: a variants field whose cases are `aifn-applied/data`'s registered
 * generators (or `regression1d` functions), each with a typed size and a noise slider read from the registry's knob
 * space, remembered per case and kept in the URL like any field. The same choice builds the worker task (`task`) and,
 * from the same stream, the dataset the page draws (`make`), so the two always agree.
 *
 *   const DATA = datasetChoice({ moons: { n: 200, noise: 0.15 }, circles: {}, xor: { knobs: { kind: 'gaussian' } } })
 *   const state = useFigureState({ data: DATA.field({ label: '1 · data' }) })
 *   const task = DATA.task(state.data, 'my-page-data')   // a worker Task<Dataset>
 *   const data = DATA.make(state.data, 'my-page-data')   // the same Dataset, in the page
 */
import { datasetRegistry, type Dataset } from 'aifn-applied/data'
import { stream, type Stream } from 'aifn/foundation/random'
import type { Space } from 'aifn/foundation/space'
import { call, int, slider, variants, type ParamDef, type ParamDefs, type Task, type VariantsDef } from '@lab/state'
import type { ReactNode } from 'react'

/** A case of the choice: a registered generator with page defaults, or a page's own generator (`custom`). */
export type DatasetCase = {
  /** The registry key (default: the case's own key; `fn:<name>` cases use `regression1d`). */
  generator?: string
  label?: string
  /** Default size (else the registry's). */
  n?: number
  /** Default noise (else the registry's); `false` offers no noise control. */
  noise?: number | false
  /** Fixed knobs passed through (e.g. `{ kind: 'gaussian' }`, `{ range: [-3, 3] }`). */
  knobs?: Record<string, unknown>
  /**
   * A generator outside the dataset registry (e.g. a model's own toy data): the worker task and the in-page build from
   * a seed and the chosen size and noise.
   */
  custom?: {
    task: (random: Task<Stream>, knobs: { n: number; noise: number }) => Task<unknown>
    /** The in-page build (omit when the page draws the data the worker returns). */
    make?: (random: Stream, knobs: { n: number; noise: number }) => Dataset
  }
}

/** What the field gives: the chosen case and its values. */
export type DatasetValue = { key: string; values: Record<string, unknown> }

type Resolved = {
  generator: string
  /** The registry's task for the generator ('regression' for `fn:` cases). */
  task: string
  label: string
  knobs: Record<string, unknown>
  noiseKnob: string | null
  custom?: DatasetCase['custom']
}

const REGRESSION_LABELS: Record<string, string> = {
  sine: 'sin x',
  linear: '0.5 + 0.8x',
  cubic: 'x³ − x',
  step: 'sign x',
  sinc: 'sin(πx)/(πx)',
  bump: 'exp(−8x²)',
  doppler: 'Doppler',
}

/** The registry knob that sets a generator's noise level, if any. */
function noiseKnobOf(knobs: Space): string | null {
  for (const k of ['noise', 'sd']) if (knobs.dims[k]?.type === 'real') return k
  return null
}

/**
 * The selector over `cases` (keys are registry generator keys, or `fn:<name>` for `regression1d` with that function).
 * Case order is the dropdown's order; classification and regression cases may share one picker (`taskOf`).
 */
export function datasetChoice(cases: Record<string, DatasetCase>) {
  const resolved: Record<string, Resolved> = {}
  const fields: Record<string, { label: string; params: ParamDefs }> = {}
  for (const [key, c] of Object.entries(cases)) {
    const fn = key.startsWith('fn:') ? key.slice(3) : null
    const generator = c.generator ?? (fn ? 'regression1d' : key)
    const entry = c.custom ? undefined : datasetRegistry[generator]
    if (!c.custom && !entry) throw new Error(`datasetChoice: no registered generator '${generator}'`)
    const space = entry?.info.knobs as Space | undefined
    const noiseKnob = c.noise === false ? null : space ? noiseKnobOf(space) : 'noise'
    const nDim = space?.dims.n
    const nMax = nDim && nDim.type === 'int' ? nDim.max : 5000
    const nDefault = c.n ?? (nDim && nDim.type === 'int' ? nDim.default : 200)
    const params: Record<string, ParamDef> = {
      n: int(nDefault, { ge: 2, le: nMax, suggestions: [50, 100, 200, 400], label: 'points n' }),
    }
    if (noiseKnob) {
      const dim = space?.dims[noiseKnob]
      const hi = dim && dim.type === 'real' ? Math.min(dim.max, 1) : 1
      const initial = typeof c.noise === 'number' ? c.noise : dim && dim.type === 'real' ? dim.default : 0.1
      params.noise = slider(0, Math.max(hi, initial), initial, {
        step: 0.01,
        label: noiseKnob === 'sd' ? 'noise (sd)' : 'noise',
      })
    }
    const label = c.label ?? (fn ? (REGRESSION_LABELS[fn] ?? fn) : (entry?.info.name.toLowerCase() ?? key))
    resolved[key] = {
      generator,
      task: fn ? 'regression' : (entry?.info.task ?? 'classification'),
      label,
      knobs: { ...(fn ? { fn } : {}), ...(c.knobs ?? {}) },
      noiseKnob,
      custom: c.custom,
    }
    fields[key] = { label, params }
  }

  const knobsOf = (v: DatasetValue) => {
    const r = resolved[v.key]
    const n = Number(v.values.n)
    const noise = Number(v.values.noise ?? 0)
    return { r, n, noise, knobs: { ...r.knobs, n, ...(r.noiseKnob ? { [r.noiseKnob]: noise } : {}) } }
  }

  return {
    /** The variants field (the generator picker, then its size and noise). */
    field: (extra: { label?: ReactNode; initial?: string; when?: VariantsDef['when']; choiceLabel?: ReactNode } = {}) =>
      variants(fields, { choiceLabel: 'dataset', ...extra }),
    /** The chosen generator's task ('classification', 'regression', 'clustering', …): one picker can mix tasks. */
    taskOf: (v: DatasetValue) => resolved[v.key].task,
    /** A one-line description, e.g. "two moons, n = 200, noise 0.15". */
    describe: (v: DatasetValue) => {
      const { r, n, noise } = knobsOf(v)
      return `${r.label}, n = ${n}${r.noiseKnob ? `, noise ${noise}` : ''}`
    },
    /** A key that changes exactly when the dataset does (for memos and stream names). */
    key: (v: DatasetValue) => JSON.stringify([v.key, knobsOf(v).knobs]),
    /** The worker task that builds the dataset from the stream named `seed`. */
    task: (v: DatasetValue, seed: string | number): Task<Dataset> => {
      const { r, n, noise, knobs } = knobsOf(v)
      const random = call<Stream>('foundation/random/stream', seed)
      if (r.custom) return r.custom.task(random, { n, noise }) as Task<Dataset>
      return call<Dataset>(`applied/data/synthetic/${r.generator}`, random, knobs)
    },
    /** The same dataset, built in the page. */
    make: (v: DatasetValue, seed: string | number): Dataset => {
      const { r, n, noise, knobs } = knobsOf(v)
      if (r.custom) {
        if (!r.custom.make) throw new Error(`datasetChoice: case '${v.key}' has no in-page build`)
        return r.custom.make(stream(seed), { n, noise })
      }
      return (datasetRegistry[r.generator] as unknown as (s: Stream, k: unknown) => Dataset)(stream(seed), knobs)
    },
  }
}

export type DatasetChoice = ReturnType<typeof datasetChoice>

/** The common classification sets (two moons, circles, XOR, spirals) with sizes and noise for small networks. */
export const CLASSIFICATION_CASES = {
  moons: { label: 'two moons', n: 200, noise: 0.15 },
  circles: { label: 'two circles', n: 200, noise: 0.08, knobs: { factor: 0.5 } },
  xor: { label: 'XOR', n: 200, noise: 0.45, knobs: { kind: 'gaussian' } },
  spirals: { label: 'two spirals', n: 200, noise: 0.03, knobs: { arms: 2 } },
} as const satisfies Record<string, DatasetCase>

/** The common one-dimensional regression curves. */
export const REGRESSION_CASES = {
  'fn:sine': { n: 80, noise: 0.1, knobs: { range: [-3, 3] } },
  'fn:sinc': { n: 80, noise: 0.05, knobs: { range: [-3, 3] } },
  'fn:cubic': { n: 80, noise: 0.1, knobs: { range: [-1.5, 1.5] } },
  'fn:step': { n: 80, noise: 0.1 },
  'fn:bump': { n: 80, noise: 0.05 },
} as const satisfies Record<string, DatasetCase>
