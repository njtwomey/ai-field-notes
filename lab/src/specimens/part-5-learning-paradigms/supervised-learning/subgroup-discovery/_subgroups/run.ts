/** The search run behind the subgroup pages: every state of `subgroupDiscoverySteps`, its history and ranges. */
import type { PlantedPattern } from 'aifn-methods/data'
import { trace } from 'aifn-compute/foundation/trace'
import {
  bitsetJaccard,
  subgroupDiscoverySteps,
  type Description,
  type QualityMeasure,
  type SelectorLanguage,
  type SubgroupOptions,
} from 'aifn-compute/learning/subgroups'
import { searchHistory, type SearchHistory, type SearchState, type SearchVisit } from 'aifn-compute/optim/search'
import { useMemo } from 'react'

export const f3 = (v: number) => (Number.isFinite(v) ? Number(v.toPrecision(3)).toString() : v > 0 ? '∞' : '−∞')

// ── The run ─────────────────────────────────────────────────────────────────────────────────────────────────────

/** Steps a run may take (each beam step is a level; other strategies take one node per step). */
const MAX_STEPS = 6000
/** Descriptions a run may evaluate. */
const MAX_NODES = 12000

export type Run = {
  steps: readonly SearchState<Description>[]
  history: SearchHistory<Description>
  /** Each visit's fate when it was generated. */
  born: readonly SearchVisit<Description>['fate'][]
  /** Visits by depth, in id order: a fixed horizontal slot for the tree plot. */
  slot: Float64Array
  maxDepth: number
  /** The finite qualities' and estimates' ranges over the whole run, so axes hold while the run plays. */
  qualityRange: [number, number]
  boundRange: [number, number]
}

/** Run subgroup discovery to the end, keeping every state. */
export function useRun(lang: SelectorLanguage, measure: QualityMeasure, options: SubgroupOptions): Run {
  return useMemo(() => {
    const t = trace(subgroupDiscoverySteps(lang, measure, { ...options, maxNodes: MAX_NODES }), undefined, MAX_STEPS, {
      timing: false,
    })
    const steps = t.steps
    const history = searchHistory(steps)
    const born: SearchVisit<Description>['fate'][] = []
    for (const s of steps) for (const v of s.generated) if (v.id >= 0) born[v.id] = v.fate
    const byDepth = new Map<number, number[]>()
    for (const v of history.visits) {
      const list = byDepth.get(v.depth) ?? []
      list.push(v.id)
      byDepth.set(v.depth, list)
    }
    const slot = new Float64Array(history.visits.length)
    for (const [depth, ids] of byDepth)
      ids.forEach((id, r) => (slot[id] = depth - 0.42 + (0.84 * (r + 0.5)) / ids.length))
    const span = (xs: number[]): [number, number] => {
      const f = xs.filter(Number.isFinite)
      if (!f.length) return [0, 1]
      const lo = Math.min(...f)
      const hi = Math.max(...f)
      const pad = 0.06 * (hi - lo || 1)
      return [lo - pad, hi + pad]
    }
    return {
      steps,
      history,
      born,
      slot,
      maxDepth: options.maxDepth ?? 3,
      qualityRange: span(history.visits.map((v) => v.quality)),
      boundRange: span(history.visits.map((v) => v.bound)),
    }
    // oxlint-disable-next-line react-hooks/exhaustive-deps -- options is rebuilt each render; its fields are the deps
  }, [lang, measure, JSON.stringify(options)])
}

/** What a node is at step k: 0 open, 1 expanded, 2 pruned, 3 not refined (dropped from the beam, or a leaf). */
export function fateAt(run: Run, id: number, k: number): 0 | 1 | 2 | 3 {
  const h = run.history
  if (h.expandedAt[id] >= 0 && h.expandedAt[id] <= k) return 1
  if (h.discardedAt[id] >= 0 && h.discardedAt[id] <= k) return 2
  const b = run.born[id]
  return b === 'pruned' ? 2 : b === 'dropped' || b === 'leaf' ? 3 : 0
}

type Visible = { ids: number[]; fate: number[] }
export function visibleAt(run: Run, k: number): Visible {
  const ids: number[] = []
  const fate: number[] = []
  run.history.visits.forEach((v, id) => {
    if (run.history.generatedAt[id] <= k && Number.isFinite(v.quality)) {
      ids.push(id)
      fate.push(fateAt(run, id, k))
    }
  })
  return { ids, fate }
}

/** The planted pattern a cover matches best, with its Jaccard index. */
export function plantedMatch(lang: SelectorLanguage, planted: readonly PlantedPattern[], d: Description) {
  let best = { index: -1, jaccard: 0 }
  planted.forEach((p, i) => {
    const j = bitsetJaccard(lang.cover(d), lang.cover(p.description))
    if (j > best.jaccard) best = { index: i, jaccard: j }
  })
  return best
}
