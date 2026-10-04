/** The worker tasks and fixed settings of the hyphenation pages: the split, the margins, Liang's and the taggers' runs. */
import type { LiangSnapshot, TaggerSnapshot } from 'aifn-methods/text/hyphenation'
import { call, type Task } from '@lab/state'

/** The train/test split is fixed, so every run and the page see the same held-out words. */
export const SPLIT_SEED = 'moby-hyphenation-split'
/** Margins: every dictionary point counts (Moby splits off single letters, as in "a-bout"). */
export const MARGINS = { leftMin: 1, rightMin: 1 } as const

export const dataTask = () =>
  call('data/real/hyphenation/mobyHyphenation', call('foundation/random/stream', SPLIT_SEED), {})

export type LiangSettings = { budget: number }
export function liangTask(s: LiangSettings): Task<LiangSnapshot> {
  return call<LiangSnapshot>(
    'applied/text/hyphenation/liangLearningRun',
    dataTask(),
    s.budget > 0 ? { ...MARGINS, maxPatterns: s.budget } : { ...MARGINS },
  )
}

export type TaggerSettings = { steps: number; seed: number }
export function taggerTask(s: TaggerSettings): Task<TaggerSnapshot> {
  return call<TaggerSnapshot>('applied/text/hyphenation/taggerTrainingRun', dataTask(), {
    steps: s.steps,
    every: Math.max(10, Math.round(s.steps / 15)),
    seed: s.seed,
  })
}
