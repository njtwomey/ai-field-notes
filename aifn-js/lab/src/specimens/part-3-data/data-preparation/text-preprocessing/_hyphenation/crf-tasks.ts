/** The worker task of the CRF hyphenator, shared by the hyphenation showcase and the CRF page. */
import { HYPHENATION_TEMPLATES, type CrfHyphenationSnapshot } from 'aifn-applied/text/hyphenation'
import { call, type Task } from '@lab/state'
import { dataTask } from './tasks'

export type CrfSettings = {
  templates: string
  /** Training words used (0: all). */
  trainWords: number
  minFrequency: number
  optimizer: 'lbfgs' | 'owlqn' | 'sgd' | 'adam'
  c1: number
  c2: number
  stepSize?: number
  batchSize?: number
  maxSteps: number
}

/** The classic templates, every training word, L-BFGS with CRF++'s -c 1 (c₂ = 0.5), 60 iterations. */
export const DEFAULT_CRF: CrfSettings = {
  templates: HYPHENATION_TEMPLATES.classic,
  trainWords: 0,
  minFrequency: 1,
  optimizer: 'lbfgs',
  c1: 0,
  c2: 0.5,
  maxSteps: 60,
}

export function crfTask(s: CrfSettings): Task<CrfHyphenationSnapshot> {
  return call<CrfHyphenationSnapshot>('applied/text/hyphenation/crfHyphenationRun', dataTask(), {
    templates: s.templates,
    trainWords: s.trainWords > 0 ? s.trainWords : undefined,
    minFrequency: s.minFrequency,
    optimizer: s.optimizer,
    c1: s.c1,
    c2: s.c2,
    stepSize: s.stepSize,
    batchSize: s.batchSize,
    maxSteps: s.maxSteps,
    every: 5,
  })
}
