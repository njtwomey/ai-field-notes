/**
 * `aifn-applied/neural`: neural-network applications built from `aifn/nn`: language models (a tiny character-level
 * GPT and a Kneser–Ney n-gram model, on a toy corpus) and contrastive alignment of two views (a tiny CLIP, in
 * `aifn-applied/neural/contrastive`).
 */

import { entries } from 'aifn/foundation/registry'
import type { ModelEntry } from 'aifn/learning/estimators'
import * as languageModels from './language-models'

export { charCorpus, charGpt, Gpt, gptLogits, kneserNey } from './language-models'

/** Every registered estimator factory of `aifn-applied/neural`, keyed by `info.key` (kind `model`). */
export const neuralModelRegistry = entries('model', languageModels) as Readonly<Record<string, ModelEntry>>
