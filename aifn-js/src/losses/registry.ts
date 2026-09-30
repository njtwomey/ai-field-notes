/**
 * The registry of every loss in the module, keyed by `info.key`, so that the lab and training code can list losses and
 * filter them by family, input kind or note.
 */

import * as classification from './classification'
import { isLoss, type Loss, type LossFamily, type LossInput } from './core'
import * as divergence from './divergence'
import * as ranking from './ranking'
import * as regression from './regression'
import * as retrieval from './retrieval'

const modules = [classification, regression, ranking, retrieval, divergence]

/** Every loss, keyed by its `info.key`. */
export const lossRegistry: Readonly<Record<string, Loss>> = Object.freeze(
  Object.fromEntries(
    modules.flatMap((m) => Object.values(m).filter(isLoss)).map((loss) => [loss.info.key, loss] as const),
  ),
)

/** The losses matching every given filter, in registry order (by family, then definition order). */
export function listLosses(filter: { family?: LossFamily; inputs?: LossInput; note?: string } = {}): Loss[] {
  return Object.values(lossRegistry).filter(
    (l) =>
      (filter.family === undefined || l.info.family === filter.family) &&
      (filter.inputs === undefined || l.info.inputs === filter.inputs) &&
      (filter.note === undefined || l.info.note === filter.note),
  )
}

/** The loss with this key; throws for an unknown key. */
export function getLoss(key: string): Loss {
  const l = lossRegistry[key]
  if (!l) throw new Error(`losses: no loss '${key}'`)
  return l
}
