/**
 * Drawn results of metrics (design S §2.12): curves with fixed axes. Today `aifn/learning/metrics` returns `RocCurve`
 * (`kind: 'roc'`) and `PrecisionRecallCurve` (`kind: 'precision-recall'`) with their own field names; phase 1 moves
 * them onto `Curve`.
 */

import type { Kinded } from './kinds'
import type { Tensor } from './numbers'

/** The curves a metric can draw; each has fixed axes and ranges, declared once next to the type. */
export type CurveName = 'roc' | 'pr' | 'det' | 'gain' | 'cost' | 'prg' | 'reliability' | 'calibration'

/** A drawn metric curve: points (x, y) in threshold order, with the summary area where one is defined. */
export interface Curve extends Kinded<'curve'> {
  readonly curve: CurveName
  readonly x: Tensor
  readonly y: Tensor
  /** Thresholds in decreasing order, aligned with the points (absent for binned curves). */
  readonly thresholds?: Tensor
  /** The summary number (AUROC, average precision, …). */
  readonly area?: number
}
