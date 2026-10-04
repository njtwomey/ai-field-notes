/**
 * `aifn-methods/evaluation`: application metrics, defined with the metric registry pattern of `aifn/learning/metrics`:
 * text, detection and segmentation, signal and image quality, generative-model, fairness and beyond-accuracy metrics,
 * collected in `evaluationMetricRegistry`. Ordinal metrics are in `aifn/learning/metrics`.
 */

import { entries } from 'aifn/foundation/registry'
import type { Metric } from 'aifn/learning/metrics'
import * as text from './text'
import * as detection from './detection'
import * as quality from './quality'
import * as generative from './generative'
import * as fairness from './fairness'
import * as beyondAccuracy from './beyond-accuracy'

/** Every application metric, keyed by its `info.key` (the counterpart of `aifn/learning/metrics`' `metricRegistry`). */
export const evaluationMetricRegistry: Readonly<Record<string, Metric>> = entries(
  'metric',
  text,
  detection,
  quality,
  generative,
  fairness,
  beyondAccuracy,
) as unknown as Readonly<Record<string, Metric>>
