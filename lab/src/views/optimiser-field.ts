/**
 * The optimiser of a small network's training run: Adam with a typed learning rate, or full-batch L-BFGS (core
 * `methodTraining`, through `fullBatchTraining`) with its memory. Offer L-BFGS only where the whole training set fits
 * one evaluation; its iterations are line-searched steps on the full objective, so the iteration count means the same.
 * `trainingMethodOf` turns the field's value into the plain `TrainingMethod` a worker run receives.
 *
 *   optimiser: optimiserField({ stepSize: 0.01 })            // in useFigureState
 *   method: trainingMethodOf(state.optimiser)               // in the run's options
 */
import type { TrainingMethod } from 'aifn-compute/nn/training'
import { float, int, variants } from 'aifn-render/state'
import type { ReactNode } from 'react'

export type OptimiserFieldOptions = {
  /** Adam's initial learning rate (default 0.01). */
  stepSize?: number
  /** Adam's suggested rates (default around `stepSize`). */
  suggestions?: readonly number[]
  /** Offer full-batch L-BFGS (default true). */
  lbfgs?: boolean
  /** The case first chosen (default `adam`). */
  initial?: 'adam' | 'lbfgs'
  /** The row's label. */
  label?: ReactNode
  /** Adam's label, e.g. 'AdamW (decoupled decay)' when the run uses it. */
  adamLabel?: string
}

/** Adam with a typed log-scale rate, and (unless `lbfgs: false`) full-batch L-BFGS with its memory m. */
export function optimiserField(options: OptimiserFieldOptions = {}) {
  const { stepSize = 0.01, lbfgs = true, initial = 'adam', label, adamLabel = 'Adam' } = options
  const suggestions = options.suggestions ?? [stepSize / 3, stepSize, stepSize * 3].map((v) => Number(v.toPrecision(2)))
  const adam = {
    label: adamLabel,
    params: { stepSize: float(stepSize, { label: 'learning rate', gt: 0, le: 1, scale: 'log10', suggestions }) },
  }
  const full = {
    label: 'L-BFGS (full batch)',
    params: { memory: int(10, { label: 'memory m', ge: 1, le: 50, suggestions: [3, 5, 10, 20] }) },
  }
  return variants(lbfgs ? { adam, lbfgs: full } : { adam }, {
    label,
    choiceLabel: 'optimiser',
    initial: lbfgs ? initial : 'adam',
  })
}

/** The plain training method of the field's value (`batchSize` for Adam's minibatches, default the whole set). */
export function trainingMethodOf(
  v: { key: string; values: Record<string, unknown> },
  extra: { batchSize?: number; clipNorm?: number } = {},
): TrainingMethod {
  if (v.key === 'lbfgs') return { method: 'lbfgs', memory: Number(v.values.memory) }
  return { method: 'adam', stepSize: Number(v.values.stepSize), ...extra }
}

/** A short label of the chosen optimiser, for captions and readouts. */
export const optimiserLabel = (v: { key: string; values: Record<string, unknown> }) =>
  v.key === 'lbfgs' ? `L-BFGS (full batch, m = ${String(v.values.memory)})` : `Adam (η = ${String(v.values.stepSize)})`
