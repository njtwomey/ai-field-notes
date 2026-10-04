/**
 * The optimiser of a template CRF as one figure field: L-BFGS with CRF++'s `-c` (L2), OWL-QN with c₁ and c₂ (L1 + L2,
 * elastic net), SGD and Adam with `-c` and a step size; and the training options it gives (`crfTrainingOptions`).
 */
import type { CrfOptimizer } from 'aifn-methods/inference/sequence-models'
import { choice, float, variants } from '@lab/state'

/** The `variants` field of a CRF's optimiser and its regularisation. */
/** Minibatch sizes of SGD and Adam; 0 is the full batch (every sequence in one step). */
const BATCHES = [1, 4, 16, 64, 256, 0] as const

/**
 * `unit` names what a sequence is on the page (sentences, words) in the batch-size label. SGD and Adam carry a batch
 * size, so the control shows only for them.
 */
export function crfOptimiserField(
  label = 'optimiser',
  defaults: { C?: number; c1?: number; c2?: number; batchSize?: number } = {},
  unit = 'sequences',
) {
  const C = defaults.C ?? 1
  const batch = () =>
    choice(
      BATCHES.map((b) => ({ value: b, label: b === 0 ? `full (all ${unit})` : String(b) })),
      (defaults.batchSize ?? 16) as (typeof BATCHES)[number],
      { label: `batch size (${unit})` },
    )
  return variants(
    {
      lbfgs: {
        label: 'L-BFGS (L2)',
        params: {
          C: float(C, { label: '-c C (L2: c₂ = 1/2C)', gt: 0, scale: 'log10', suggestions: [0.1, 1, 10, 100] }),
        },
      },
      owlqn: {
        label: 'OWL-QN (L1 + L2)',
        params: {
          c1: float(defaults.c1 ?? 0.5, {
            label: 'c₁ (L1)',
            ge: 0,
            scale: 'log10',
            suggestions: [0.05, 0.2, 0.5, 1, 2],
          }),
          c2: float(defaults.c2 ?? 0.01, { label: 'c₂ (L2)', ge: 0, step: 0.01, suggestions: [0, 0.01, 0.1, 0.5] }),
        },
      },
      sgd: {
        label: 'SGD (L2)',
        params: {
          C: float(C, { label: '-c C (L2)', gt: 0, scale: 'log10', suggestions: [0.1, 1, 10, 100] }),
          stepSize: float(0.1, { label: 'step size', gt: 0, scale: 'log10', suggestions: [0.01, 0.05, 0.1, 0.3] }),
          batchSize: batch(),
        },
      },
      adam: {
        label: 'Adam (L2)',
        params: {
          C: float(C, { label: '-c C (L2)', gt: 0, scale: 'log10', suggestions: [0.1, 1, 10, 100] }),
          stepSize: float(0.05, { label: 'step size', gt: 0, scale: 'log10', suggestions: [0.01, 0.05, 0.1] }),
          batchSize: batch(),
        },
      },
    },
    { label, choiceLabel: 'optimiser' },
  )
}

/** The plain training options of a chosen optimiser case (CRF++'s C as c₂ = 1/(2C)). */
export function crfTrainingOptions(v: { key: string; values: Record<string, unknown> }): {
  optimizer: CrfOptimizer
  c1: number
  c2: number
  stepSize?: number
  batchSize?: number
} {
  const n = (k: string) => Number(v.values[k])
  switch (v.key) {
    case 'owlqn':
      return { optimizer: 'owlqn', c1: n('c1'), c2: n('c2') }
    case 'sgd':
    case 'adam':
      return { optimizer: v.key, c1: 0, c2: 1 / (2 * n('C')), stepSize: n('stepSize'), batchSize: n('batchSize') }
    default:
      return { optimizer: 'lbfgs', c1: 0, c2: 1 / (2 * n('C')) }
  }
}

/** A short description of the regularisation of a training run, e.g. "L-BFGS, -c 1 (c₂ = 0.5)". */
export function describeOptimiser(o: {
  optimizer: CrfOptimizer
  c1: number
  c2: number
  stepSize?: number
  batchSize?: number
}): string {
  const C = o.c2 > 0 ? 1 / (2 * o.c2) : Infinity
  const c = Number.isFinite(C) ? `-c ${Number(C.toPrecision(3))}` : 'no L2'
  switch (o.optimizer) {
    case 'owlqn':
      return `OWL-QN, c₁ = ${o.c1}, c₂ = ${o.c2}`
    case 'sgd':
      return `SGD (η = ${o.stepSize}, batch ${batchName(o.batchSize)}), ${c}`
    case 'adam':
      return `Adam (η = ${o.stepSize}, batch ${batchName(o.batchSize)}), ${c}`
    default:
      return `L-BFGS, ${c}`
  }
}

const batchName = (b: number | undefined) => (b === 0 ? 'full' : String(b ?? 16))
