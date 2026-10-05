/**
 * Shared by the explanation pages: a small MLP trained in the worker (`aifn-methods/neural/full-batch`'s
 * `fullBatchComparison`, Adam on minibatches) and the trained network as an `aifn/learning/explain` `DenseNetwork`;
 * number and image formatting.
 */
import { useMemo, useState } from 'react'
import { comparisonModel, type ComparisonSnapshot } from 'aifn-methods/neural/full-batch'
import { fromMlpParams, type DenseNetwork } from 'aifn-compute/learning/explain'
import { call, useStreamed, type Task } from 'aifn-render/state'

/** What a training run is: the data task, the network and the optimiser budget. */
export type MlpSetup = {
  data: Task
  inputs: number
  width: number
  depth: number
  activation: 'tanh' | 'relu'
  steps: number
  rate: number
  l2: number
  seed: number
  batch?: number
}

/** The streamed training run of a setup (null until Train is pressed) and the network at the chosen checkpoint. */
export function useTrainedMlp(setup: MlpSetup | null, options: { latest?: boolean } = {}) {
  const task = useMemo((): Task<ComparisonSnapshot> | null => {
    if (!setup) return null
    return call<ComparisonSnapshot>('applied/neural/full-batch/fullBatchComparison', setup.data, {
      task: 'classification',
      network: { width: setup.width, depth: setup.depth, activation: setup.activation },
      optimisers: ['adam'],
      iterations: setup.steps,
      adamStep: setup.rate,
      batchSize: setup.batch ?? 64,
      l2: setup.l2,
      seed: setup.seed,
      checkpoints: 40,
    })
  }, [setup])
  const run = useStreamed(task)
  const latest = run.value?.runs[0]?.checkpoints
  const checkpoints = useMemo(() => latest ?? [], [latest])
  const [picked, setPicked] = useState<{ task: typeof task; index: number } | null>(null)
  const last = Math.max(0, checkpoints.length - 1)
  // The player follows the newest checkpoint until the reader picks one.
  const cpIndex = picked?.task === task && !options.latest ? Math.min(picked.index, last) : last
  const net = useMemo((): DenseNetwork | null => {
    if (!setup) return null
    const cp = checkpoints[cpIndex]
    if (!cp) return null
    const { unravel } = comparisonModel({
      inputs: setup.inputs,
      width: setup.width,
      depth: setup.depth,
      activation: setup.activation,
    })
    return fromMlpParams(unravel(cp.theta) as object[], setup.activation)
  }, [setup, checkpoints, cpIndex])
  return {
    run,
    task,
    checkpoints,
    cpIndex,
    setCp: (i: number) => setPicked({ task, index: i }),
    net,
    accuracy: run.value?.runs[0]?.score ?? NaN,
    done: run.value ? run.value.done / Math.max(1, run.value.total) : 0,
  }
}

/** Format a number to a few significant digits (— when not finite). */
export const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? String(Number(v.toPrecision(digits))) : '—')

/** Rows of an image [h, w] from a flat row-major vector, top row last (so row 0 is drawn at the top). */
export const imageRows = (v: ArrayLike<number>, h: number, w: number, offset = 0) =>
  Array.from({ length: h }, (_, r) => Array.from({ length: w }, (_, c) => v[offset + (h - 1 - r) * w + c]))
