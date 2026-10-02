/**
 * Pieces shared by the two mixture density figures: the run task, the checkpoint picked by the player (a new run opens
 * at step 0) and the networks rebuilt from a snapshot's specs. The training-curve plots are in `curves.tsx`.
 */
import { useMemo, useState } from 'react'
import type { Params } from 'aifn/foundation/pytree'
import { mdnModel, type MdnCheckpoint, type MdnModel, type MdnSnapshot } from 'aifn-applied/learning/mixture-density'
import { call, type Task } from '@lab/state'

/** Slots: the squared-error network is slot 0 everywhere, the MDN slot 1 (its second mode slot 2). */
export const SLOT = { mean: 0, mdn: 1, second: 2 } as const

/** The settings of one run: the dataset (a registered generator and its knobs) and the networks. */
export type RunSettings = {
  generator: string
  knobs: Record<string, number>
  dataSeed: number
  components: number
  hidden: number[]
  stepSize: number
  steps: number
  seed: number
}

/** The worker task: `mixtureDensityRun` on the registered dataset, 50 checkpoints. */
export const runTask = (s: RunSettings): Task<MdnSnapshot> =>
  call<MdnSnapshot>('applied/learning/mixture-density/mixtureDensityRun', {
    data: call(`applied/data/synthetic/${s.generator}`, call('foundation/random/stream', s.dataSeed), s.knobs),
    components: s.components,
    hidden: s.hidden,
    stepSize: s.stepSize,
    steps: s.steps,
    every: Math.max(1, Math.round(s.steps / 50)),
    seed: s.seed,
  })

/** Both networks of a run, rebuilt from its specs. */
export function useNetworks(snap: MdnSnapshot | undefined): { mdn: MdnModel; mean: MdnModel } | null {
  const key = snap ? JSON.stringify([snap.spec, snap.meanSpec]) : ''
  return useMemo(() => {
    if (!key) return null
    const [spec, meanSpec] = JSON.parse(key) as [MdnSnapshot['spec'], MdnSnapshot['spec']]
    return { mdn: mdnModel(spec), mean: mdnModel(meanSpec) }
  }, [key])
}

/** The checkpoint shown: picked on the player or by dragging the step marker; a new run opens at step 0. */
export function useCheckpoint(snap: MdnSnapshot | undefined, run: unknown) {
  const shots = snap?.checkpoints ?? []
  const [picked, setPicked] = useState<{ run: unknown; index: number } | null>(null)
  const index = Math.min(picked && picked.run === run ? picked.index : 0, Math.max(0, shots.length - 1))
  const pick = (i: number) => setPicked({ run, index: i })
  const pickStep = (step: number) => {
    let best = 0
    shots.forEach((c, i) => {
      if (Math.abs(c.step - step) < Math.abs(shots[best].step - step)) best = i
    })
    pick(best)
  }
  const shot: MdnCheckpoint | undefined = shots[index]
  return { shots, index, shot, pick, pickStep }
}

/** Parameters of a checkpoint, typed for `mdnPredict`. */
export const paramsOf = (p: unknown) => p as Params[]
