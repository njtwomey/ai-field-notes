/**
 * `useTrainedRun(settings, makeTask)`: the run of a streamed worker task (`useStreamed`) built from the settings current
 * when Train is pressed; a changed setting marks the shown run as stale until Retrain. Drawn by `TrainControls`.
 */
import { useMemo, useState } from 'react'
import { useStreamed, type Streamed, type Task } from 'aifn-render/state'

export type TrainedRun<S, T> = {
  /** The settings of the run shown (null before the first Train). */
  trained: S | null
  /** The settings changed since the run shown was started. */
  stale: boolean
  run: Streamed<T> & { stop: () => void }
  /** Start a run with the current settings. */
  train: () => void
}

/**
 * The run of a streamed worker task built from settings at the moment Train is pressed. `settings` must be plain
 * data (compared as JSON).
 */
export function useTrainedRun<S, T>(settings: S, makeTask: (settings: S) => Task<T>): TrainedRun<S, T> {
  const [trained, setTrained] = useState<{ settings: S } | null>(null)
  const key = JSON.stringify(settings)
  const stale = trained !== null && JSON.stringify(trained.settings) !== key
  // A fresh object per press, so pressing again with the same settings reruns.
  // oxlint-disable-next-line react-hooks/exhaustive-deps -- the task follows the press, not the builder's identity
  const task = useMemo(() => (trained ? makeTask(trained.settings) : null), [trained])
  const run = useStreamed<T>(task)
  return { trained: trained?.settings ?? null, stale, run, train: () => setTrained({ settings }) }
}
