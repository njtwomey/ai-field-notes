import { useEffect, useState } from 'react'
import { useStreamed, type Task } from 'aifn-render'
import type { PinnRun } from 'aifn-methods/neural/pinn'

/**
 * Streams an engine `pinnRun` from the compute worker and keeps every snapshot it sends, so that a `Player` can scrub
 * the training. The list restarts when `task` changes identity, so pass a memoised task.
 *
 * @param task The worker task, `call('neural/pinn/pinnRun', problem, options)`.
 * @returns The snapshots so far, whether the run is still going, and the worker's error, if any.
 */
export function useStreamedRun(task: Task<PinnRun>): { snapshots: PinnRun[]; running: boolean; error?: string } {
  const run = useStreamed(task)
  const [kept, setKept] = useState<{ task: Task<PinnRun>; snapshots: PinnRun[] }>({ task, snapshots: [] })
  const value = run.value
  useEffect(() => {
    if (!value) return
    setKept((k) => {
      const list = k.task === task ? k.snapshots : []
      if (list.length && list[list.length - 1].done === value.done && !value.finished) return k
      if (list.length && list[list.length - 1].done === value.done)
        return { task, snapshots: [...list.slice(0, -1), value] }
      return { task, snapshots: [...list, value] }
    })
  }, [value, task])
  return { snapshots: kept.task === task ? kept.snapshots : [], running: run.running, error: run.error }
}
