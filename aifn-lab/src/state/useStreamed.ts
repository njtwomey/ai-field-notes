/**
 * `useStreamed(task)`: run a streaming worker task (a generator, such as `aifn-applied/gym` `training`) and follow its
 * partial answers. Each new task (by identity) cancels the one in flight; `null` runs nothing. The value is the latest
 * partial or final answer, so a figure fills in while the worker computes.
 */
import { useEffect, useRef, useState } from 'react'
import type { Task } from './task'
import { ComputeWorker } from './worker'

export type Streamed<T> = {
  /** The latest answer (partial while `running`), or null before the first. */
  value: T | null
  /** True while the task is still yielding. */
  running: boolean
  /** Milliseconds in the worker so far. */
  ms: number
  error?: string
}

export function useStreamed<T>(task: Task<T> | null): Streamed<T> {
  const worker = useRef<ComputeWorker | null>(null)
  // The answers so far, tagged with the task they belong to: a newer task is running until its first answer.
  const [state, setState] = useState<Streamed<T> & { task: Task<T> | null }>({
    task: null,
    value: null,
    running: false,
    ms: 0,
  })
  useEffect(() => {
    // A superseded job is cancelled at once: training runs are long and only the newest matters.
    worker.current ??= new ComputeWorker(0)
    return () => {
      worker.current?.dispose()
      worker.current = null
    }
  }, [])
  useEffect(() => {
    if (!task) return
    worker.current ??= new ComputeWorker(0)
    let live = true
    worker.current.submit(
      task,
      (r) => {
        if (!live) return
        setState((s) =>
          r.ok
            ? { task, value: r.value as T, running: false, ms: r.ms }
            : { ...s, task, running: false, ms: r.ms, error: r.error },
        )
      },
      (value, ms) => live && setState({ task, value: value as T, running: true, ms }),
    )
    return () => {
      live = false
    }
  }, [task])
  const { task: of, ...rest } = state
  return of === task ? rest : { ...rest, running: task !== null, error: undefined }
}
