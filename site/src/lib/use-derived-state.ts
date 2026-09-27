import { useCallback, useState } from 'react'

type Update<T> = T | ((current: T) => T)

/**
 * State that starts from `initial` and returns to it whenever `initial` changes identity (memoise it on its inputs).
 * It replaces calling a reset from an effect, which renders once with stale state and then again. Edits made with
 * `set` hold until `initial` changes or `reset` is called.
 */
export function useDerivedState<T>(initial: T): [T, (update: Update<T>) => void, () => void] {
  const [edit, setEdit] = useState<{ from: T; value: T }>()
  const value = edit && edit.from === initial ? edit.value : initial
  const set = useCallback(
    (update: Update<T>) =>
      setEdit((e) => {
        const current = e && e.from === initial ? e.value : initial
        return { from: initial, value: typeof update === 'function' ? (update as (c: T) => T)(current) : update }
      }),
    [initial],
  )
  const reset = useCallback(() => setEdit(undefined), [])
  return [value, set, reset]
}
