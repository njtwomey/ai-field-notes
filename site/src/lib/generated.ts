import { useEffect, useState } from 'react'
import type { Manifest, RunRecord } from '@/generated/contracts'

const cache = new Map<string, Promise<unknown>>()

/** URL of a file under site/public/generated/. */
export function generatedUrl(relative: string): string {
  return `${import.meta.env.BASE_URL}generated/${relative}`
}

function fetchJson<T>(relative: string): Promise<T> {
  let pending = cache.get(relative)
  if (!pending) {
    pending = fetch(generatedUrl(relative)).then((r) => {
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}: generated/${relative}. Run \`npm run assets\`.`)
      return r.json()
    })
    cache.set(relative, pending)
  }
  return pending as Promise<T>
}

export type Loadable<T> = { data?: T; error?: Error; loading: boolean }

export function useGeneratedJson<T>(relative: string | undefined): Loadable<T> {
  // Results are keyed by path, so a new path reads as loading without resetting state inside the effect.
  const [state, setState] = useState<{ relative: string; data?: T; error?: Error }>()
  useEffect(() => {
    if (!relative) return
    let live = true
    fetchJson<T>(relative).then(
      (data) => live && setState({ relative, data }),
      (error: Error) => live && setState({ relative, error }),
    )
    return () => {
      live = false
    }
  }, [relative])
  if (!relative) return { loading: false }
  if (state?.relative !== relative) return { loading: true }
  return { data: state.data, error: state.error, loading: false }
}

export const useManifest = () => useGeneratedJson<Manifest>('manifest.json')

export const useRun = (path: string | undefined) => useGeneratedJson<RunRecord>(path)

/**
 * Data written by a Python `@figure` builder (python/mlc/figures/). The type parameter mirrors the builder's pydantic
 * return model.
 */
export function useFigure<T>(id: string): Loadable<T> {
  return useGeneratedJson<T>(`figures/${id}.json`)
}
