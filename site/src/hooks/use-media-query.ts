import { useSyncExternalStore } from 'react'

/** Live `matchMedia` result, e.g. `useMediaQuery('(min-width: 1280px)')`. */
export function useMediaQuery(query: string): boolean {
  return useSyncExternalStore(
    (onChange) => {
      const mq = matchMedia(query)
      mq.addEventListener('change', onChange)
      return () => mq.removeEventListener('change', onChange)
    },
    () => matchMedia(query).matches,
  )
}

/** The `xl` breakpoint, at which the note page shows its margin column. */
export const MARGIN_QUERY = '(min-width: 1280px)'
