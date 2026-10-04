import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'
import { ThemeContext as RenderThemeContext } from 'aifn-render'

export type ThemePreference = 'light' | 'dark' | 'system'
type ThemeContext = {
  preference: ThemePreference
  resolved: 'light' | 'dark'
  setPreference: (p: ThemePreference) => void
}

const Context = createContext<ThemeContext | null>(null)
const STORAGE_KEY = 'mlc-theme'

function readPreference(): ThemePreference {
  try {
    const v = localStorage.getItem(STORAGE_KEY)
    return v === 'light' || v === 'dark' ? v : 'system'
  } catch {
    return 'system'
  }
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [preference, setPreferenceState] = useState<ThemePreference>(readPreference)
  const [systemDark, setSystemDark] = useState(() =>
    typeof matchMedia !== 'undefined' ? matchMedia('(prefers-color-scheme: dark)').matches : false,
  )

  useEffect(() => {
    if (typeof matchMedia === 'undefined') return
    const mq = matchMedia('(prefers-color-scheme: dark)')
    const onChange = () => setSystemDark(mq.matches)
    mq.addEventListener('change', onChange)
    return () => mq.removeEventListener('change', onChange)
  }, [])

  const resolved = preference === 'system' ? (systemDark ? 'dark' : 'light') : preference

  useEffect(() => {
    document.documentElement.classList.toggle('dark', resolved === 'dark')
  }, [resolved])

  const setPreference = (p: ThemePreference) => {
    setPreferenceState(p)
    try {
      localStorage.setItem(STORAGE_KEY, p)
    } catch {
      // Storage unavailable (private mode). The preference lasts for this session only.
    }
  }

  const value = { preference, resolved, setPreference }

  return (
    <Context.Provider value={value}>
      <RenderThemeContext.Provider value={value}>{children}</RenderThemeContext.Provider>
    </Context.Provider>
  )
}

// eslint-disable-next-line react/only-export-components
export function useTheme(): ThemeContext {
  // Both hooks run on every render (rules of hooks); the site's context wins over the render package's.
  const site = useContext(Context)
  const render = useContext(RenderThemeContext)
  const ctx = site ?? render
  if (!ctx) {
    const isDark =
      typeof document !== 'undefined'
        ? document.documentElement.classList.contains('dark') ||
          (typeof window !== 'undefined' && window.matchMedia?.('(prefers-color-scheme: dark)').matches)
        : false
    return {
      preference: 'system',
      resolved: isDark ? 'dark' : 'light',
      setPreference: () => {},
    }
  }
  return ctx
}
