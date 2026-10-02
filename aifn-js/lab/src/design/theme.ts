import { createContext, useContext } from 'react'
import type { Mode } from './palette'

export type ThemePreference = 'light' | 'dark' | 'system'

export type ThemeContextValue = {
  preference: ThemePreference
  /** The theme in effect: the preference, or the system's scheme when the preference is `system`. */
  resolved: Mode
  setPreference: (p: ThemePreference) => void
}

export const ThemeContext = createContext<ThemeContextValue | null>(null)

export function useTheme(): ThemeContextValue {
  const ctx = useContext(ThemeContext)
  if (!ctx) throw new Error('useTheme must be used inside <ThemeProvider>')
  return ctx
}
