import type { ReactNode } from 'react'
import type { ThemePreference } from '@lab/design/theme'
import { ThemeProvider } from '@lab/design/ThemeProvider'
import { TooltipProvider } from '@lab/ui/tooltip'

/** Everything the lab's components expect around them: the theme and the tooltip provider. */
export function Providers({ children, theme }: { children: ReactNode; theme?: ThemePreference }) {
  return (
    <ThemeProvider initial={theme}>
      <TooltipProvider delay={300}>{children}</TooltipProvider>
    </ThemeProvider>
  )
}
