import { cn } from '@/lib/utils'
import { useEffect } from 'react'
import { Outlet, useLocation } from 'react-router'
import { ContentErrors } from './ContentErrors'
import { SiteFooter } from './SiteFooter'
import { SiteHeader } from './SiteHeader'

export function AppShell() {
  const { pathname, hash } = useLocation()
  useEffect(() => {
    if (!hash) window.scrollTo(0, 0)
  }, [pathname, hash])
  return (
    <div className="flex min-h-svh flex-col bg-background text-foreground">
      <SiteHeader />
      <ContentErrors />
      <div className="flex-1">
        <Outlet />
      </div>
      <SiteFooter />
    </div>
  )
}

/** Standard page container for index pages (home, browse, tags, references). */
export function PageContainer({ children, wide = false }: { children: React.ReactNode; wide?: boolean }) {
  return <main className={cn('mx-auto px-4 py-10 lg:px-6', wide ? 'max-w-screen-2xl' : 'max-w-5xl')}>{children}</main>
}
