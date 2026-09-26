import { useEffect } from 'react'
import { Outlet, useLocation } from 'react-router'
import { ContentErrors } from './ContentErrors'
import { SiteHeader } from './SiteHeader'

export function AppShell() {
  const { pathname, hash } = useLocation()
  useEffect(() => {
    if (!hash) window.scrollTo(0, 0)
  }, [pathname, hash])
  return (
    <div className="min-h-svh bg-background text-foreground">
      <SiteHeader />
      <ContentErrors />
      <Outlet />
    </div>
  )
}

/** Standard page container for index pages (home, browse, tags, references). */
export function PageContainer({ children }: { children: React.ReactNode }) {
  return <main className="mx-auto max-w-5xl px-4 py-10 lg:px-6">{children}</main>
}
