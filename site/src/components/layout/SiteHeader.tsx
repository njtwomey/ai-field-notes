import { Menu, Sigma } from 'lucide-react'
import { useState } from 'react'
import { Link, NavLink } from 'react-router'
import { Button } from '@/components/ui/button'
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from '@/components/ui/sheet'
import { cn } from '@/lib/utils'
import { TopicRail } from '@/components/browse/TopicRail'
import { SearchCommand } from './SearchCommand'
import { ThemeToggle } from './ThemeToggle'

const links = [
  { to: '/browse', label: 'Browse' },
  { to: '/tags', label: 'Tags' },
  { to: '/references', label: 'References' },
  { to: '/notation', label: 'Notation' },
]

export function SiteHeader() {
  const [open, setOpen] = useState(false)
  return (
    <header className="sticky top-0 z-40 border-b bg-background/85 backdrop-blur">
      <div className="flex h-14 items-center gap-4 px-4 lg:px-8">
        <Sheet open={open} onOpenChange={setOpen}>
          <SheetTrigger render={<Button variant="ghost" size="icon" className="lg:hidden" aria-label="Menu" />}>
            <Menu />
          </SheetTrigger>
          <SheetContent side="left" className="w-80 overflow-y-auto">
            <SheetHeader>
              <SheetTitle>ML Concepts</SheetTitle>
            </SheetHeader>
            <div className="px-4 pb-6">
              <TopicRail params={{}} onNavigate={() => setOpen(false)} />
            </div>
          </SheetContent>
        </Sheet>
        <Link to="/" className="flex items-center gap-2 font-semibold tracking-tight">
          <Sigma className="size-5" aria-hidden />
          <span className="hidden sm:inline">ML Concepts</span>
        </Link>
        <nav className="hidden items-center gap-1 lg:flex">
          {links.map((l) => (
            <NavLink
              key={l.to}
              to={l.to}
              className={({ isActive }) =>
                cn(
                  'rounded-md px-3 py-1.5 text-sm transition-colors hover:bg-muted',
                  isActive ? 'text-foreground' : 'text-muted-foreground',
                )
              }
            >
              {l.label}
            </NavLink>
          ))}
        </nav>
        <div className="ml-auto flex flex-1 items-center justify-end gap-2">
          <div className="w-full max-w-xs sm:w-auto">
            <SearchCommand />
          </div>
          <ThemeToggle />
        </div>
      </div>
    </header>
  )
}
