import { Menu } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { Badge } from '@lab/ui/badge'
import { Button } from '@lab/ui/button'
import { cn } from '@lab/lib/utils'
import type { Specimen } from '../specimen'
import { FAMILIES, LAB_FAMILY, familyOf, importPath } from './families'
import { FigurePage } from './FigurePage'
import { hrefOf, pathKey, redirectHashRoute, revealHash, specimenPath } from './paths'
import { Sidebar, type Entry, type LabPage } from './Sidebar'

export type { LabPage }

/**
 * The lab shell: a sidebar index of pages and specimens as a tree of family → module → page, with search, that scrolls on its own; the selected
 * page fills the rest. The page lives in the URL path (`/<module>/<specimen>`), a figure in the hash (`#<figure-id>`).
 */
export function Shell({ specimens, pages = [] }: { specimens: readonly Specimen[]; pages?: readonly LabPage[] }) {
  const entries = useMemo((): Entry[] => {
    const lab = FAMILIES.find((f) => f.id === LAB_FAMILY)!
    const own: Entry[] = pages.map((page) => ({
      kind: 'page',
      key: page.key,
      module: '',
      family: LAB_FAMILY,
      title: page.title,
      search: [lab.title, page.title, page.description].join(' ').toLowerCase(),
      page,
    }))
    const seen = new Set<string>()
    const listed: Entry[] = specimens.map((s) => {
      // `make lab-check` fails on duplicate paths; the suffix only keeps the shell usable meanwhile.
      let key = specimenPath(s)
      for (let n = 2; seen.has(key); n++) key = `${specimenPath(s)}-${n}`
      seen.add(key)
      const family = familyOf(s.module)
      return {
        kind: 'specimen',
        key,
        module: s.module,
        family: family.id,
        title: s.title,
        search: [family.title, s.module, s.title, s.description, ...(s.tags ?? [])].join(' ').toLowerCase(),
        specimen: s,
      }
    })
    return [...own, ...listed]
  }, [specimens, pages])

  const [selected, setSelected] = useState(() => {
    if (typeof location === 'undefined') return ''
    redirectHashRoute()
    return pathKey()
  })
  useEffect(() => {
    const onPop = () => setSelected(pathKey())
    addEventListener('popstate', onPop)
    addEventListener('hashchange', revealHash)
    return () => {
      removeEventListener('popstate', onPop)
      removeEventListener('hashchange', revealHash)
    }
  }, [])
  const current = entries.find((e) => e.key === selected) ?? entries.find((e) => e.kind === 'specimen') ?? entries[0]

  const [open, setOpen] = useState(false)
  const main = useRef<HTMLElement>(null)
  const go = (key: string) => {
    if (key !== selected || location.hash) history.pushState(null, '', hrefOf(key))
    setSelected(key)
    setOpen(false)
    main.current?.scrollTo({ top: 0 })
  }
  // Once a page has rendered, show the figure its hash names (on load, or after following a figure link).
  const shown = current?.key
  useEffect(() => {
    if (!shown || !location.hash) return
    const frame = requestAnimationFrame(revealHash)
    return () => cancelAnimationFrame(frame)
  }, [shown])

  return (
    <div className="flex h-svh overflow-hidden bg-background text-foreground">
      <aside
        className={cn(
          'fixed inset-y-0 left-0 z-40 w-72 shrink-0 border-r bg-sidebar text-sidebar-foreground transition-transform md:static md:translate-x-0',
          open ? 'translate-x-0 shadow-xl' : '-translate-x-full',
        )}
      >
        <Sidebar entries={entries} current={current} onSelect={go} onClose={() => setOpen(false)} />
      </aside>
      {open && <div className="fixed inset-0 z-30 bg-black/30 md:hidden" onClick={() => setOpen(false)} />}
      <main ref={main} className="min-w-0 flex-1 overflow-y-auto overscroll-contain">
        <div className="sticky top-0 z-20 flex items-center gap-2 border-b bg-background/90 px-4 py-2 backdrop-blur md:hidden">
          <Button variant="ghost" size="icon-sm" aria-label="Open index" onClick={() => setOpen(true)}>
            <Menu />
          </Button>
          <span className="truncate text-sm font-medium">{current?.title}</span>
        </div>
        <div className="mx-auto flex w-full max-w-[1400px] flex-col gap-6 px-4 py-6 md:px-8">
          {current ? <Page entry={current} /> : <p className="text-sm text-muted-foreground">No specimens yet.</p>}
        </div>
      </main>
    </div>
  )
}

function Page({ entry }: { entry: Entry }) {
  if (entry.kind === 'page')
    return (
      <FigurePage scope={entry.key} key={entry.key}>
        <PageHeader kicker="aifn lab" title={entry.title} description={entry.page.description} />
        {entry.page.render()}
      </FigurePage>
    )
  const s = entry.specimen
  return (
    // Keyed so a specimen's state starts afresh when another is opened.
    <FigurePage scope={entry.key} key={entry.key}>
      <PageHeader kicker={importPath(s.module)} title={s.title} description={s.description} tags={s.tags} />
      <div className="flex flex-col gap-6">{s.render()}</div>
    </FigurePage>
  )
}

function PageHeader({
  kicker,
  title,
  description,
  tags,
}: {
  kicker: string
  title: string
  description: string
  tags?: readonly string[]
}) {
  return (
    <header className="space-y-1.5">
      <div className="font-mono text-xs text-muted-foreground">{kicker}</div>
      <h1 className="text-xl font-semibold tracking-tight">{title}</h1>
      <p className="max-w-prose text-sm text-muted-foreground">{description}</p>
      {tags && tags.length > 0 && (
        <div className="flex flex-wrap gap-1 pt-1">
          {tags.map((t) => (
            <Badge key={t} variant="secondary" className="font-normal">
              {t}
            </Badge>
          ))}
        </div>
      )}
    </header>
  )
}
