import { Menu } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { Badge } from '@lab/ui/badge'
import { Button } from '@lab/ui/button'
import { cn } from '@lab/lib/utils'
import type { Specimen } from '../specimen'
import { FAMILIES, LAB_FAMILY, familyOf, importPath, TAXONOMY } from './families'
import { FigurePage } from './FigurePage'
import { hrefOf, pathKey, redirectHashRoute, revealHash, specimenPath } from './paths'
import { Sidebar, type Entry, type LabPage, type PlaceholderPage } from './Sidebar'

export type { LabPage, PlaceholderPage }

/**
 * The lab shell: a sidebar index of pages, specimens and taxonomy roadmap placeholders,
 * with search and arbitrary recursive navigation.
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

    // Generate roadmap placeholder entries for taxonomy topics that have no specimens yet
    const specimenModules = new Set(specimens.map((s) => s.module))
    const placeholders: Entry[] = []

    for (const part of TAXONOMY) {
      for (const subj of part.subjects) {
        for (const topic of subj.topics) {
          const modPath =
            topic.id && topic.id !== 'overview' ? `${part.id}/${subj.id}/${topic.id}` : `${part.id}/${subj.id}`
          const hasSpecimen = [...specimenModules].some((m) => m === modPath || m.startsWith(`${modPath}/`))
          if (!hasSpecimen) {
            const key = `placeholder/${part.id}/${subj.id}/${topic.id}`
            const title = topic.title === 'Overview' ? subj.title : topic.title
            placeholders.push({
              kind: 'placeholder',
              key,
              module: modPath,
              family: part.id,
              title,
              search: [part.roman, part.title, subj.title, topic.title, ...(topic.items ?? [])].join(' ').toLowerCase(),
              placeholder: {
                key,
                title,
                description: `Planned topics and capabilities in ${subj.title}.`,
                part: { id: part.id, roman: part.roman, title: part.title },
                subject: { id: subj.id, title: subj.title },
                topic: { id: topic.id, title, items: topic.items ?? [] },
              },
            })
          }
        }
      }
    }

    return [...own, ...listed, ...placeholders]
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
  if (entry.kind === 'page') {
    return (
      <FigurePage scope={entry.key} key={entry.key}>
        <PageHeader kicker="aifn lab" title={entry.title} description={entry.page.description} />
        {entry.page.render()}
      </FigurePage>
    )
  }

  if (entry.kind === 'placeholder') {
    return (
      <FigurePage scope={entry.key} key={entry.key}>
        <PageHeader
          kicker={importPath(entry.module)}
          title={entry.title}
          description={entry.placeholder.description}
          tags={['roadmap', 'planned']}
        />
        <PlaceholderView placeholder={entry.placeholder} />
      </FigurePage>
    )
  }

  const s = entry.specimen
  return (
    <FigurePage scope={entry.key} key={entry.key}>
      <PageHeader kicker={importPath(s.module)} title={s.title} description={s.description} tags={s.tags} />
      <div className="flex flex-col gap-6">{s.render()}</div>
    </FigurePage>
  )
}

function PlaceholderView({ placeholder }: { placeholder: PlaceholderPage }) {
  return (
    <div className="flex max-w-4xl flex-col gap-6">
      <div className="rounded-xl border border-dashed border-border bg-card/60 p-6 text-card-foreground">
        <div className="mb-3 flex items-center gap-2">
          <Badge variant="outline" className="font-mono text-xs tracking-wider text-muted-foreground uppercase">
            Roadmap Topic
          </Badge>
          <span className="text-xs text-muted-foreground">AI Field Notes · Planned Capabilities</span>
        </div>
        <h2 className="mb-2 text-xl font-semibold tracking-tight">{placeholder.title}</h2>
        <p className="mb-6 text-sm text-muted-foreground">
          This topic belongs to{' '}
          <span className="font-medium text-foreground">
            {placeholder.part.roman}: {placeholder.part.title}
          </span>{' '}
          in the <span className="font-medium text-foreground">{placeholder.subject.title}</span> subject. Interactive
          specimens and algorithmic implementations are planned for this section.
        </p>

        {placeholder.topic.items.length > 0 && (
          <div className="space-y-3">
            <h3 className="text-xs font-semibold tracking-wider text-muted-foreground uppercase">
              Planned Coverage & Algorithms
            </h3>
            <ul className="grid grid-cols-1 gap-2 text-sm text-foreground/80 md:grid-cols-2">
              {placeholder.topic.items.map((item) => (
                <li key={item} className="flex items-center gap-2 rounded-md bg-muted/40 px-3 py-2">
                  <span className="size-1.5 shrink-0 rounded-full bg-primary/70" />
                  <span className="truncate">{item}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
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
