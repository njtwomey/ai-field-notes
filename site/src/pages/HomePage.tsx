import { ArrowRight, Search, Waypoints } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router'
import { TopicIcon } from '@/components/layout/category-icon'
import { openSearch } from '@/components/layout/SearchCommand'
import { Kbd } from '@/components/ui/kbd'
import { browseUrl, notes, notesInCategory, references, taxonomy } from '@/lib/content'
import { cn } from '@/lib/utils'

/**
 * Editorial Home Page:
 * Re-invented for the canonical 8-Part Master Taxonomy.
 * Compact, formal, and functional:
 * - Dynamic scroll-spy floating index that highlights and auto-centers the active section
 * - Clear distinctions between the 8 Parts
 * - Compact subject rows with inline child-topic pills and note counts
 */
export function HomePage() {
  const [activePart, setActivePart] = useState<string>(taxonomy[0]?.path ?? '')
  const navContainerRef = useRef<HTMLDivElement>(null)
  const buttonRefs = useRef<Record<string, HTMLButtonElement | null>>({})
  const isClickScrolling = useRef(false)

  const totalSubjects = useMemo(() => taxonomy.reduce((sum, p) => sum + p.children.length, 0), [])

  // Scroll-spy: track which section is currently in view
  useEffect(() => {
    const handleScroll = () => {
      if (isClickScrolling.current) return
      const scrollY = window.scrollY
      const offset = 240

      for (let i = taxonomy.length - 1; i >= 0; i--) {
        const el = document.getElementById(taxonomy[i].path)
        if (el) {
          const top = el.getBoundingClientRect().top + window.scrollY
          if (scrollY >= top - offset) {
            setActivePart(taxonomy[i].path)
            return
          }
        }
      }
      if (taxonomy[0]) setActivePart(taxonomy[0].path)
    }

    window.addEventListener('scroll', handleScroll, { passive: true })
    handleScroll()
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  // Auto-center the active part button in the floating nav
  useEffect(() => {
    if (!activePart) return
    const btn = buttonRefs.current[activePart]
    const container = navContainerRef.current
    if (btn && container) {
      const btnLeft = btn.offsetLeft
      const btnWidth = btn.offsetWidth
      const containerWidth = container.offsetWidth
      const targetScroll = btnLeft - (containerWidth - btnWidth) / 2
      container.scrollTo({ left: targetScroll, behavior: 'smooth' })
    }
  }, [activePart])

  const scrollToPart = (path: string) => {
    setActivePart(path)
    isClickScrolling.current = true
    const el = document.getElementById(path)
    if (el) {
      el.scrollIntoView({ behavior: 'smooth' })
    }
    setTimeout(() => {
      isClickScrolling.current = false
    }, 800)
  }

  return (
    <main className="mx-auto max-w-6xl px-4 pt-12 pb-24 lg:px-8">
      {/* Hero Section */}
      <section className="mx-auto max-w-2xl space-y-5 text-center">
        <h1 className="font-prose text-4xl font-bold tracking-tight sm:text-5xl">AI Field Notes</h1>

        <p className="font-prose text-base leading-relaxed text-muted-foreground sm:text-lg">
          Machine learning notes from my research career, digitised and made interactive.
        </p>

        <button
          type="button"
          onClick={openSearch}
          className="flex h-11 w-full items-center gap-3 rounded-xl border bg-muted/40 px-4 text-left text-muted-foreground shadow-xs transition-all hover:border-foreground/30 hover:bg-muted"
        >
          <Search className="size-4.5" aria-hidden />
          <span className="flex-1 text-sm">Search concepts, techniques and tests…</span>
          <Kbd>⌘K</Kbd>
        </button>

        <div className="flex flex-wrap items-center justify-center gap-x-5 gap-y-2 text-xs text-muted-foreground sm:text-sm">
          <span>
            <strong className="font-semibold text-foreground">{notes.length}</strong> notes ·{' '}
            <strong className="font-semibold text-foreground">8</strong> parts ·{' '}
            <strong className="font-semibold text-foreground">{totalSubjects}</strong> subjects ·{' '}
            <strong className="font-semibold text-foreground">{Object.keys(references).length}</strong> references
          </span>
          <span className="hidden text-muted-foreground/40 sm:inline">|</span>
          <Link to={browseUrl()} className="inline-flex items-center gap-1 font-medium text-foreground hover:underline">
            Browse all <ArrowRight className="size-3.5" aria-hidden />
          </Link>
          <Link
            to={browseUrl({ view: 'map' })}
            className="inline-flex items-center gap-1 font-medium text-foreground hover:underline"
          >
            <Waypoints className="size-3.5" aria-hidden /> Open map
          </Link>
        </div>
      </section>

      {/* Floating Index with Auto-Centering & Scroll-Spy */}
      <nav
        ref={navContainerRef}
        aria-label="Canon volume index"
        className="sticky top-14 z-20 -mx-4 mt-10 overflow-x-auto border-y border-border/60 bg-background/85 px-4 py-2.5 shadow-xs backdrop-blur-md lg:mx-0 lg:rounded-xl lg:border"
      >
        <div className="mx-auto flex min-w-max items-center justify-center gap-1.5">
          {taxonomy.map((part) => {
            const count = notesInCategory(part.path).length
            const isSelected = activePart === part.path
            return (
              <button
                key={part.path}
                ref={(el) => {
                  buttonRefs.current[part.path] = el
                }}
                type="button"
                onClick={() => scrollToPart(part.path)}
                className={cn(
                  'inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-xs transition-all select-none',
                  isSelected
                    ? 'bg-foreground font-semibold text-background shadow-xs'
                    : 'text-muted-foreground hover:bg-muted hover:text-foreground',
                )}
                title={`${part.roman ?? ''}: ${part.title} (${count} notes)`}
              >
                <span className={cn('font-mono text-[10px]', isSelected ? 'opacity-90' : 'opacity-70')}>
                  {part.roman ?? ''}
                </span>
                <span className="max-w-[130px] truncate sm:max-w-none">{part.title}</span>
                <span className={cn('font-mono text-[10px] tabular-nums', isSelected ? 'opacity-80' : 'opacity-50')}>
                  {count}
                </span>
              </button>
            )
          })}
        </div>
      </nav>

      {/* Part Sections: Refined, Functional, Compact Chapter Rows */}
      <div className="mt-8 space-y-10">
        {taxonomy.map((part, idx) => {
          const partNotesCount = notesInCategory(part.path).length

          return (
            <section
              key={part.path}
              id={part.path}
              aria-labelledby={`heading-${part.path}`}
              className="scroll-mt-28 border-t border-border/70 pt-8 pb-4 first:border-t-0 first:pt-2"
            >
              {/* Part Header: Clear Distinction */}
              <div className="mb-3 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
                <div className="flex flex-wrap items-baseline gap-x-2.5 gap-y-1">
                  <span className="font-mono text-xs font-bold tracking-wider text-primary uppercase">
                    {part.roman ?? `Part ${idx + 1}`}
                  </span>
                  <h2
                    id={`heading-${part.path}`}
                    className="font-prose text-xl font-bold tracking-tight text-foreground sm:text-2xl"
                  >
                    {part.title}
                  </h2>
                  <span className="text-xs text-muted-foreground tabular-nums">
                    ({partNotesCount} {partNotesCount === 1 ? 'note' : 'notes'})
                  </span>
                </div>

                <Link
                  to={browseUrl({ part: part.path })}
                  className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground hover:underline"
                >
                  Browse {part.roman ? `${part.roman}` : ''} →
                </Link>
              </div>

              {part.description && (
                <p className="mb-3.5 max-w-3xl text-xs leading-relaxed text-muted-foreground">{part.description}</p>
              )}

              {/* List of top-level items (Subjects), and inline child-topic pills */}
              <div className="divide-y divide-border/30 rounded-lg border border-border/50 bg-card/20 px-3.5 py-0.5">
                {part.children.map((subject) => {
                  const subjectNotesCount = notesInCategory(subject.path).length
                  const hasNotes = subjectNotesCount > 0

                  return (
                    <div
                      key={subject.path}
                      className="flex flex-col gap-2 py-2.5 md:flex-row md:items-baseline md:gap-4"
                    >
                      {/* Top-level subject item */}
                      <div className="flex shrink-0 items-center gap-2 md:w-52">
                        <TopicIcon icon={subject.icon} className="size-3.5 shrink-0 text-muted-foreground" />
                        <Link
                          to={browseUrl({ c: subject.path })}
                          className={cn(
                            'truncate text-xs font-semibold hover:underline',
                            hasNotes ? 'text-foreground' : 'text-muted-foreground',
                          )}
                          title={subject.title}
                        >
                          {subject.title}
                        </Link>
                        <span className="ml-auto font-mono text-[10px] text-muted-foreground/70 tabular-nums md:ml-1">
                          {subjectNotesCount}
                        </span>
                      </div>

                      {/* Inline list of child-topics under that level as buttons/pills with note counts at the end */}
                      <div className="flex min-w-0 flex-1 flex-wrap items-center gap-1.5">
                        {subject.children.length > 0 ? (
                          subject.children.map((topic) => {
                            const count = notesInCategory(topic.path).length
                            return (
                              <Link
                                key={topic.path}
                                to={browseUrl({ c: topic.path })}
                                className={cn(
                                  'inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs transition-colors',
                                  count > 0
                                    ? 'bg-muted/70 text-foreground hover:bg-muted hover:text-foreground'
                                    : 'border border-dashed border-border/60 text-muted-foreground/50 hover:bg-muted/20 hover:text-muted-foreground',
                                )}
                              >
                                <span className="max-w-[180px] truncate sm:max-w-[220px]">{topic.title}</span>
                                <span className="font-mono text-[10px] text-muted-foreground tabular-nums">
                                  {count}
                                </span>
                              </Link>
                            )
                          })
                        ) : (
                          <span className="text-[11px] text-muted-foreground/50 italic">
                            {hasNotes ? `${subjectNotesCount} overview notes` : 'Roadmap area · notes planned'}
                          </span>
                        )}
                      </div>
                    </div>
                  )
                })}
              </div>
            </section>
          )
        })}
      </div>
    </main>
  )
}
