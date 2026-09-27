import { useEffect, useMemo, useRef, useState } from 'react'
import { Navigate, useNavigate, useParams } from 'react-router'
import { NoteContext } from '@/components/content/note-context'
import { MarginNotes, SidenoteProvider } from '@/components/content/sidenotes'
import { Toc } from '@/components/layout/Toc'
import { CodeIndex, CodeTab } from '@/components/note/CodeTab'
import { ConceptTab } from '@/components/note/ConceptTab'
import { NoteAside } from '@/components/note/NoteAside'
import { NoteBar, type NoteTab } from '@/components/note/NoteBar'
import { NoteHeader } from '@/components/note/NoteHeader'
import { OutputsTab, RunIndexNav } from '@/components/note/OutputsTab'
import { noteReferences, notesBySlug, noteUrl, prefetchNotesWhenIdle } from '@/lib/content'
import { MARGIN_QUERY, useMediaQuery } from '@/hooks/use-media-query'
import { sectionTrail, useReadingPosition } from '@/hooks/use-reading-position'
import { useManifest } from '@/lib/generated'

type Tab = NoteTab

/**
 * Three columns: sticky, independently scrolling index on the left (TOC, files or runs, per tab); content in the
 * centre; references and relations on the right, scrolling with the page.
 */
export function NotePage({ tab }: { tab: Tab }) {
  const { slug = '', run: runParam } = useParams()
  const navigate = useNavigate()
  const note = notesBySlug.get(slug)
  const { data: manifest } = useManifest()
  const example = note?.code ? manifest?.examples[note.code] : undefined
  const [file, setFile] = useState<string>()
  const contentRef = useRef<HTMLDivElement>(null)
  const wide = useMediaQuery(MARGIN_QUERY)

  const headings = useMemo(
    () =>
      !note || tab !== 'concept'
        ? []
        : noteReferences(note).length
          ? [...note.headings, { id: 'references', text: 'References', depth: 2 }]
          : note.headings,
    [note, tab],
  )
  const position = useReadingPosition(headings)

  // Warm the notes this one points to, so following a link is instant.
  useEffect(() => {
    if (!note) return
    return prefetchNotesWhenIdle([...note.linked, ...note.requires, ...note.related, ...note.partOf])
  }, [note])

  if (!note) return <Navigate to="/404" replace />
  if (tab !== 'concept' && !note.code) return <Navigate to={noteUrl(slug)} replace />

  const activeFile = file && example?.sources.includes(file) ? file : example?.sources[0]
  const activeRun = example?.runs.find((r) => r.name === runParam) ?? example?.runs[0]

  const left =
    tab === 'concept' ? (
      <Toc items={headings} active={position.active} />
    ) : !example ? null : tab === 'code' ? (
      <CodeIndex slug={slug} example={example} file={activeFile!} onSelect={setFile} />
    ) : (
      <RunIndexNav slug={slug} example={example} active={activeRun?.name ?? ''} />
    )

  const centre =
    tab === 'concept' ? (
      <ConceptTab note={note} />
    ) : !manifest ? (
      <p className="text-sm text-muted-foreground">Loading…</p>
    ) : !example ? (
      <p className="text-sm text-destructive">
        No generated outputs for example “{note.code}”. Run <code>make assets</code>.
      </p>
    ) : tab === 'code' ? (
      <CodeTab example={example} file={activeFile!} />
    ) : activeRun ? (
      <OutputsTab run={activeRun} />
    ) : null

  return (
    <NoteContext.Provider value={note}>
      <SidenoteProvider contentRef={contentRef}>
        <main className="px-4 pb-8 lg:px-8">
          <div className="grid grid-cols-1 gap-10 lg:grid-cols-[220px_minmax(0,1fr)] xl:grid-cols-[240px_minmax(0,1fr)_300px]">
            <aside className="hidden pt-8 lg:block">
              <div className="sticky top-30 max-h-[calc(100svh-8.5rem)] overflow-y-auto overscroll-contain pr-2 pb-6">{left}</div>
            </aside>
            <div ref={contentRef} className="min-w-0 space-y-6">
              <NoteBar
                note={note}
                tab={tab}
                onTab={(t) => navigate(noteUrl(slug, t === 'concept' ? undefined : t))}
                section={sectionTrail(headings, position.active)}
                progress={tab === 'concept' ? position.progress : undefined}
              />
              <NoteHeader note={note} />
              {centre}
            </div>
            <aside className="space-y-10 lg:col-start-2 xl:col-start-auto xl:pt-8">
              <NoteAside note={note} />
              {tab === 'concept' && wide && <MarginNotes />}
            </aside>
          </div>
        </main>
      </SidenoteProvider>
    </NoteContext.Provider>
  )
}
