import { ArrowRight, FileText, Folder } from 'lucide-react'
import { Link } from 'react-router'
import { MathText } from '@/components/content/MathText'
import { browseUrl, noteUrl, notes, prefetchNote, type CategoryNode, type NoteMeta } from '@/lib/content'
import { cn } from '@/lib/utils'
import { inCategory } from './filters'

/**
 * One level of the taxonomy: the topic named by `?c=`, its overview, then what sits directly under it (its subtopics,
 * which open as their own pages, and its own notes in reading order). Nothing below that level is shown.
 */
export function TopicContents({ node, shown }: { node: CategoryNode; shown: NoteMeta[] }) {
  const index = node.index ? shown.find((n) => n.slug === node.index) : undefined
  // The overview carries the topic's own name; an index note with another title is listed first instead.
  const overview = index?.title === node.title ? index : undefined
  const own = shown.filter((n) => n.category === node.path && n.slug !== index?.slug)
  const listed = index && !overview ? [index, ...own] : own
  const topics = node.children.filter((c) => shown.some((n) => inCategory(n, c.path)))
  const both = topics.length > 0 && listed.length > 0
  const blurb = overview?.summary ?? node.description

  return (
    <div className="space-y-10">
      <header
        className={cn(
          'group relative overflow-hidden rounded-xl border bg-linear-to-br from-muted/70 via-muted/20 to-transparent p-5 sm:p-6',
          overview && 'transition-colors hover:border-foreground/30 hover:bg-muted/40',
        )}
      >
        <span className="absolute inset-y-0 left-0 w-1 bg-foreground/70" aria-hidden />
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] font-semibold tracking-wider text-muted-foreground uppercase">
          <span>{node.roman ?? 'Overview'}</span>
          <Count path={node.path} />
        </div>
        <h2 className="mt-2 font-prose text-2xl font-semibold sm:text-3xl">{node.title}</h2>
        {blurb && (
          <p className="mt-2 max-w-3xl text-muted-foreground">
            <MathText text={blurb} />
          </p>
        )}
        {overview && (
          <Link
            to={noteUrl(overview.slug)}
            onMouseEnter={() => prefetchNote(overview.slug)}
            // The link covers the whole card, so a click anywhere on it opens the overview.
            className="mt-4 inline-flex items-center gap-1 text-sm font-medium group-hover:underline after:absolute after:inset-0"
          >
            Read the overview <ArrowRight className="size-4 transition-transform group-hover:translate-x-0.5" />
          </Link>
        )}
      </header>

      {topics.length > 0 && (
        <section className="space-y-2">
          {both && <SectionHeading>Topics</SectionHeading>}
          <ul className="divide-y border-y">
            {topics.map((c) => (
              <TopicRow key={c.path} node={c} />
            ))}
          </ul>
        </section>
      )}

      {listed.length > 0 && (
        <section className="space-y-2">
          {both && <SectionHeading>Notes</SectionHeading>}
          <ol className="divide-y border-y">
            {listed.map((n) => (
              <NoteRow key={n.slug} note={n} />
            ))}
          </ol>
        </section>
      )}

      {!topics.length && !listed.length && <p className="text-sm text-muted-foreground">No notes here yet.</p>}
    </div>
  )
}

/** The top level: the Parts, the pillars of the collection, each with the topics it holds. */
export function PartsIndex({ parts, shown }: { parts: CategoryNode[]; shown: NoteMeta[] }) {
  return (
    <ol className="grid gap-x-10 border-t md:grid-cols-2">
      {parts.map((p) => {
        const count = shown.filter((n) => inCategory(n, p.path)).length
        return (
          <li key={p.path} className="border-b">
            <Link
              to={browseUrl({ c: p.path })}
              className="group -mx-3 flex gap-5 rounded-md px-3 py-7 transition-colors hover:bg-muted/50"
            >
              <span className="w-14 shrink-0 font-prose text-3xl text-muted-foreground">
                {p.roman?.replace(/^Part /, '')}
              </span>
              <span className="min-w-0 space-y-2">
                <span className="block font-prose text-2xl font-semibold group-hover:underline">{p.title}</span>
                {p.description && <span className="block text-sm text-muted-foreground">{p.description}</span>}
                <span className="block text-xs leading-relaxed text-muted-foreground">
                  {p.children.map((c) => c.title).join(' · ')}
                </span>
                <span className="block text-xs text-muted-foreground tabular-nums">{count} notes</span>
              </span>
            </Link>
          </li>
        )
      })}
    </ol>
  )
}

/** A subtopic: its name, what it covers and how many notes it holds; it opens as its own browse page. */
function TopicRow({ node }: { node: CategoryNode }) {
  const blurb = overviewOf(node)?.summary ?? node.description
  return (
    <li>
      <Link
        to={browseUrl({ c: node.path })}
        className="group -mx-2 flex items-start gap-3 rounded-md px-2 py-3 transition-colors hover:bg-muted/50"
      >
        <Folder className="mt-0.5 size-4 shrink-0 text-muted-foreground" aria-label="Topic" />
        <span className="min-w-0 flex-1">
          <span className="flex items-baseline justify-between gap-4">
            <span className="font-medium group-hover:underline">{node.title}</span>
            <Count path={node.path} />
          </span>
          {blurb && (
            <span className="mt-0.5 block max-w-4xl text-sm text-muted-foreground">
              <MathText text={blurb} />
            </span>
          )}
        </span>
      </Link>
    </li>
  )
}

/** A note, in reading order: its title and full summary; a stub is dimmed. */
export function NoteRow({ note }: { note: NoteMeta }) {
  const stub = note.status === 'stub'
  return (
    <li>
      <Link
        to={noteUrl(note.slug)}
        onMouseEnter={() => prefetchNote(note.slug)}
        onFocus={() => prefetchNote(note.slug)}
        className="group -mx-2 flex items-start gap-3 rounded-md px-2 py-3 transition-colors hover:bg-muted/50"
      >
        <FileText className="mt-0.5 size-4 shrink-0 text-muted-foreground" aria-label="Note" />
        <span className="min-w-0 flex-1">
          <span className="flex flex-wrap items-baseline gap-x-2">
            <span className={cn('font-medium group-hover:underline', stub && 'text-muted-foreground')}>
              {note.title}
            </span>
            {stub && <span className="text-xs text-muted-foreground italic">stub</span>}
          </span>
          <span className={cn('mt-0.5 block max-w-4xl text-sm text-muted-foreground', stub && 'opacity-70')}>
            <MathText text={note.summary} />
          </span>
        </span>
      </Link>
    </li>
  )
}

function SectionHeading({ children }: { children: string }) {
  return <h3 className="text-xs font-semibold tracking-wider text-muted-foreground uppercase">{children}</h3>
}

/** Notes under a topic, and how many are written while some are still stubs. */
function Count({ path }: { path: string }) {
  const all = notes.filter((n) => inCategory(n, path))
  const written = all.filter((n) => n.status !== 'stub').length
  return (
    <span className="shrink-0 text-xs font-normal tracking-normal text-muted-foreground normal-case tabular-nums">
      {written === all.length ? `${all.length} notes` : `${written} of ${all.length} written`}
    </span>
  )
}

/** A topic's overview: its index note when that note carries the topic's name. */
function overviewOf(node: CategoryNode): NoteMeta | undefined {
  const n = node.index ? notes.find((x) => x.slug === node.index) : undefined
  return n && n.title === node.title ? n : undefined
}
