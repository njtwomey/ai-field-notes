import { MDXProvider } from '@mdx-js/react'
import { createElement, lazy, Suspense, useMemo, type ComponentType } from 'react'
import { mdxComponents } from '@/components/content/mdx-components'
import { loadNote, type NoteMeta } from '@/lib/content'
import { ReferenceList } from './ReferenceList'

const cache = new Map<string, ComponentType>()

function noteComponent(slug: string): ComponentType {
  let c = cache.get(slug)
  if (!c) {
    c = lazy(() => loadNote(slug))
    cache.set(slug, c)
  }
  return c
}

export function ConceptTab({ note }: { note: NoteMeta }) {
  const Body = useMemo(() => noteComponent(note.slug), [note.slug])
  return (
    <article className="note-prose prose max-w-none prose-neutral dark:prose-invert">
      <MDXProvider components={mdxComponents}>
        <Suspense fallback={<p className="text-muted-foreground">Loading…</p>}>{createElement(Body)}</Suspense>
      </MDXProvider>
      <ReferenceList note={note} />
    </article>
  )
}
