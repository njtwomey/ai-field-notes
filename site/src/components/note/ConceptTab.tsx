import { MDXProvider } from '@mdx-js/react'
import { createElement, lazy, Suspense, useEffect, useMemo, type ComponentType } from 'react'
import { useLocation } from 'react-router'
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
        <Suspense fallback={<p className="text-muted-foreground">Loading…</p>}>
          {createElement(Body)}
          <ScrollToHash />
        </Suspense>
      </MDXProvider>
      <ReferenceList note={note} />
    </article>
  )
}

/**
 * The body loads lazily, so on a reload or a shared link the browser looks for `#section` before it exists. Mounted
 * inside the same Suspense boundary as the body, this runs once the body is in the DOM and scrolls to the heading.
 * Figures, maths and fonts keep changing the layout for a moment, so it re-aligns on every resize for two seconds, and
 * stops as soon as the reader scrolls, clicks or presses a key.
 */
function ScrollToHash() {
  const { hash } = useLocation()
  useEffect(() => {
    const id = decodeURIComponent(hash.slice(1))
    if (!id) return
    const align = () => document.getElementById(id)?.scrollIntoView({ block: 'start' })
    align()
    const observer = new ResizeObserver(align)
    observer.observe(document.body)
    const stop = () => {
      observer.disconnect()
      clearTimeout(timer)
      for (const e of events) window.removeEventListener(e, stop)
    }
    const events = ['wheel', 'touchstart', 'keydown', 'pointerdown'] as const
    for (const e of events) window.addEventListener(e, stop, { passive: true })
    const timer = setTimeout(stop, 2000)
    return stop
  }, [hash])
  return null
}
