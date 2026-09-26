import katex from 'katex'
import { Fragment } from 'react'
import { macros } from '@content/macros'
import { INLINE_MATH } from '@/lib/math-text'

const cache = new Map<string, string>()

function render(tex: string): string {
  let html = cache.get(tex)
  if (html === undefined) {
    // A fresh copy of the macros per formula: KaTeX may add definitions to the object it is given.
    html = katex.renderToString(tex, { macros: { ...macros }, throwOnError: false, strict: 'ignore' })
    cache.set(tex, html)
  }
  return html
}

/** Renders a string whose `$…$` spans are KaTeX maths, with the site's macros. Everything else is plain text. */
export function MathText({ text }: { text: string }) {
  const parts = text.split(INLINE_MATH)
  // split() with one capture group alternates text, maths, text, …
  return (
    <>
      {parts.map((part, i) =>
        i % 2 === 0 ? (
          <Fragment key={i}>{part}</Fragment>
        ) : (
          <span key={i} dangerouslySetInnerHTML={{ __html: render(part) }} />
        ),
      )}
    </>
  )
}
