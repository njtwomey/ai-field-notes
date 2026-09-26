/** Inline maths in plain strings such as note summaries: text with `$…$` spans, as in note bodies. */
export const INLINE_MATH = /\$([^$]+)\$/g

/** The string as searchable plain text: maths delimiters and LaTeX commands removed. */
export function plainMath(text: string): string {
  return text
    .replace(INLINE_MATH, (_, tex: string) => tex.replace(/\\[a-zA-Z]+/g, ' ').replace(/[{}^_\\]/g, ''))
    .replace(/\s+/g, ' ')
}
