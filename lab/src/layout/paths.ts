/** Lab URLs: `/<module>/<specimen-slug>` for a specimen, `/<page>` for a lab page, `#<figure-id>` for a figure. */
import { slug } from 'github-slugger'

/** The site's slug (github-slugger), so a figure's anchor matches the one its note figure would get. */
export const slugify = (text: string) => slug(text)

/** A specimen's path, without the leading slash. */
export function specimenPath(s: { module: string; title: string }): string {
  return `${s.module}/${slugify(s.title)}`
}

/** The href of a page, from its key (its path without the leading slash). */
export const hrefOf = (key: string) => `/${encodeURIComponent(key).replace(/%2F/g, '/')}`

/** The entry named by the URL path. */
export function pathKey(): string {
  return decodeURIComponent(location.pathname.replace(/^\/+|\/+$/g, ''))
}

/** Rewrites an old hash route (`#/module/slug`) to its path, keeping it as the current history entry. */
export function redirectHashRoute(): void {
  if (!location.hash.startsWith('#/')) return
  history.replaceState(null, '', `/${location.hash.slice(2)}`)
}

/** Scrolls the figure named by the URL hash into view and highlights it briefly. */
export function revealHash(): void {
  const id = decodeURIComponent(location.hash.slice(1))
  const el = id ? document.getElementById(id) : null
  if (!el) return
  el.scrollIntoView({ block: 'start', behavior: 'smooth' })
  el.setAttribute('data-highlight', '')
  setTimeout(() => el.removeAttribute('data-highlight'), 1600)
}
