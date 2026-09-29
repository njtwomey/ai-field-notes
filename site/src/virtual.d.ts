declare module 'virtual:content' {
  import type { CategoryNode, GlossaryEntry, NoteMeta, Reference, TopicGroup } from '@/lib/content-schema'
  export const notes: NoteMeta[]
  export const references: Record<string, Reference>
  /** content/glossary.yaml, keyed by entry key. */
  export const glossary: Record<string, GlossaryEntry>
  export const taxonomy: CategoryNode[]
  /** Home-page groups of the top-level topics, from content/groups.yaml. */
  export const groups: TopicGroup[]
  /** Content errors. Always empty in a build (which fails instead); listed in a banner by the dev server. */
  export const contentErrors: string[]
  /** Site totals, computed at build time. */
  export const stats: {
    notes: number
    topics: number
    categories: number
    glossary: number
    references: number
    tags: number
    runnable: number
    workedExamples: number
  }
}

declare module 'virtual:search' {
  /** Plain-text note bodies keyed by slug, for the full-text search index. */
  const bodies: Record<string, string>
  export default bodies
}
