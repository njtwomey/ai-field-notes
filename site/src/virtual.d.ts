declare module 'virtual:content' {
  import type { CategoryNode, NoteMeta, Reference } from '@/lib/content-schema'
  export const notes: NoteMeta[]
  export const references: Record<string, Reference>
  export const taxonomy: CategoryNode[]
  /** Content errors. Always empty in a build (which fails instead); listed in a banner by the dev server. */
  export const contentErrors: string[]
}
