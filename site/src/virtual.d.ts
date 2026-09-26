declare module 'virtual:content' {
  import type { CategoryNode, NoteMeta, Reference } from '@/lib/content-schema'
  export const notes: NoteMeta[]
  export const references: Record<string, Reference>
  export const taxonomy: CategoryNode[]
}

declare module 'virtual:search' {
  const bodies: Record<string, string>
  export default bodies
}
