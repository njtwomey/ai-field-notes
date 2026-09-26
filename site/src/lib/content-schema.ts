/**
 * Schemas for content/ (note frontmatter, references.yaml, taxonomy.yaml) and the index types derived from them.
 * Isomorphic: imported by the Vite plugin (plugins/content-index.ts) for validation and by the app for types.
 */
import { z } from 'zod'

export const slug = z.string().regex(/^[a-z0-9]+(?:-[a-z0-9]+)*$/, 'must be kebab-case')
export const categoryPath = z.string().regex(/^[a-z0-9-]+(?:\/[a-z0-9-]+)*$/, 'must be a slash-separated path of slugs')

export const noteKinds = ['concept', 'distribution', 'technique', 'test', 'case-study', 'overview'] as const

export const frontmatterSchema = z
  .object({
    title: z.string().min(1),
    kind: z.enum(noteKinds),
    category: categoryPath,
    summary: z.string().min(1).max(280),
    tags: z.array(slug).default([]),
    aliases: z.array(z.string()).default([]),
    requires: z.array(slug).default([]),
    partOf: z.array(slug).default([]),
    related: z.array(slug).default([]),
    code: slug.optional(),
    references: z.array(z.string()).default([]),
    status: z.enum(['stub', 'draft', 'stable']).default('draft'),
    updated: z.union([z.string(), z.date()]).transform((d) => (d instanceof Date ? d.toISOString().slice(0, 10) : d)),
  })
  .strict()

export const referenceSchema = z
  .object({
    type: z.enum(['paper', 'blog', 'book', 'video', 'docs', 'code', 'course']),
    title: z.string(),
    authors: z.array(z.string()).default([]),
    year: z.number().int().optional(),
    venue: z.string().optional(),
    url: z.url(),
    note: z.string().optional(),
  })
  .strict()

/** Icons a top-level category may use. Mapped to lucide components in site/src/components/layout/category-icon.tsx. */
export const categoryIcons = [
  'sigma',
  'target',
  'shapes',
  'chart-spline',
  'bell',
  'network',
  'cpu',
  'book-open',
  'brain',
  'gauge',
  'dices',
  'workflow',
  'microscope',
] as const
export type CategoryIcon = (typeof categoryIcons)[number]

export type CategoryInput = {
  id: string
  title: string
  description?: string
  icon?: CategoryIcon
  children?: CategoryInput[]
}
export const categorySchema: z.ZodType<CategoryInput> = z.lazy(() =>
  z
    .object({
      id: slug,
      title: z.string(),
      description: z.string().optional(),
      icon: z.enum(categoryIcons).optional(),
      children: z.array(categorySchema).optional(),
    })
    .strict(),
)

export type Frontmatter = z.infer<typeof frontmatterSchema>

export type NoteMeta = Frontmatter & {
  slug: string
  /** Path relative to content/, e.g. `notes/linear-regression/index.mdx`. */
  file: string
  headings: { depth: number; text: string; id: string }[]
  /** Reference keys cited inline with <Cite>, in first-use order. */
  cited: string[]
  /** Slugs linked inline with <NoteLink>. */
  linked: string[]
  wordCount: number
}

export type CategoryNode = {
  path: string
  title: string
  description?: string
  icon?: CategoryIcon
  children: CategoryNode[]
}

export type NoteKind = (typeof noteKinds)[number]
export type Reference = z.infer<typeof referenceSchema>
