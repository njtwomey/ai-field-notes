/**
 * Schemas for content/ (note frontmatter, references.yaml, taxonomy.yaml) and the index types derived from them.
 * Isomorphic: imported by the Vite plugin (plugins/content-index.ts) for validation and by the app for types.
 */
import { z } from 'zod'

export const slug = z.string().regex(/^[a-z0-9]+(?:-[a-z0-9]+)*$/, 'must be kebab-case')
export const categoryPath = z.string().regex(/^[a-z0-9-]+(?:\/[a-z0-9-]+)*$/, 'must be a slash-separated path of slugs')

export const noteKinds = ['concept', 'distribution', 'technique', 'test', 'example', 'case-study', 'overview'] as const

export const frontmatterSchema = z
  .object({
    title: z.string().min(1),
    kind: z.enum(noteKinds),
    /** Only for notes directly under content/notes/; nested notes take their category from their folder path. */
    category: categoryPath.optional(),
    /**
     * Plain text with optional `$…$` inline maths. The 280-character limit applies to the text as read (see the content
     * plugin), so LaTeX source does not count against it; this bound only stops runaway source.
     */
    summary: z.string().min(1).max(600),
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

export const glossaryKinds = ['acronym', 'concept', 'proper-name'] as const

/**
 * One entry of content/glossary.yaml, which is keyed by slug. Notes use an entry with `<Gloss name="key" />` (or any
 * of its aliases); the first use in a note shows "long (short)", later uses the short form.
 */
export const glossaryEntrySchema = z
  .object({
    /** The abbreviation shown after the first use. Required for acronyms and proper names. */
    short: z.string().min(1).optional(),
    /** The full name in Title Case, with every nested acronym spelled out. */
    long: z.string().min(1),
    /** Only when appending "s" to the long form is wrong. */
    longPlural: z.string().min(1).optional(),
    kind: z.enum(glossaryKinds),
    /** Tells apart entries that share one short form, e.g. the two LDAs. */
    sense: z.string().min(1).optional(),
    /** Other slugs that resolve to this entry, e.g. the acronym itself. */
    aliases: z.array(slug).default([]),
    /** Taxonomy path(s), as in taxonomy.yaml. */
    category: z.union([categoryPath, z.array(categoryPath).min(1)]).transform((c) => (typeof c === 'string' ? [c] : c)),
    /** The note whose subject the entry is (the glossary links straight to it); empty or absent when there is none. */
    note: z
      .union([slug, z.literal('')])
      .optional()
      .transform((n) => n || undefined),
    /** Notes that discuss the entry without being about it, e.g. true positive → the confusion matrix. */
    see: z.array(slug).default([]),
    /** Set while a definition awaits checking. */
    review: z.boolean().optional(),
    /** One or two plain sentences; `$…$` maths with the site macros. */
    definition: z.string().min(1).max(600),
  })
  .strict()
  .refine((e) => e.kind === 'concept' || e.short !== undefined, {
    message: 'acronyms and proper names need a short form',
    path: ['short'],
  })

export type GlossaryEntry = z.output<typeof glossaryEntrySchema> & { key: string }
export type GlossaryKind = (typeof glossaryKinds)[number]

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
  'git-fork',
  'graduation-cap',
  'trending-down',
  'ruler',
  'radar',
  'activity',
  'audio-waveform',
  'gamepad',
  'languages',
  'eye',
  'sliders',
  'library',
  'flask',
  'shield',
  'blend',
  'tags',
  'repeat',
  'arrow-right-left',
  'grid',
  'layers',
  'message-square',
  'sparkles',
  'waves',
  'share',
  'fingerprint',
] as const
export type CategoryIcon = (typeof categoryIcons)[number]

export type CategoryInput = {
  id: string
  title: string
  description?: string
  icon?: CategoryIcon
  /** The category's index note: `true` for the note whose slug is the category id, or another slug in the category. */
  index?: boolean | string
  children?: CategoryInput[]
}
export const categorySchema: z.ZodType<CategoryInput> = z.lazy(() =>
  z
    .object({
      id: slug,
      title: z.string(),
      description: z.string().optional(),
      icon: z.enum(categoryIcons).optional(),
      index: z.union([z.boolean(), slug]).optional(),
      children: z.array(categorySchema).optional(),
    })
    .strict(),
)

export type Frontmatter = z.infer<typeof frontmatterSchema>

export type NoteMeta = Omit<Frontmatter, 'category'> & {
  /** Category path, from the note's folder (or legacy frontmatter). Always set. */
  category: string
  slug: string
  /** Path relative to content/, e.g. `notes/supervised-learning/regression/linear-regression/index.mdx`. */
  file: string
  headings: { depth: number; text: string; id: string }[]
  /** Reference keys cited inline with <Cite>, in first-use order. */
  cited: string[]
  /** Slugs linked inline with <NoteLink>. */
  linked: string[]
  /** Glossary keys used with <Gloss>, resolved from aliases, in first-use order. */
  glossed: string[]
  wordCount: number
}

export type CategoryNode = {
  path: string
  title: string
  description?: string
  icon?: CategoryIcon
  /** Slug of the note that introduces the category and is listed first; see `index` in taxonomy.yaml. */
  index?: string
  children: CategoryNode[]
}

/** A home-page group of top-level topics (content/groups.yaml). */
export type TopicGroup = { title: string; topics: string[] }

export type NoteKind = (typeof noteKinds)[number]
export type Reference = z.infer<typeof referenceSchema>
