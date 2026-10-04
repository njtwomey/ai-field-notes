/**
 * The sidebar's top level, read from the master 8-Part / 42-Subject AI Field Notes taxonomy
 * in `taxonomy.json`: one family per Part, plus Lab and Other.
 */
import {
  Boxes,
  Brain,
  Database,
  FlaskConical,
  GraduationCap,
  Layers,
  Network,
  Package,
  Shapes,
  Sigma,
  Waves,
  type LucideIcon,
} from 'lucide-react'
import taxonomy from './taxonomy.json' with { type: 'json' }

export type TopicSpec = {
  id: string
  title: string
  items: readonly string[]
}

export type SubjectSpec = {
  id: string
  title: string
  topics: readonly TopicSpec[]
}

export type PartSpec = {
  id: string
  num: number
  roman: string
  title: string
  subjects: readonly SubjectSpec[]
}

export const TAXONOMY: readonly PartSpec[] = taxonomy as readonly PartSpec[]

export type Family = { id: string; title: string; icon: LucideIcon; modules: readonly string[] }

/** The family of the lab's own pages (UI kit, diagrams), which sit directly under it rather than under a module. */
export const LAB_FAMILY = 'lab'
/** The family of modules missing from the tree. */
export const OTHER_FAMILY = 'other'

/** An icon per taxonomy Part (presentation, so it lives here rather than in taxonomy.json). */
const PART_ICONS: Record<string, LucideIcon> = {
  'part-1-mathematical-foundations': Sigma,
  'part-2-signals-systems-and-information': Waves,
  'part-3-data': Database,
  'part-4-principles-of-learning': GraduationCap,
  'part-5-learning-paradigms': Brain,
  'part-6-neural-architectures': Network,
  'part-7-application-domains': Layers,
  'part-8-practice': Boxes,
  [LAB_FAMILY]: FlaskConical,
  [OTHER_FAMILY]: Shapes,
}

function collectModules(part: PartSpec): string[] {
  const list: string[] = [part.id]
  for (const subj of part.subjects) {
    list.push(`${part.id}/${subj.id}`)
    for (const topic of subj.topics) {
      if (topic.id && topic.id !== 'overview') {
        list.push(`${part.id}/${subj.id}/${topic.id}`)
      }
    }
  }
  return list
}

const PART_FAMILIES: readonly Family[] = TAXONOMY.map((part) => ({
  id: part.id,
  title: `${part.roman}: ${part.title}`,
  icon: PART_ICONS[part.id] ?? Package,
  modules: collectModules(part),
}))

export const FAMILIES: readonly Family[] = [
  ...PART_FAMILIES,
  { id: LAB_FAMILY, title: 'Lab', icon: FlaskConical, modules: [] },
  { id: OTHER_FAMILY, title: 'Other', icon: Shapes, modules: [] },
]

const familyById = new Map(FAMILIES.map((f) => [f.id, f]))
const byModule = new Map(FAMILIES.flatMap((f) => f.modules.map((m) => [m, f] as const)))

/** The family a node path belongs to, or "Other". */
export function familyOf(module: string): Family {
  if (!module) return FAMILIES.find((f) => f.id === LAB_FAMILY)!
  const direct = byModule.get(module)
  if (direct) return direct
  const root = module.split('/')[0]
  return familyById.get(root) ?? FAMILIES.find((f) => f.id === OTHER_FAMILY)!
}

/** The position of a node within its family (unmapped nodes sort last, alphabetically). */
export function moduleRank(module: string): number {
  const f = familyOf(module)
  if (!f) return Number.MAX_SAFE_INTEGER
  const idx = f.modules.indexOf(module)
  return idx >= 0 ? idx : Number.MAX_SAFE_INTEGER
}

/** The node paths, of those given, that the tree does not list. */
export function unmappedModules(modules: Iterable<string>): string[] {
  return [...new Set(modules)]
    .filter((m) => {
      const f = familyOf(m)
      return f.id === OTHER_FAMILY
    })
    .sort()
}

/** A node's label within its family: e.g. "Linear algebra" or "Mathematics · Linear algebra". */
export function moduleLabel(module: string): string {
  const f = familyOf(module)
  if (!f || f.id === OTHER_FAMILY || f.id === LAB_FAMILY) return module
  if (module === f.id) return '(all)'
  const rel = module.slice(f.id.length + 1)
  const parts = rel.split('/')
  return parts.map((p) => p.replace(/-/g, ' ')).join(' · ')
}

/** The breadcrumb / kicker for a module: e.g. "Part I · Mathematics · Linear algebra". */
export function importPath(module: string): string {
  const f = familyOf(module)
  if (!f || f.id === OTHER_FAMILY || f.id === LAB_FAMILY) return module || 'aifn lab'
  const part = TAXONOMY.find((p) => p.id === f.id)
  const prefix = part ? part.roman : f.title
  if (module === f.id) return prefix
  const rel = module.slice(f.id.length + 1)
  const segments = rel.split('/')
  const formatted = segments.map((seg) => seg.charAt(0).toUpperCase() + seg.slice(1).replace(/-/g, ' ')).join(' · ')
  return `${prefix} · ${formatted}`
}
