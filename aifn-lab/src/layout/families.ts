/**
 * The sidebar's top level, read from the module tree in `aifn-js/modules.json`: one family per core family (in tier
 * order), then one per application area (in DAG order). Each lists its nodes in tree order: the family or area itself
 * (for specimens that span it), then its groups and modules. A specimen's `module` is a node path: `numerics/linalg`
 * for core, `applied/learning/generalised/glm` for applications. A path not in the tree is listed under "Other", with
 * a console warning in dev and a `make lab-check` warning. Nothing here is kept by hand except the icons.
 */
import {
  Activity,
  Binary,
  Boxes,
  Brain,
  Cable,
  Database,
  Dices,
  Eye,
  FlaskConical,
  Gauge,
  GitFork,
  GraduationCap,
  Layers,
  Network,
  Orbit,
  Package,
  Radio,
  Search,
  Shapes,
  Sigma,
  Sparkles,
  Target,
  TrendingUp,
  Truck,
  Waves,
  type LucideIcon,
} from 'lucide-react'
import spec from 'aifn-js/modules.json' with { type: 'json' }

export type Family = { id: string; title: string; icon: LucideIcon; modules: readonly string[] }

/** The family of the lab's own pages (UI kit, diagrams), which sit directly under it rather than under a module. */
export const LAB_FAMILY = 'lab'
/** The family of modules missing from the tree. */
export const OTHER_FAMILY = 'other'

/** An icon per core family and application area (presentation, so it lives here rather than in modules.json). */
const ICONS: Record<string, LucideIcon> = {
  foundation: Layers,
  numerics: Sigma,
  graph: GitFork,
  probability: Dices,
  optim: Target,
  systems: Cable,
  inference: Network,
  dynamics: Activity,
  signal: Radio,
  transport: Truck,
  learning: Brain,
  nn: Brain,
  'applied/learning': GraduationCap,
  'applied/unsupervised': Boxes,
  'applied/inference': Network,
  'applied/timeseries': TrendingUp,
  'applied/signals': Waves,
  'applied/vision': Eye,
  'applied/dynamics': Orbit,
  'applied/decisions': Target,
  'applied/generative': Sparkles,
  'applied/neural': Brain,
  'applied/retrieval': Search,
  'applied/evaluation': Gauge,
  'applied/information': Binary,
  'applied/algorithms': Binary,
  'applied/data': Database,
}

type AppNode = { module: string; status?: string } | { group: string; children: AppNode[] }

/** The node paths under an application node, in tree order (groups before their children; gaps left out). */
const appPaths = (prefix: string, list: readonly AppNode[]): string[] =>
  list.flatMap((n) =>
    'module' in n
      ? n.status === 'gap'
        ? []
        : [`${prefix}/${n.module}`]
      : [`${prefix}/${n.group}`, ...appPaths(`${prefix}/${n.group}`, n.children)],
  )

const CORE_FAMILIES: readonly Family[] = spec.core.families.map((f) => ({
  id: f.family,
  title: f.title,
  icon: ICONS[f.family] ?? Package,
  modules: [
    f.family,
    ...(f.modules as { module: string; status?: string }[])
      .filter((m) => m.status !== 'gap')
      .map((m) => `${f.family}/${m.module}`),
  ],
}))

const APPLICATION_FAMILIES: readonly Family[] = spec.applications.areas.map((a) => ({
  id: `applied/${a.area}`,
  title: `Applied: ${a.title.toLowerCase()}`,
  icon: ICONS[`applied/${a.area}`] ?? Package,
  modules: [`applied/${a.area}`, ...appPaths(`applied/${a.area}`, a.children as AppNode[])],
}))

export const FAMILIES: readonly Family[] = [
  ...CORE_FAMILIES,
  ...APPLICATION_FAMILIES,
  { id: OTHER_FAMILY, title: 'Other', icon: Shapes, modules: [] },
  { id: LAB_FAMILY, title: 'Lab', icon: FlaskConical, modules: [] },
]

const byModule = new Map(FAMILIES.flatMap((f) => f.modules.map((m) => [m, f] as const)))

/** The family a node path belongs to, or "Other". */
export function familyOf(module: string): Family {
  return byModule.get(module) ?? FAMILIES.find((f) => f.id === OTHER_FAMILY)!
}

/** The position of a node within its family (unmapped nodes sort last, alphabetically). */
export function moduleRank(module: string): number {
  const f = byModule.get(module)
  return f ? f.modules.indexOf(module) : Number.MAX_SAFE_INTEGER
}

/** The node paths, of those given, that the tree does not list. */
export function unmappedModules(modules: Iterable<string>): string[] {
  return [...new Set(modules)].filter((m) => !byModule.has(m)).sort()
}

/** A node's label within its family: its path below the family (`linalg`), or `(all)` for the family itself. */
export function moduleLabel(module: string): string {
  const f = byModule.get(module)
  if (!f) return module
  return module === f.id ? '(all)' : module.slice(f.id.length + 1)
}

/** The import path of a node: `aifn/<path>` for core, `aifn-applied/<path>` for an application node. */
export function importPath(module: string): string {
  return module.startsWith('applied/') ? `aifn-applied/${module.slice('applied/'.length)}` : `aifn/${module}`
}
