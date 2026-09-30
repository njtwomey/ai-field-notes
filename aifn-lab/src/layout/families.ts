/**
 * The sidebar's top level: aifn's module families (docs/aifn-plan.md §3), each listing its modules in reading order.
 * A new module slots in by adding its name here; one missing from the map is listed under "Other", with a console
 * warning in dev and a `make lab-check` warning.
 */
import { Activity, Brain, Database, Dices, FlaskConical, Radio, Shapes, Sigma, type LucideIcon } from 'lucide-react'

export type Family = { id: string; title: string; icon: LucideIcon; modules: readonly string[] }

/** The family of the lab's own pages (UI kit, diagrams), which sit directly under it rather than under a module. */
export const LAB_FAMILY = 'lab'
/** The family of modules missing from the map. */
export const OTHER_FAMILY = 'other'

export const FAMILIES: readonly Family[] = [
  {
    id: 'numerics',
    title: 'Numerics',
    icon: Sigma,
    modules: ['special', 'tensor', 'linalg', 'graph', 'autodiff', 'optim', 'programming', 'solve', 'quadrature'],
  },
  {
    id: 'probability',
    title: 'Probability',
    icon: Dices,
    modules: ['random', 'distributions', 'stats', 'info', 'pgm', 'ep', 'mcmc', 'vi', 'gp', 'kernels'],
  },
  {
    id: 'dynamics',
    title: 'Dynamics',
    icon: Activity,
    modules: ['trace', 'ode', 'fields', 'pde', 'maps', 'control', 'sde', 'diffusion', 'timeseries'],
  },
  {
    id: 'learning',
    title: 'Learning',
    icon: Brain,
    modules: [
      'estimators',
      'preprocess',
      'compose',
      'validate',
      'glm',
      'smooth',
      'gam',
      'classify',
      'cluster',
      'embed',
      'nn',
      'losses',
      'metrics',
    ],
  },
  { id: 'signals', title: 'Signals and decisions', icon: Radio, modules: ['dsp', 'bandits', 'rl', 'ot'] },
  { id: 'support', title: 'Support', icon: Database, modules: ['datasets', 'geometry'] },
  { id: OTHER_FAMILY, title: 'Other', icon: Shapes, modules: [] },
  { id: LAB_FAMILY, title: 'Lab', icon: FlaskConical, modules: [] },
]

const byModule = new Map(FAMILIES.flatMap((f) => f.modules.map((m) => [m, f] as const)))

/** The family a module belongs to: its mapped family, or "Other". */
export function familyOf(module: string): Family {
  return byModule.get(module) ?? FAMILIES.find((f) => f.id === OTHER_FAMILY)!
}

/** The position of a module within its family (unmapped modules sort last, alphabetically). */
export function moduleRank(module: string): number {
  const f = byModule.get(module)
  return f ? f.modules.indexOf(module) : Number.MAX_SAFE_INTEGER
}

/** The modules, of those given, that the map does not list. */
export function unmappedModules(modules: Iterable<string>): string[] {
  return [...new Set(modules)].filter((m) => !byModule.has(m)).sort()
}
