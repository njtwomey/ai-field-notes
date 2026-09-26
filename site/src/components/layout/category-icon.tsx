import {
  BookOpen,
  Brain,
  ChartArea,
  ChartSpline,
  Cpu,
  Dices,
  Folder,
  Gauge,
  Microscope,
  Network,
  Shapes,
  Sigma,
  Target,
  Workflow,
  type LucideIcon,
} from 'lucide-react'
import type { CategoryIcon } from '@/lib/content-schema'

const icons: Record<CategoryIcon, LucideIcon> = {
  sigma: Sigma,
  target: Target,
  shapes: Shapes,
  'chart-spline': ChartSpline,
  bell: ChartArea,
  network: Network,
  cpu: Cpu,
  'book-open': BookOpen,
  brain: Brain,
  gauge: Gauge,
  dices: Dices,
  workflow: Workflow,
  microscope: Microscope,
}

export function categoryIcon(icon: CategoryIcon | undefined): LucideIcon {
  return icon ? icons[icon] : Folder
}
