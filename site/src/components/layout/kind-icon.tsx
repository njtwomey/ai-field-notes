import {
  BookOpen,
  ChartArea,
  FlaskConical,
  Layers,
  Lightbulb,
  NotebookPen,
  Wrench,
  type LucideIcon,
} from 'lucide-react'
import type { NoteKind } from '@/lib/content'

export const kindIcons: Record<NoteKind, LucideIcon> = {
  concept: Lightbulb,
  distribution: ChartArea,
  technique: Wrench,
  test: FlaskConical,
  example: NotebookPen,
  'case-study': BookOpen,
  overview: Layers,
}
