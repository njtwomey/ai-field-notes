import { BookOpen, FlaskConical, Layers, Lightbulb, Wrench, type LucideIcon } from 'lucide-react'
import type { NoteKind } from '@/lib/content'

export const kindIcons: Record<NoteKind, LucideIcon> = {
  concept: Lightbulb,
  technique: Wrench,
  test: FlaskConical,
  'case-study': BookOpen,
  overview: Layers,
}
