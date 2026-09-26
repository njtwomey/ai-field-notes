import { ChevronRight } from 'lucide-react'
import type { ReactNode } from 'react'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'

/**
 * Optional depth, collapsed by default: derivations, proofs, edge cases. The note must read completely without it.
 */
export function Derivation({ title = 'Derivation', children }: { title?: string; children: ReactNode }) {
  return (
    <Collapsible className="not-prose my-6 rounded-lg border">
      <CollapsibleTrigger className="group flex w-full items-center gap-2 px-4 py-2.5 text-left text-sm font-medium">
        <ChevronRight className="size-4 transition-transform group-data-panel-open:rotate-90" aria-hidden />
        {title}
      </CollapsibleTrigger>
      <CollapsibleContent className="note-prose prose max-w-none border-t px-5 py-4 text-[1.0625rem] dark:prose-invert">
        {children}
      </CollapsibleContent>
    </Collapsible>
  )
}
