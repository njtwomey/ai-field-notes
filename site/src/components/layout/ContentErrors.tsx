import { ChevronDown, TriangleAlert } from 'lucide-react'
import { useState } from 'react'
import { contentErrors } from '@/lib/content'

/**
 * Dev server only: content errors that a build would fail on. The site keeps running without the broken pieces, and
 * this banner lists what was dropped. A production build never shows it, because the build fails instead.
 */
export function ContentErrors() {
  const [open, setOpen] = useState(false)
  if (!import.meta.env.DEV || contentErrors.length === 0) return null
  return (
    <div className="border-b border-destructive/30 bg-destructive/10 px-4 py-2 text-sm lg:px-8">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 text-left text-destructive"
        aria-expanded={open}
      >
        <TriangleAlert className="size-4 shrink-0" aria-hidden />
        <span className="flex-1">
          {contentErrors.length} content error{contentErrors.length === 1 ? '' : 's'}. The build would fail; the dev
          server is showing everything else.
        </span>
        <ChevronDown className={`size-4 shrink-0 transition-transform ${open ? 'rotate-180' : ''}`} aria-hidden />
      </button>
      {open && (
        <ul className="mt-2 max-h-64 space-y-1 overflow-y-auto font-mono text-xs text-foreground">
          {contentErrors.map((e) => (
            <li key={e} className="break-all whitespace-pre-wrap">
              {e}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
