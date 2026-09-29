import { slug } from 'github-slugger'
import { Component, type ReactNode } from 'react'
import { AnchorTitle } from '@/components/content/Anchored'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { cn } from '@/lib/utils'

export type InteractiveProps = {
  title: string
  /** One or two sentences: what to change and what to watch. */
  caption?: ReactNode
  /** Parameter controls, laid out in a responsive grid above the figure. */
  controls?: ReactNode
  /** Small readouts (e.g. current loss) shown under the figure. */
  readout?: ReactNode
  children: ReactNode
  className?: string
}

/** The standard frame for every interactive figure: title, controls, figure, readout, caption. */
export function Interactive({ title, caption, controls, readout, children, className }: InteractiveProps) {
  // The title's slug is the figure's anchor, so a link can point at one figure: /n/<note>#<figure-title>.
  const id = slug(title)
  return (
    <Card id={id} className={cn('not-prose my-8 gap-4', className)}>
      <CardHeader>
        <CardTitle className="text-sm font-medium">
          <AnchorTitle id={id}>{title}</AnchorTitle>
        </CardTitle>
        {caption && <CardDescription className="text-xs">{caption}</CardDescription>}
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {controls && <div className="grid grid-cols-1 gap-x-6 gap-y-4 sm:grid-cols-2 lg:grid-cols-3">{controls}</div>}
        <FigureBoundary>{children}</FigureBoundary>
        {readout && <div className="flex flex-wrap gap-x-6 gap-y-1 text-xs text-muted-foreground">{readout}</div>}
      </CardContent>
    </Card>
  )
}

export function Readout({ label, value }: { label: ReactNode; value: ReactNode }) {
  return (
    <span>
      {label} <span className="font-mono text-foreground tabular-nums">{value}</span>
    </span>
  )
}

/**
 * Contains a crash inside one figure, so the rest of the note still renders. The message names the error; a build does
 * not catch these, so the headless render check in the dev workflow is what finds them.
 */
class FigureBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null }
  static getDerivedStateFromError(error: Error) {
    return { error }
  }
  render() {
    if (!this.state.error) return this.props.children
    return (
      <div
        role="alert"
        className="rounded-md border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive"
      >
        This figure failed to render: {this.state.error.message}
      </div>
    )
  }
}
