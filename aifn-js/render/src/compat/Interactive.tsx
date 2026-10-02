import { Check, Link2 } from 'lucide-react'
import { slug } from 'github-slugger'
import { Component, useState, type MouseEvent, type ReactNode } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@render/ui/card'
import { cn } from '@render/lib/utils'

export type InteractiveProps = {
  title: string
  caption?: ReactNode
  controls?: ReactNode
  readout?: ReactNode
  children: ReactNode
  className?: string
}

function AnchorTitle({ id, children, className }: { id: string; children: ReactNode; className?: string }) {
  const [copied, setCopied] = useState(false)
  const onClick = (e: MouseEvent) => {
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return
    e.preventDefault()
    if (typeof window !== 'undefined') {
      window.history.replaceState(null, '', `#${id}`)
      document.getElementById(id)?.scrollIntoView({ block: 'start' })
      navigator.clipboard?.writeText(`${window.location.origin}${window.location.pathname}#${id}`).then(
        () => {
          setCopied(true)
          setTimeout(() => setCopied(false), 1500)
        },
        () => {},
      )
    }
  }
  const Icon = copied ? Check : Link2
  return (
    <a
      href={`#${id}`}
      onClick={onClick}
      className={cn('group/anchor inline-flex items-baseline gap-1.5 text-inherit no-underline', className)}
      title={copied ? 'Link copied' : 'Link to this section (click to copy)'}
    >
      <span>{children}</span>
      <Icon
        className="size-[0.7em] shrink-0 self-center text-muted-foreground opacity-0 transition-opacity group-hover/anchor:opacity-100 group-focus-visible/anchor:opacity-100"
        aria-hidden
      />
    </a>
  )
}

/** The standard frame for every interactive figure: title, controls, figure, readout, caption. */
export function Interactive({ title, caption, controls, readout, children, className }: InteractiveProps) {
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
