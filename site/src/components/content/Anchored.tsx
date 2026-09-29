import { Check, Link2 } from 'lucide-react'
import { useState, type ComponentProps, type MouseEvent, type ReactNode } from 'react'
import { useNavigate } from 'react-router'
import { cn } from '@/lib/utils'

/**
 * A heading or figure title that is its own permalink. Clicking it puts `#id` in the URL (so the address bar holds a
 * link to this spot) and copies that link; hovering shows a link icon so the reader can tell the title is clickable.
 */
export function AnchorTitle({ id, children, className }: { id: string; children: ReactNode; className?: string }) {
  const navigate = useNavigate()
  const [copied, setCopied] = useState(false)
  const onClick = (e: MouseEvent) => {
    // Let modified clicks (new tab, copy link address) behave as a plain link.
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return
    e.preventDefault()
    navigate({ hash: id }, { preventScrollReset: true })
    document.getElementById(id)?.scrollIntoView({ block: 'start' })
    navigator.clipboard?.writeText(`${window.location.origin}${window.location.pathname}#${id}`).then(
      () => {
        setCopied(true)
        setTimeout(() => setCopied(false), 1500)
      },
      () => {},
    )
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

type HeadingProps = ComponentProps<'h2'>

/** MDX headings whose text links to themselves; ids come from rehype-slug. */
export function H2({ id, children, ...props }: HeadingProps) {
  return (
    <h2 id={id} {...props}>
      {id ? <AnchorTitle id={id}>{children}</AnchorTitle> : children}
    </h2>
  )
}

export function H3({ id, children, ...props }: ComponentProps<'h3'>) {
  return (
    <h3 id={id} {...props}>
      {id ? <AnchorTitle id={id}>{children}</AnchorTitle> : children}
    </h3>
  )
}

export function H4({ id, children, ...props }: ComponentProps<'h4'>) {
  return (
    <h4 id={id} {...props}>
      {id ? <AnchorTitle id={id}>{children}</AnchorTitle> : children}
    </h4>
  )
}
