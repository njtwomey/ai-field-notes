import katex from 'katex'
import { useMemo } from 'react'
import { cn } from '@render/lib/utils'

/**
 * Maths set by KaTeX, for labels, readouts and descriptions: `<Tex>{String.raw`\sigma^2`}</Tex>`. Invalid input renders
 * in red rather than throwing. `display` sets it as a centred block.
 */
export function Tex({
  children,
  display = false,
  className,
}: {
  children: string
  display?: boolean
  className?: string
}) {
  const html = useMemo(
    () => katex.renderToString(children, { displayMode: display, throwOnError: false, output: 'html' }),
    [children, display],
  )
  const Tag = display ? 'div' : 'span'
  return <Tag className={cn('font-prose', className)} dangerouslySetInnerHTML={{ __html: html }} />
}
