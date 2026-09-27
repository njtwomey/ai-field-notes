import { isValidElement, type ComponentProps, type ReactElement, type ReactNode } from 'react'
import { CodeBlock } from '@/components/code/CodeBlock'

/** Fenced code blocks in MDX, rendered with syntax highlighting. */
export function Pre({ children }: { children?: ReactNode }) {
  const code = isValidElement(children) ? (children as ReactElement<{ className?: string; children?: string }>) : null
  const lang = code?.props.className?.replace('language-', '')
  return <CodeBlock code={String(code?.props.children ?? '')} lang={lang} />
}

/** GitHub-flavoured tables, scrolling sideways when wider than the column. */
export function Table(props: ComponentProps<'table'>) {
  return (
    <div className="my-6 overflow-x-auto">
      <table {...props} className="my-0" />
    </div>
  )
}
