/**
 * Components available in every note without an import. Notes import only note-local widgets and
 * '@/components/viz' / '@/components/widgets' pieces.
 */
import { isValidElement, type ComponentProps, type ReactElement, type ReactNode } from 'react'
import { CodeBlock } from '@/components/code/CodeBlock'
import { Interactive, Readout } from '@/components/viz'
import { Asset } from './Asset'
import { Callout, Definition } from './Callout'
import { Cite } from './Cite'
import { Derivation } from './Derivation'
import { NoteLink } from './NoteLink'
import { SpecTable } from './SpecTable'

function Pre({ children }: { children?: ReactNode }) {
  const code = isValidElement(children) ? (children as ReactElement<{ className?: string; children?: string }>) : null
  const lang = code?.props.className?.replace('language-', '')
  return <CodeBlock code={String(code?.props.children ?? '')} lang={lang} />
}

function Table(props: ComponentProps<'table'>) {
  return (
    <div className="my-6 overflow-x-auto">
      <table {...props} className="my-0" />
    </div>
  )
}

export const mdxComponents = {
  pre: Pre,
  table: Table,
  Asset,
  Callout,
  Cite,
  Definition,
  Derivation,
  Interactive,
  NoteLink,
  Readout,
  SpecTable,
}
