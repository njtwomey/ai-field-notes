/**
 * Components available in every note without an import. Notes import only note-local widgets and
 * '@/components/viz' / '@/components/widgets' pieces.
 */
import { Interactive, Readout } from '@/components/viz'
import { DistributionExplorer } from '@/components/widgets/DistributionExplorer'
import { Asset } from './Asset'
import { Pre, Table } from './MdxElements'
import { Callout, Definition } from './Callout'
import { Cite } from './Cite'
import { Derivation } from './Derivation'
import { Gloss } from './Gloss'
import { NoteLink } from './NoteLink'
import { SpecTable } from './SpecTable'

export const mdxComponents = {
  pre: Pre,
  table: Table,
  Asset,
  Callout,
  Cite,
  Definition,
  Derivation,
  DistributionExplorer,
  Gloss,
  Interactive,
  NoteLink,
  Readout,
  SpecTable,
}
