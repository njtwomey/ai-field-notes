/**
 * Components available in every note without an import. Notes import only note-local widgets and
 * '@/components/viz' / '@/components/widgets' pieces.
 */
import { Readout } from 'aifn-render'
import { DistributionExplorer } from '@/components/widgets/DistributionExplorer'
import { Asset } from './Asset'
import { H2, H3, H4 } from './Anchored'
import { Pre, Table } from './MdxElements'
import { Callout, Definition } from './Callout'
import { Cite } from './Cite'
import { Derivation } from './Derivation'
import { Gloss } from './Gloss'
import { NoteLink } from './NoteLink'
import { SpecTable } from './SpecTable'

export const mdxComponents = {
  h2: H2,
  h3: H3,
  h4: H4,
  pre: Pre,
  table: Table,
  Asset,
  Callout,
  Cite,
  Definition,
  Derivation,
  DistributionExplorer,
  Gloss,
  NoteLink,
  Readout,
  SpecTable,
}
