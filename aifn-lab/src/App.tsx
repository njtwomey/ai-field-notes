import { Shell, type LabPage } from '@lab/layout'
import { DiagramsKit } from './kit/DiagramsKit'
import { plotKitPages } from './kit/plot/pages'
import { stateKitPages } from './kit/state/pages'
import { UiKit } from './kit/UiKit'
import type { Specimen } from './specimen'

// Every file under ./specimens (at `<family>/<module>.tsx`, mirroring the aifn module tree) exports
// `specimens: Specimen[]`; folders and files starting with `_` hold shared code.
const modules = import.meta.glob<{ specimens: Specimen[] }>(
  ['./specimens/**/*.tsx', '!./specimens/**/_*/**', '!./specimens/**/_*.tsx'],
  { eager: true },
)
const specimens: Specimen[] = Object.values(modules).flatMap((m) => m.specimens ?? [])

const pages: LabPage[] = [
  {
    key: 'ui-kit',
    title: 'UI kit',
    description:
      'Every v2 control, chart and frame in one place: sliders, fields and choices; line, scatter, bar and heatmap charts with zoom, pan, hover and handles; figure sizing and data export; the palette in both themes.',
    render: () => <UiKit />,
  },
  ...plotKitPages,
  ...stateKitPages,
  {
    key: 'diagrams',
    title: 'Diagrams',
    description:
      'The lab’s diagram system: hand-placed plate models and factor graphs, the automatic layered layout for graphs given as data, step states, notes, edge chips and arrows.',
    render: () => <DiagramsKit />,
  },
]

/** The lab: the shell over the lab's own pages and every module's specimens. */
export function App() {
  return <Shell specimens={specimens} pages={pages} />
}
