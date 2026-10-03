import { MathText } from 'aifn-render'
import { Diagram } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'
import { Interactive } from 'aifn-render'

/** A graphical model in plate notation, framed like every other figure; the caption may contain `$…$` maths. */
export function Plate({ spec, title, caption }: { spec: DiagramSpec; title: string; caption: string }) {
  return (
    <Interactive title={title} caption={<MathText text={caption} />}>
      <Diagram spec={spec} ariaLabel={title} />
    </Interactive>
  )
}
