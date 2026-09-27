import { MathText } from '@/components/content/MathText'
import { Diagram } from '@/components/diagram/Diagram'
import type { DiagramSpec } from '@/components/diagram/types'
import { Interactive } from '@/components/viz'

/** A graphical model in plate notation, framed like every other figure; the caption may contain `$…$` maths. */
export function Plate({ spec, title, caption }: { spec: DiagramSpec; title: string; caption: string }) {
  return (
    <Interactive title={title} caption={<MathText text={caption} />}>
      <Diagram spec={spec} ariaLabel={title} />
    </Interactive>
  )
}
