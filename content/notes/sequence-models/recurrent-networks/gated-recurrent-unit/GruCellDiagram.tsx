import { Interactive } from '@/components/viz'
import { Diagram } from '@/components/diagram/Diagram'
import { gru } from '@/components/diagram/specs/recurrent'

/** One step of a GRU */
export function GruCellDiagram() {
  return (
    <Interactive
      title="One step of a GRU"
      caption="The state runs along the top. The update gate multiplies the old state directly, and its complement multiplies the candidate; the two products are added. The reset gate scales the old state before it enters the candidate. There is no separate cell state and no output gate."
    >
      <Diagram
        spec={gru}
        ariaLabel="GRU cell: reset gate scales the previous state before the candidate, update gate blends the previous state and the candidate"
      />
    </Interactive>
  )
}
