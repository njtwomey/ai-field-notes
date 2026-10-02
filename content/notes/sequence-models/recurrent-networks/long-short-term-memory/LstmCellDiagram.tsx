import { Interactive } from 'aifn-render'
import { Diagram } from '@/components/diagram/Diagram'
import { lstm } from '@/components/diagram/specs/recurrent'

/** One step of an LSTM */
export function LstmCellDiagram() {
  return (
    <Interactive
      title="One step of an LSTM"
      caption="The cell state runs along the top and meets only an elementwise product with the forget gate and an addition. The forget, input and output gates (sigmoid) and the candidate (tanh) all read the previous hidden state and the current input from the bus below. The new hidden state is the output gate times the squashed cell state."
    >
      <Diagram
        spec={lstm}
        ariaLabel="LSTM cell: cell state along the top, forget, input, candidate and output gates fed by the previous hidden state and the input"
      />
    </Interactive>
  )
}
