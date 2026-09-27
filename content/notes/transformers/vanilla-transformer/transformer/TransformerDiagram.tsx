import { Interactive } from '@/components/viz'
import { Diagram } from '@/components/diagram/Diagram'
import { transformer } from '@/components/diagram/specs/transformer'

/** The encoder–decoder transformer */
export function TransformerDiagram() {
  return (
    <Interactive
      title="The encoder–decoder transformer"
      caption="Left, one of the N encoder blocks: self-attention and a feed-forward network, each wrapped by a residual connection and a layer norm (post-norm). Right, one of the N decoder blocks, which adds masked self-attention over the target prefix and cross-attention whose keys and values come from the final encoder output. A linear layer and a softmax turn the last decoder state into next-token probabilities."
    >
      <Diagram
        spec={transformer}
        ariaLabel="Transformer: encoder stack of self-attention and feed-forward sublayers, decoder stack of masked self-attention, cross-attention and feed-forward sublayers, each followed by add and norm"
      />
    </Interactive>
  )
}
