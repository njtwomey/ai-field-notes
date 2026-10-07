import { Diagram, Figure } from 'aifn-render'
import type { DiagramNode, DiagramSpec } from 'aifn-render'

const box = (id: string, x: number, y: number, label: string, tone: DiagramNode['tone']): DiagramNode => ({
  id,
  x,
  y,
  w: 3.8,
  h: 1.3,
  label,
  tone,
})

const spec: DiagramSpec = {
  unit: 40,
  nodes: [
    box('req', 0, 0, 'request JSON\nstate + typed questions', 1),
    box('fields', 4.6, 0, 'fields with codes\nA, B, C … per question', 'neutral'),
    box('prompts', 9.2, 0, 'one prompt per question\nchat template', 'neutral'),
    box('prefix', 13.8, 0, 'shared prefix\ninstructions, state, schema', 0),
    box('endings', 18.4, 0, 'one ending per question\nRequested field: "team"', 0),
    box('logits', 18.4, 3.4, 'code logits $\\zvec_{\\Ccal}$\nat the answer boundary', 0),
    box('probs', 11.5, 3.4, 'probabilities\n$\\operatorname{softmax}(\\zvec_{\\Ccal} / T)$', 'neutral'),
    box('resp', 4.6, 3.4, 'response JSON\nkeyed by option names', 1),
  ],
  edges: [
    { from: 'req', to: 'fields', label: 'compile' },
    { from: 'fields', to: 'prompts', label: 'render' },
    { from: 'prompts', to: 'prefix', label: 'tokenize' },
    { from: 'prefix', to: 'endings', label: 'cached' },
    {
      from: 'prompts:n',
      to: 'endings:n',
      via: [
        [9.2, -1.5],
        [18.4, -1.5],
      ],
      label: 'the part that differs',
    },
    { from: 'endings', to: 'logits', label: 'read', labelSide: 'right' },
    { from: 'logits', to: 'probs', label: 'softmax over codes' },
    { from: 'probs', to: 'resp', label: 'map codes to names' },
  ],
  groups: [
    {
      id: 'pass',
      label: 'one forward pass',
      around: ['prefix', 'endings'],
      pad: 0.35,
      dashed: true,
      labelAt: 'bottom-left',
    },
  ],
}

/** The path of one request through a decision-model server, from the client's JSON to the client's JSON. */
export function IoFlow() {
  return (
    <Figure
      title="From request to response"
      caption="The server compiles each question into a field whose allowed answers carry letter codes, and renders one prompt per question in the chat template the model was trained with. Tokenised, the prompts share a long prefix and differ only in a short ending. One forward pass (dashed) reads the prefix once and runs every ending as a batch against its cached state. The code logits at the end of each ending become probabilities, which the server re-keys to the client's option names."
    >
      <Diagram
        spec={spec}
        ariaLabel="Request JSON is compiled to fields with codes, rendered to one prompt per question, tokenised into a shared prefix and per-question endings, read in one forward pass, giving code logits, then probabilities, then the response JSON"
      />
    </Figure>
  )
}
