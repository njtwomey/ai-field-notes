import { Diagram, Figure } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const spec: DiagramSpec = {
  nodes: [
    { id: 'ctx', x: 0, y: 1, w: 3, h: 1.1, label: 'context\ntask, tool schemas, history', tone: 'neutral' },
    { id: 'lm', x: 4.2, y: 1, w: 2.4, h: 0.9, label: 'language model', tone: 0 },
    { id: 'dec', x: 8, y: 1, shape: 'pill', w: 2.6, h: 0.8, label: 'final answer?', tone: 'neutral' },
    { id: 'ans', x: 11.4, y: 1, w: 1.6, h: 0.8, label: 'answer', tone: 'neutral' },
    { id: 'run', x: 8, y: 3.8, w: 3, h: 1.1, label: 'runtime\nparse, validate, execute', tone: 2 },
    {
      id: 'tools',
      x: 12,
      y: 3.8,
      shape: 'stack',
      w: 2.8,
      h: 1.1,
      label: 'tools\nsearch, calculator, code',
      tone: 'neutral',
    },
  ],
  edges: [
    { from: 'ctx', to: 'lm' },
    { from: 'lm', to: 'dec', label: 'output' },
    { from: 'dec', to: 'ans', label: 'yes' },
    { from: 'dec', to: 'run', label: 'no: tool call', labelRotate: false, labelSide: 'right' },
    { from: 'run', to: 'tools', arrow: 'both' },
    { from: 'run:w', to: 'ctx:s', via: [[0, 3.8]], label: 'append the call and its result' },
  ],
}

/** The agent loop: the model writes, the runtime acts, the result goes back into the context. */
export function AgentLoopDiagram() {
  return (
    <Figure
      title="The agent loop"
      caption="The model only ever writes text. When the text is a tool call, the runtime executes it and appends the call and its result to the context, and the model runs again. The loop ends when the model writes a final answer or a step limit is reached."
    >
      <Diagram
        spec={spec}
        ariaLabel="Agent loop: context to language model; if the output is a final answer, return it; otherwise the runtime executes the tool call against the tools and appends the result to the context"
      />
    </Figure>
  )
}
