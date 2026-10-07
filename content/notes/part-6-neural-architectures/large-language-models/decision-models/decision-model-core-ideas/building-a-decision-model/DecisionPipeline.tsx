import { Diagram, Figure } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const box = (id: string, x: number, y: number, label: string, tone: 0 | 1 | 2 | 'neutral', dashed = false) => ({
  id,
  x,
  y,
  w: 3.6,
  h: 1.3,
  label,
  tone,
  dashed,
})

const spec: DiagramSpec = {
  unit: 40,
  nodes: [
    box('contract', 0, 0, '1. Prompt contract\nlayout, codes, base model', 'neutral'),
    box('data', 4.6, 0, '2. Decision data\npairs, rules, splits', 0),
    box('sft', 9.2, 0, '3. Supervised fine-tune\nLoRA, candidate loss', 0),
    box('rl', 9.2, 2.8, '4. Optional: calibrated RL\nproper-score reward', 1, true),
    box('cal', 4.6, 2.8, '5. Calibrate\n$T$ per type, thresholds', 2),
    box('eval', 0, 2.8, '6. Evaluate\nfrozen test, per type', 'neutral'),
  ],
  edges: [
    { from: 'contract', to: 'data' },
    { from: 'data', to: 'sft' },
    { from: 'sft', to: 'rl' },
    { from: 'rl', to: 'cal' },
    { from: 'cal', to: 'eval' },
    { from: 'eval:n', to: 'contract:s', label: 'failures → new data', labelSide: 'right' },
  ],
}

/** The build order for an open decision model, from the prompt contract to evaluation. */
export function DecisionPipeline() {
  return (
    <Figure
      title="Building a decision model"
      caption="Each stage fixes something the next relies on. The prompt contract is frozen first because every later stage is tied to its tokens. Stage 4, reinforcement learning with a proper-score reward, is optional and can be skipped, going straight from stage 3 to stage 5: it matters when the feedback is a check on sampled answers rather than a label, or when the model reasons before it decides. Evaluation failures become new contrastive data, never tuning on the test set."
    >
      <Diagram
        spec={spec}
        ariaLabel="Six stages: prompt contract, decision data, supervised fine-tune, optional calibrated reinforcement learning, calibration, evaluation; evaluation feeds back to the contract and data"
      />
    </Figure>
  )
}
