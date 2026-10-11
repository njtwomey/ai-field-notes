import { choice, Diagram, Figure, useFigureState } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const LAYOUTS = [
  { value: 'early', label: 'early fusion' },
  { value: 'late', label: 'late fusion' },
  { value: 'intermediate', label: 'intermediate fusion' },
  { value: 'interleaved', label: 'interleaved sequence' },
] as const
type Layout = (typeof LAYOUTS)[number]['value']

/** Image inputs and their parts in tone 0, text in tone 1, joint parts neutral. */
const SPECS: Record<Layout, DiagramSpec> = {
  early: {
    unit: 40,
    nodes: [
      { id: 'img', x: 0, y: 0, w: 2.6, h: 0.9, label: 'image features', tone: 0 },
      { id: 'txt', x: 0, y: 2.2, w: 2.6, h: 0.9, label: 'text features', tone: 1 },
      { id: 'cat', x: 3.6, y: 1.1, shape: 'op', label: '‖' },
      { id: 'vec', x: 6.4, y: 1.1, w: 3.2, h: 0.9, label: 'one joint vector', tone: 'neutral' },
      { id: 'model', x: 10, y: 1.1, w: 2.4, h: 0.9, label: 'one model', tone: 'neutral' },
      { id: 'y', x: 12.6, y: 1.1, shape: 'pill', w: 1.2, h: 0.8, label: '$\\hat y$', tone: 'neutral' },
    ],
    edges: [
      { from: 'img', to: 'cat' },
      { from: 'txt', to: 'cat' },
      { from: 'cat', to: 'vec' },
      { from: 'vec', to: 'model' },
      { from: 'model', to: 'y' },
    ],
  },
  late: {
    unit: 40,
    nodes: [
      { id: 'img', x: 0, y: 0, w: 2.2, h: 0.9, label: 'image', tone: 0 },
      { id: 'txt', x: 0, y: 2.6, w: 2.2, h: 0.9, label: 'text', tone: 1 },
      { id: 'ma', x: 3.8, y: 0, w: 2.6, h: 0.9, label: 'image model', tone: 0 },
      { id: 'mb', x: 3.8, y: 2.6, w: 2.6, h: 0.9, label: 'text model', tone: 1 },
      { id: 'pa', x: 7.4, y: 0, shape: 'pill', w: 1.6, h: 0.8, label: '$p_\\text{image}$', tone: 0 },
      { id: 'pb', x: 7.4, y: 2.6, shape: 'pill', w: 1.6, h: 0.8, label: '$p_\\text{text}$', tone: 1 },
      { id: 'comb', x: 10, y: 1.3, w: 2.6, h: 0.9, label: 'combine', tone: 'neutral' },
      { id: 'y', x: 12.6, y: 1.3, shape: 'pill', w: 1.2, h: 0.8, label: '$\\hat y$', tone: 'neutral' },
    ],
    edges: [
      { from: 'img', to: 'ma' },
      { from: 'txt', to: 'mb' },
      { from: 'ma', to: 'pa' },
      { from: 'mb', to: 'pb' },
      { from: 'pa', to: 'comb' },
      { from: 'pb', to: 'comb' },
      { from: 'comb', to: 'y', label: 'average or learned weights' },
    ],
  },
  intermediate: {
    unit: 40,
    nodes: [
      { id: 'img', x: 0, y: 0, w: 2.2, h: 0.9, label: 'image', tone: 0 },
      { id: 'txt', x: 0, y: 3, w: 2.2, h: 0.9, label: 'text', tone: 1 },
      { id: 'ea', x: 3.6, y: 0, shape: 'encoder', w: 2.4, h: 1.1, label: 'image encoder', tone: 0 },
      { id: 'eb', x: 3.6, y: 3, shape: 'encoder', w: 2.4, h: 1.1, label: 'text encoder', tone: 1 },
      { id: 'ta', x: 6.8, y: 0, shape: 'stack', w: 1.8, h: 0.9, label: 'image tokens', tone: 0 },
      { id: 'tb', x: 6.8, y: 3, shape: 'stack', w: 1.8, h: 0.9, label: 'text tokens', tone: 1 },
      { id: 'joint', x: 9.8, y: 1.5, w: 2.8, h: 1.4, label: 'joint layers\ncross-attention', tone: 'neutral' },
      { id: 'y', x: 12.6, y: 1.5, shape: 'pill', w: 1.2, h: 0.8, label: '$\\hat y$', tone: 'neutral' },
    ],
    edges: [
      { from: 'img', to: 'ea' },
      { from: 'txt', to: 'eb' },
      { from: 'ea', to: 'ta' },
      { from: 'eb', to: 'tb' },
      { from: 'ta', to: 'joint' },
      { from: 'tb', to: 'joint' },
      { from: 'joint', to: 'y' },
    ],
  },
  interleaved: {
    unit: 40,
    nodes: [
      { id: 't1', x: 0, y: 3, w: 1.3, h: 0.8, label: 'text', tone: 1 },
      { id: 't2', x: 1.6, y: 3, w: 1.3, h: 0.8, label: 'text', tone: 1 },
      { id: 'i1', x: 3.2, y: 3, w: 1.3, h: 0.8, label: 'image', tone: 0 },
      { id: 'i2', x: 4.8, y: 3, w: 1.3, h: 0.8, label: 'image', tone: 0 },
      { id: 't3', x: 6.4, y: 3, w: 1.3, h: 0.8, label: 'text', tone: 1 },
      { id: 'i3', x: 8, y: 3, w: 1.3, h: 0.8, label: 'image', tone: 0 },
      { id: 't4', x: 9.6, y: 3, w: 1.3, h: 0.8, label: 'text', tone: 1 },
      { id: 'lm', x: 4.8, y: 0.8, w: 10.6, h: 1.1, label: 'one transformer over the whole sequence', tone: 'neutral' },
      { id: 'y', x: 12.4, y: 0.8, shape: 'pill', w: 1.8, h: 0.8, label: 'next token', tone: 'neutral' },
    ],
    edges: [
      { from: 't1', to: 'lm' },
      { from: 't2', to: 'lm' },
      { from: 'i1', to: 'lm' },
      { from: 'i2', to: 'lm' },
      { from: 't3', to: 'lm' },
      { from: 'i3', to: 'lm' },
      { from: 't4', to: 'lm' },
      { from: 'lm', to: 'y' },
    ],
  },
}

const CAPTIONS: Record<Layout, string> = {
  early:
    'Early fusion: each modality is turned into features, the features are concatenated into one vector per example, and one model reads the vector. Interactions between modalities are available from the first layer; every example needs every modality, aligned to the same unit.',
  late: 'Late fusion: a separate model per modality makes its own prediction, and the predictions are combined. Any subset of modalities can be used at prediction; interactions between modalities are never learned.',
  intermediate:
    'Intermediate fusion: an encoder per modality turns each input into a sequence of tokens, and joint layers, here cross-attention, let the token sequences exchange information. Each encoder can be pretrained on its own modality.',
  interleaved:
    'Interleaved sequence: images become blocks of tokens placed where they occur in a document, between the text tokens, and one transformer models the whole sequence. The layout follows the source document, so a training example is a document rather than a pair.',
}

/** The four ways to arrange multimodal inputs for a model, one at a time. */
export function FusionLayouts() {
  const state = useFigureState({
    layout: choice(LAYOUTS, 'early', { label: 'layout' }),
  })
  const layout = state.layout as Layout
  return (
    <Figure title="Four data layouts for fusion" state={state} hoverReadout={false} caption={CAPTIONS[layout]}>
      <Diagram
        spec={SPECS[layout]}
        ariaLabel={`Diagram of ${LAYOUTS.find((l) => l.value === layout)?.label}: ${CAPTIONS[layout]}`}
      />
    </Figure>
  )
}
