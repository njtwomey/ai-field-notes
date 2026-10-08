import { useMemo } from 'react'
import {
  adapterParameters,
  fineTuningCompute,
  fineTuningMemory,
  LLAMA_3_1_70B,
  LLAMA_3_1_8B,
  type AdapterTarget,
  type FineTuningMemory,
  type FineTuningSetup,
} from 'aifn-methods/neural/quantisation'
import {
  Annotation,
  Bars,
  choice,
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Readout,
  setting,
  useAxis,
  useFigureState,
  when,
} from 'aifn-render'

const GB = 1e9
const MODELS = { '8b': LLAMA_3_1_8B, '70b': LLAMA_3_1_70B }
const TARGETS: Record<string, readonly AdapterTarget[]> = {
  all: ['q', 'k', 'v', 'o', 'gate', 'up', 'down'],
  attention: ['q', 'k', 'v', 'o'],
  qv: ['q', 'v'],
}
/** Accelerators: memory in GB and peak dense bf16 throughput in FLOP/s, from the makers' datasheets. */
const GPUS = {
  h100: { name: 'H100', memoryGb: 80, peakFlops: 990e12 },
  a100: { name: 'A100 80 GB', memoryGb: 80, peakFlops: 312e12 },
}
const METHODS = [
  { method: 'full', label: 'full fine-tuning', short: 'full' },
  { method: 'lora', label: 'LoRA, bf16 base', short: 'LoRA' },
  { method: 'qlora', label: 'QLoRA, NF4 base', short: 'QLoRA' },
] as const
/** The parts of the memory, bottom of each bar first; each keeps its slot across the three bars. */
const PARTS = [
  { key: 'weights', label: 'base weights' },
  { key: 'adapterState', label: 'adapter weights' },
  { key: 'optimiserState', label: 'optimiser state' },
  { key: 'gradients', label: 'gradients' },
  { key: 'activations', label: 'activations' },
  { key: 'logits', label: 'logits' },
] as const satisfies readonly { key: keyof FineTuningMemory; label: string }[]
/** Common accelerator memory sizes, drawn as guides. */
const GUIDES = [24, 48, 80]
/** An 8-GPU node of 80 GB devices, drawn when the largest bar passes half of it. */
const NODE = 640

const gb = (bytes: number) => `${formatNumber(bytes / GB)} GB`

function duration(seconds: number) {
  if (seconds < 3600) return `${formatNumber(seconds / 60)} min`
  if (seconds < 48 * 3600) return `${formatNumber(seconds / 3600)} h`
  return `${formatNumber(seconds / 86400)} days`
}

export function MemoryBudget() {
  const state = useFigureState({
    model: choice(
      [
        { value: '8b', label: 'Llama 3.1 8B' },
        { value: '70b', label: 'Llama 3.1 70B' },
      ],
      '8b',
      { label: 'model' },
    ),
    rank: int(16, { ge: 1, le: 1024, suggestions: [8, 16, 64, 256], label: 'adapter rank r' }),
    targets: choice(
      [
        { value: 'all', label: 'all seven linear layers' },
        { value: 'attention', label: 'attention only (q, k, v, o)' },
        { value: 'qv', label: 'query and value only' },
      ],
      'all',
      { label: 'adapted layers' },
    ),
    microBatch: int(8, { ge: 1, le: 256, suggestions: [1, 4, 8, 16, 32], label: 'micro-batch b (sequences)' }),
    sequence: int(1024, {
      ge: 64,
      le: 131072,
      suggestions: [512, 1024, 2048, 4096, 8192, 32768],
      label: 'sequence length s (tokens)',
    }),
    checkpointing: setting(true, 'gradient checkpointing'),
    logits: choice(
      [
        { value: 'chunked', label: 'chunked loss' },
        { value: 'full', label: 'materialised in 32 bits' },
      ],
      'chunked',
      { label: 'logits' },
    ),
    optimiser: choice(
      [
        { value: 'adam32', label: 'Adam, 32-bit moments' },
        { value: 'adam8', label: 'Adam, 8-bit moments' },
        { value: 'paged', label: 'paged Adam, 32-bit' },
      ],
      'adam32',
      { label: 'optimiser' },
    ),
    doubleQuant: setting(true, 'double quantisation (QLoRA)'),
    tokens: float(2.52e6, {
      gt: 0,
      le: 1e12,
      scale: 'log10',
      suggestions: [1e6, 2.52e6, 1e7, 1e8, 1e9],
      label: 'tokens per epoch',
    }),
    epochs: int(3, { ge: 1, le: 100, suggestions: [1, 2, 3, 5], label: 'epochs' }),
    mfu: float(0.3, { gt: 0, le: 1, suggestions: [0.2, 0.3, 0.4, 0.5], label: 'model FLOP utilisation' }),
    gpu: choice(
      [
        { value: 'h100', label: 'H100: 80 GB, 990 TFLOP/s' },
        { value: 'a100', label: 'A100: 80 GB, 312 TFLOP/s' },
      ],
      'h100',
      { label: 'accelerator (dense bf16)' },
    ),
    price: float(3, { gt: 0, le: 100, suggestions: [1, 2, 3, 4], label: 'price per GPU-hour ($)' }),
    attention: choice(
      [
        { value: 'fused', label: 'fused kernel, recomputed' },
        { value: 'stored', label: 'stored, 5as/h term' },
      ],
      'fused',
      { label: 'attention matrix', when: when('checkpointing', false) },
    ),
  })
  const { model, rank, targets, microBatch, sequence, checkpointing, logits, optimiser, doubleQuant } = state
  const { tokens, epochs, mfu, gpu, price, attention } = state
  const shape = MODELS[model]
  const device = GPUS[gpu]

  const result = useMemo(() => {
    const setup = (method: FineTuningSetup['method']): FineTuningSetup => ({
      method,
      rank,
      targets: TARGETS[targets],
      microBatch,
      sequence,
      checkpointing,
      logits,
      optimiser,
      doubleQuant,
      attention,
    })
    const run = { tokens, epochs, mfu, peakFlops: device.peakFlops, pricePerHour: price }
    return METHODS.map(({ method, label }) => ({
      method,
      label,
      memory: fineTuningMemory(shape, setup(method), { memoryGb: device.memoryGb }),
      compute: fineTuningCompute(shape, setup(method), run),
    }))
  }, [
    shape,
    rank,
    targets,
    microBatch,
    sequence,
    checkpointing,
    logits,
    optimiser,
    doubleQuant,
    attention,
    tokens,
    epochs,
    mfu,
    device,
    price,
  ])

  const adapters = adapterParameters(shape, rank, TARGETS[targets])
  const largest = Math.max(...result.map((r) => r.memory.total)) / GB

  // One bar per method, stacked part by part; a layer per (part, method) carries its own base.
  const stacks = useMemo(
    () =>
      result.map((r, k) => {
        let base = 0
        return PARTS.map((part, slot) => {
          const top = base + r.memory[part.key] / GB
          const bar = { k, slot, part: part.label, base, top }
          base = top
          return bar
        })
      }),
    [result],
  )

  const x = useAxis({ label: 'accelerator memory (GB, 10⁹ bytes)', range: [0, undefined], key: model })
  const y = useAxis({ label: 'method', categories: METHODS.map((m) => m.short), inverse: true })

  const [full, lora, qlora] = result
  return (
    <Figure
      title="Fine-tuning memory budget"
      state={state}
      defaultSize="L"
      caption={`The memory one accelerator holds for a fine-tuning run of Llama 3.1, by part: the base weights (bf16, or NF4 linear layers with bf16 embeddings for QLoRA), the adapters' 32-bit weights, the optimiser state (Adam's two moments, plus the 32-bit master copy for full fine-tuning), the gradients (bf16 for full fine-tuning, 32-bit for adapters), the activations of one micro-batch and its 32-bit logits. Dashed lines mark 24, 48 and 80 GB devices. The defaults are the run of the bootstrapping note: micro-batch 8 × 1,024 tokens, rank 16 on all seven linear layers, checkpointing and a chunked loss, for 132.1 GB, 20.3 GB and 10.0 GB. The optimiser state dominates full fine-tuning and the bf16 base dominates LoRA. QLoRA's 4-bit base is small enough that the activations of one micro-batch are over a third of its total. Turn checkpointing off or materialise the logits to watch the activation terms overtake the model state; raise the rank to see how little the adapters weigh. Paged Adam still counts its state, but a run fits if the rest does, since the state can spill to host memory. Compute counts 6Ψ FLOPs per token for full fine-tuning (8Ψ with checkpointing) and 4Ψ + 2Ψₐ for LoRA and QLoRA (6Ψ + 2Ψₐ with checkpointing), for Ψ parameters and Ψₐ adapter parameters; QLoRA's dequantisation adds time but no matrix FLOPs, so its real runs are slower than this count. Time and cost are GPU-hours at the stated utilisation, whether or not the run fits one device.`}
      readouts={{
        'memory per device': (
          <>
            {result.map((r) => (
              <Readout
                key={r.method}
                label={r.label}
                value={`${gb(r.memory.total)}${r.memory.fits ? `, fits one ${device.name}` : ''}`}
              />
            ))}
          </>
        ),
        adapters: (
          <>
            <Readout label="adapter parameters Ψₐ" value={formatNumber(adapters)} />
            <Readout label="share of the model" value={`${formatNumber((100 * adapters) / shape.parameters)}%`} />
          </>
        ),
        [`compute, ${formatNumber(tokens * epochs)} tokens on ${device.name}`]: (
          <>
            <Readout label="FLOPs per token, full" value={formatNumber(full.compute.flopsPerToken)} />
            <Readout label="FLOPs per token, LoRA and QLoRA" value={formatNumber(lora.compute.flopsPerToken)} />
            <Readout
              label="full: GPU time, cost"
              value={`${duration(full.compute.seconds)}, $${formatNumber(full.compute.cost)}`}
            />
            <Readout
              label="LoRA and QLoRA: GPU time, cost"
              value={`${duration(qlora.compute.seconds)}, $${formatNumber(qlora.compute.cost)}`}
            />
          </>
        ),
      }}
    >
      <Plot x={x} y={y}>
        {stacks.flat().map((bar) => (
          <Bars
            key={`${bar.slot}-${bar.k}`}
            id={`part-${bar.slot}-${bar.k}`}
            name={bar.part}
            x={[bar.k]}
            y={[bar.top]}
            base={bar.base}
            width={0.6}
            orient="y"
            slot={bar.slot}
          />
        ))}
        {GUIDES.map((g) => (
          <Annotation key={g} x={g} text={`${g} GB`} dashed muted />
        ))}
        {largest > NODE / 2 && <Annotation x={NODE} text={`8 × 80 GB`} dashed muted />}
      </Plot>
    </Figure>
  )
}
