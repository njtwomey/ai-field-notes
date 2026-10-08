import { Diagram, Figure, MathText } from 'aifn-render'
import type { DiagramSpec, FigureSize } from 'aifn-render'
import { adapter, ia3, lora, overview, prefixTuning, promptTuning, reft, subsets } from './diagrams'

function PeftFigure({
  title,
  spec,
  aria,
  caption,
  size,
}: {
  title: string
  spec: DiagramSpec
  aria: string
  caption: string
  size?: FigureSize
}) {
  // A diagram has no values under the pointer, so the hover readout line is switched off.
  return (
    <Figure title={title} caption={<MathText text={caption} />} defaultSize={size} hoverReadout={false}>
      <Diagram spec={spec} ariaLabel={aria} />
    </Figure>
  )
}

/** One frozen transformer block with every method's attachment point marked. */
export function PeftOverview() {
  return (
    <PeftFigure
      title="Where each method attaches"
      spec={overview}
      size="L"
      aria="A pre-norm transformer block: norm, query, key and value projections, attention, output projection and residual sum, then norm, gated feed-forward network and residual sum. Labels mark where prompt tuning, LoRA, IA3, prefix tuning, BitFit, adapters and ReFT attach."
      caption="One pre-norm transformer block with a gated feed-forward network: the attention sublayer on the top row and the feed-forward sublayer on the bottom row, each read left to right. Frozen parts are neutral and trainable parts are in the accent colour, here and in the diagrams below. Each label points to where a method's parameters act: prompt tuning at the input of the first block, LoRA beside a weight matrix (any of the seven), IA3 on the keys, values and feed-forward activation, prefix tuning on the keys and values inside attention, BitFit on the biases of every linear layer, adapters after each sublayer, and ReFT on the hidden state $\hvec$. A typical configuration uses one method."
    />
  )
}

export function AdapterDiagram() {
  return (
    <PeftFigure
      title="Bottleneck adapter"
      spec={adapter}
      aria="The sublayer output h passes through a trainable down-projection, a nonlinearity f and a trainable up-projection, and is added back to h; the block's residual connection then adds the input x."
      caption="A Houlsby adapter after one frozen sublayer; a second adapter sits after the other sublayer. The trainable $\Wmat_{\text{down}}$ maps the sublayer output $\hvec$ from width $d$ to width $m$, the nonlinearity $f$ acts, and the trainable $\Wmat_{\text{up}}$ maps back to width $d$. The adapter adds the result to $\hvec$, and the block's residual connection then adds the block input $\xvec$."
    />
  )
}

export function PromptTuningDiagram() {
  return (
    <PeftFigure
      title="Prompt tuning"
      spec={promptTuning}
      aria="Trainable soft-prompt vectors P are placed before the frozen token embeddings, and the sequence of l plus n vectors enters the frozen stack of transformer blocks."
      caption="Prompt tuning places $\ell$ trainable vectors $\Pmat$ before the $n$ frozen token embeddings, and the frozen stack processes all $\ell + n$ positions. Only $\Pmat$ is updated, and its gradient passes back through every block. P-tuning produces the vectors with a small trainable prompt encoder and can mix them with discrete prompt tokens."
    />
  )
}

export function PrefixTuningDiagram() {
  return (
    <PeftFigure
      title="Prefix tuning"
      spec={prefixTuning}
      aria="Inside one attention head, trainable prefix keys and values are placed before the keys and values computed by the frozen projections; the query attends over all of them."
      caption="One attention head of one block; every block has its own prefix. The $\ell$ trainable prefix keys $\pvec_{1:\ell}$ and values $\uvec_{1:\ell}$ sit before the keys $\kvec_{1:n}$ and values $\vvec_{1:n}$ that the frozen projections compute from the context. The query $\qvec$ attends over all $\ell + n$ keys, and the weights average all $\ell + n$ values. Deep prompts, as in P-tuning v2, act at the same place."
    />
  )
}

export function Ia3Diagram() {
  return (
    <PeftFigure
      title="Scaling vectors"
      spec={ia3}
      aria="IA3: the keys and values leaving the frozen projections are multiplied element-wise by trainable vectors l_k and l_v; in the feed-forward network the gated activation is multiplied by l_ff before the down projection."
      caption="IA3 multiplies the keys, the values and the feed-forward network's inner activation element-wise by the trainable vectors $\lvec_k$, $\lvec_v$ and $\lvec_{\text{ff}}$. Every projection stays frozen and the queries are unchanged. Norms and residual connections are omitted."
    />
  )
}

export function SubsetsDiagram() {
  return (
    <PeftFigure
      title="Training a subset"
      spec={subsets}
      aria="BitFit: a frozen weight matrix W with a trainable bias b. Sparse fine-tuning: a frozen matrix W0 plus a difference delta with a few trainable entries. Partial fine-tuning: embeddings and lower blocks frozen, top two blocks trainable."
      caption="Three ways to train part of the existing parameters. BitFit trains only the bias $\bvec$ of each layer. Sparse fine-tuning trains a few entries of a difference $\deltavec$ added to the frozen $\Wmat_0$; the dots are entries that stay zero. Partial fine-tuning trains whole tensors, here the top two of $L$ blocks."
    />
  )
}

export function LoraDiagram() {
  return (
    <PeftFigure
      title="Low-rank adaptation"
      spec={lora}
      aria="LoRA: the input x passes through the frozen matrix W0 and, in parallel, through trainable A and B and a scale alpha over r; the two results are summed."
      caption="LoRA adds a trainable path beside a frozen weight matrix $\Wmat_0$. The input $\xvec$ of width $k$ passes through $\Amat$ to width $r$ and through $\Bmat$ back to width $d$, and the result is scaled by $\alpha / r$ and added to $\Wmat_0 \xvec$. After training, $\Wmat_0 + (\alpha / r)\, \Bmat \Amat$ replaces $\Wmat_0$ and the side path disappears."
    />
  )
}

export function ReftDiagram() {
  return (
    <PeftFigure
      title="Representation intervention"
      spec={reft}
      aria="LoReFT: between two frozen blocks, the hidden state h feeds a trainable side path computing R transpose times (W h plus b minus R h), which is added back to h."
      caption="LoReFT edits the hidden state $\hvec$ between two frozen blocks, at chosen positions only: $\hvec \leftarrow \hvec + \Rmat^{\tr}(\Wmat \hvec + \bvec - \Rmat \hvec)$. The trainable $\Rmat \in \reals^{r \times d}$ has orthonormal rows and appears twice, and $\Wmat \in \reals^{r \times d}$ and $\bvec \in \reals^{r}$ are also trained. The edit replaces the coordinates of $\hvec$ along the $r$ rows of $\Rmat$ with $\Wmat \hvec + \bvec$. Other positions pass unchanged."
    />
  )
}
