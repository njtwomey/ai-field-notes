import { useMemo, useState } from 'react'
import { MathText } from 'aifn-render'
import { Diagram } from 'aifn-render'
import { Interactive, ParamChoice } from 'aifn-render'
import { ntmSpec, type NtmDecoder, type NtmInput } from './specs'

type Variant = 'nvdm' | 'avitm' | 'prodlda' | 'etm' | 'ctm' | 'zeroshot'

const VARIANTS: Record<Variant, { label: string; decoder: NtmDecoder; input: NtmInput; caption: string }> = {
  nvdm: {
    label: 'NVDM',
    decoder: 'nvdm',
    input: 'bow',
    caption:
      'NVDM: the Gaussian latent $\\hvec_d$ feeds a softmax decoder directly. There are no topic proportions on the simplex, and the prior is $\\Gauss(\\zeros, \\Imat)$.',
  },
  avitm: {
    label: 'LDA (AVITM)',
    decoder: 'mixture',
    input: 'bow',
    caption:
      'LDA trained by amortised inference: the softmax of $\\hvec_d$ gives topic proportions $\\thetavec_d$, and the decoder mixes the topic–word distributions (rows of $\\Bmat$ on the simplex). The prior $\\Gauss(\\muvec_0, \\Sigmamat_0)$ is the Laplace approximation of a Dirichlet.',
  },
  prodlda: {
    label: 'ProdLDA',
    decoder: 'product',
    input: 'bow',
    caption:
      'ProdLDA: as LDA, but $\\Bmat$ is unnormalised and the mixing happens before the softmax, which makes the word distribution a weighted product of experts.',
  },
  etm: {
    label: 'ETM',
    decoder: 'etm',
    input: 'bow',
    caption:
      'ETM: each topic is a vector $\\alphavec_k$ in word-embedding space, and topic $k$ gives word $v$ probability proportional to $\\exp(\\rhovec_v^\\top\\alphavec_k)$. The prior on $\\hvec_d$ is $\\Gauss(\\zeros, \\Imat)$.',
  },
  ctm: {
    label: 'CTM',
    decoder: 'product',
    input: 'combined',
    caption:
      'Combined contextualised topic model: the encoder reads the bag of words and a sentence embedding of the raw text. The decoder is ProdLDA’s.',
  },
  zeroshot: {
    label: 'ZeroShotTM',
    decoder: 'product',
    input: 'contextual',
    caption:
      'ZeroShotTM: the encoder reads only the sentence embedding, so a multilingual embedder lets it infer $\\thetavec_d$ for text in languages it never saw during training.',
  },
}

/** The encoder–decoder architecture of a neural topic model, with a choice of variant. */
export function NtmDiagram({ variants, initial }: { variants: Variant[]; initial?: Variant }) {
  const [variant, setVariant] = useState<Variant>(initial ?? variants[0])
  const v = VARIANTS[variant]
  const spec = useMemo(() => ntmSpec(v.decoder, v.input), [v.decoder, v.input])
  return (
    <Interactive
      title="A neural topic model"
      caption={
        <MathText
          text={`${v.caption} The reparameterisation block writes $\\hvec_d = \\muvec + \\sigmavec \\odot \\epsilonvec$, so gradients reach the encoder; the KL term acts on the encoder's output and the reconstruction term on the decoder's.`}
        />
      }
      controls={
        variants.length > 1 ? (
          <ParamChoice
            label="model"
            value={variant}
            onChange={setVariant}
            options={variants.map((id) => ({ value: id, label: VARIANTS[id].label }))}
          />
        ) : undefined
      }
    >
      <Diagram
        spec={spec}
        ariaLabel={`Architecture of ${v.label}: encoder, reparameterised Gaussian latent, decoder`}
      />
    </Interactive>
  )
}
