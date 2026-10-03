/** Architecture diagrams shared by the neural topic model notes. Style follows the VAE spec in specs/generative.ts. */
import { merge, op, projector, reparam } from 'aifn-render'
import type { DiagramEdge, DiagramNode, DiagramSpec } from 'aifn-render'

export type NtmInput = 'bow' | 'combined' | 'contextual'
export type NtmDecoder = 'nvdm' | 'mixture' | 'product' | 'etm'

const DECODER_LABEL: Record<NtmDecoder, string> = {
  nvdm: 'decoder\n$\\operatorname{softmax}(\\Rmat^\\top\\hvec + \\bvec)$',
  mixture: 'decoder\n$\\thetavec^\\top\\Bmat$',
  product: 'decoder\n$\\operatorname{softmax}(\\thetavec^\\top\\Bmat)$',
  etm: 'decoder\n$\\sum_k \\theta_k\\betavec_k$\n$\\beta_{kv} \\propto e^{\\rhovec_v^\\top\\alphavec_k}$',
}

const PRIOR_LABEL: Record<NtmDecoder, string> = {
  nvdm: 'KL to $\\Gauss(\\zeros, \\Imat)$',
  mixture: 'KL to $\\Gauss(\\muvec_0, \\Sigmamat_0)$',
  product: 'KL to $\\Gauss(\\muvec_0, \\Sigmamat_0)$',
  etm: 'KL to $\\Gauss(\\zeros, \\Imat)$',
}

/**
 * Encoder network → Gaussian latent h (reparameterised) → softmax → θ → decoder → word distribution. NVDM has no
 * softmax: the decoder reads h directly. `input` adds a contextual sentence embedding beside, or instead of, the
 * bag of words, as in contextualised topic models.
 */
export function ntmSpec(decoder: NtmDecoder, input: NtmInput = 'bow'): DiagramSpec {
  const simplex = decoder !== 'nvdm'
  const xDec = simplex ? 14.4 : 11.6
  const nodes: DiagramNode[] = [
    projector('encoder', 'enc', 2.4, 3, 'encoder\n$\\text{MLP}_{\\phivec}$', { w: 2, h: 3 }),
    { id: 'h', x: 9.3, y: 3, shape: 'latent', label: '$\\hvec_d$', tone: 2 },
    { id: 'kl', x: 4.8, y: 0.6, shape: 'text', small: true, label: PRIOR_LABEL[decoder] },
    { id: 'dec', x: xDec, y: 3, w: 2.8, h: 1.6, label: DECODER_LABEL[decoder], tone: 1 },
    { id: 'out', x: xDec + 2.9, y: 3, w: 1.1, h: 0.9, label: '$\\hat\\wvec_d$' },
    { id: 'rec', x: xDec + 2.9, y: 5.3, shape: 'text', small: true, label: 'reconstruct\nthe words' },
  ]
  const edges: DiagramEdge[] = [
    { from: 'rpadd', to: 'h' },
    { from: 'rpmu:n', to: 'kl:s', dashed: true, arrow: 'none' },
    { from: 'dec', to: 'out' },
    { from: 'out', to: 'rec', dashed: true, arrow: 'none' },
  ]
  if (simplex) {
    nodes.push(op('sm', 10.7, 3, '$\\sigma$'), {
      id: 'theta',
      x: 12,
      y: 3,
      shape: 'latent',
      label: '$\\thetavec_d$',
      tone: 2,
    })
    edges.push({ from: 'h', to: 'sm' }, { from: 'sm', to: 'theta' }, { from: 'theta', to: 'dec' })
  } else {
    edges.push({ from: 'h', to: 'dec' })
  }
  if (input !== 'contextual') {
    nodes.push({ id: 'bow', x: 0, y: input === 'combined' ? 2.2 : 3, w: 1.2, h: 0.9, label: '$\\wvec_d$' })
    edges.push({ from: 'bow', to: 'enc' })
  }
  if (input !== 'bow') {
    nodes.push({ id: 'sbert', x: 0, y: input === 'combined' ? 4.2 : 3, w: 1.4, h: 0.9, label: 'SBERT', tone: 3 })
    edges.push({ from: 'sbert', to: 'enc' })
  }
  return merge({ nodes, edges }, reparam('rp', 4.8, 3, 'enc'))
}

/** BERTopic: embed, reduce, cluster, then describe each cluster with class-based TF-IDF. */
export const bertopicSpec: DiagramSpec = {
  nodes: [
    { id: 'docs', x: 0, y: 1.5, w: 1.8, h: 0.9, label: 'documents' },
    { id: 'emb', x: 2.9, y: 1.5, w: 2.2, h: 0.9, label: 'SBERT\nembeddings', tone: 3 },
    { id: 'umap', x: 5.9, y: 1.5, w: 2, h: 0.9, label: 'UMAP\nreduce', tone: 0 },
    { id: 'hdb', x: 8.8, y: 1.5, w: 2.2, h: 0.9, label: 'HDBSCAN\nclusters', tone: 0 },
    { id: 'ctfidf', x: 11.8, y: 1.5, w: 2.2, h: 0.9, label: 'c-TF-IDF', tone: 1 },
    { id: 'words', x: 14.6, y: 1.5, w: 1.9, h: 0.9, label: 'topic\nwords' },
    { id: 'bow', x: 8.8, y: 3.6, w: 2.6, h: 0.9, label: 'bag of words\nper cluster', tone: 'neutral' },
  ],
  edges: [
    { from: 'docs', to: 'emb' },
    { from: 'emb', to: 'umap' },
    { from: 'umap', to: 'hdb' },
    { from: 'hdb', to: 'ctfidf' },
    { from: 'ctfidf', to: 'words' },
    { from: 'docs:s', to: 'bow:w', via: [[0, 3.6]], dashed: true, label: 'concatenate', labelPos: 0.7 },
    { from: 'hdb', to: 'bow', dashed: true },
    { from: 'bow:e', to: 'ctfidf:s', via: [[11.8, 3.6]] },
  ],
  groups: [
    { id: 'geo', label: 'clustering in embedding space', tone: 0, dashed: true, around: ['emb', 'umap', 'hdb'] },
  ],
}
