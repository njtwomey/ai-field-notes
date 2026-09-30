/**
 * Ranking losses of one list, as functions of the item scores s and graded relevance labels rel (0 = irrelevant).
 * Each returns the loss and its gradient with respect to every score, so a figure can show where each family pushes.
 * Pairwise and lambda losses sum over the pairs (i, j) with rel_i > rel_j.
 */

export type LossId =
  'pointwise-bce' | 'pointwise-mse' | 'ranknet' | 'hinge' | 'lambdarank' | 'listnet' | 'listmle' | 'softmax'

export type LossValue = { loss: number; grad: number[] }

export const LOSSES: Record<LossId, { label: string; family: 'pointwise' | 'pairwise' | 'listwise' }> = {
  'pointwise-bce': { label: 'pointwise: binary cross-entropy', family: 'pointwise' },
  'pointwise-mse': { label: 'pointwise: squared error on grades', family: 'pointwise' },
  ranknet: { label: 'pairwise: logistic (RankNet, BPR)', family: 'pairwise' },
  hinge: { label: 'pairwise: hinge (RankSVM)', family: 'pairwise' },
  lambdarank: { label: 'pairwise: LambdaRank (ΔNDCG-weighted)', family: 'pairwise' },
  softmax: { label: 'listwise: softmax cross-entropy', family: 'listwise' },
  listnet: { label: 'listwise: ListNet (top-one)', family: 'listwise' },
  listmle: { label: 'listwise: ListMLE (Plackett–Luce)', family: 'listwise' },
}

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z))
/** log(1 + e^z) without overflow. */
const softplus = (z: number) => Math.max(z, 0) + Math.log1p(Math.exp(-Math.abs(z)))

function softmax(s: number[]): number[] {
  const m = Math.max(...s)
  const e = s.map((v) => Math.exp(v - m))
  const z = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / z)
}

/** Positions (0 = top) of each item when sorted by score, ties broken by index. */
export function ranks(s: number[]): number[] {
  const order = s.map((_, i) => i).sort((a, b) => s[b] - s[a] || a - b)
  const pos = Array<number>(s.length)
  order.forEach((item, r) => (pos[item] = r))
  return pos
}

const gain = (rel: number) => 2 ** rel - 1
const discount = (rank: number) => 1 / Math.log2(rank + 2)

/** NDCG of the ranking that the scores induce, over the whole list. */
export function ndcg(s: number[], rel: number[]): number {
  const pos = ranks(s)
  const dcg = rel.reduce((a, r, i) => a + gain(r) * discount(pos[i]), 0)
  const ideal = [...rel].sort((a, b) => b - a).reduce((a, r, k) => a + gain(r) * discount(k), 0)
  return ideal > 0 ? dcg / ideal : 0
}

export function rankingLoss(id: LossId, s: number[], rel: number[]): LossValue {
  const n = s.length
  const grad = Array<number>(n).fill(0)
  let loss = 0
  const pairs = () => {
    const out: [number, number][] = []
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (rel[i] > rel[j]) out.push([i, j])
    return out
  }
  switch (id) {
    case 'pointwise-bce':
      for (let i = 0; i < n; i++) {
        const y = rel[i] > 0 ? 1 : 0
        loss += softplus(s[i]) - y * s[i]
        grad[i] = sigmoid(s[i]) - y
      }
      break
    case 'pointwise-mse':
      for (let i = 0; i < n; i++) {
        loss += 0.5 * (s[i] - rel[i]) ** 2
        grad[i] = s[i] - rel[i]
      }
      break
    case 'ranknet':
      for (const [i, j] of pairs()) {
        loss += softplus(-(s[i] - s[j]))
        const g = -sigmoid(-(s[i] - s[j]))
        grad[i] += g
        grad[j] -= g
      }
      break
    case 'hinge':
      for (const [i, j] of pairs()) {
        const margin = 1 - (s[i] - s[j])
        if (margin > 0) {
          loss += margin
          grad[i] -= 1
          grad[j] += 1
        }
      }
      break
    case 'lambdarank': {
      // RankNet's pair gradient scaled by the change in NDCG from swapping the two items. There is no closed-form
      // loss; `loss` reports the ΔNDCG-weighted RankNet sum, whose gradient at fixed ranks is these lambdas.
      const pos = ranks(s)
      const ideal = [...rel].sort((a, b) => b - a).reduce((a, r, k) => a + gain(r) * discount(k), 0)
      for (const [i, j] of pairs()) {
        const delta = Math.abs((gain(rel[i]) - gain(rel[j])) * (discount(pos[i]) - discount(pos[j]))) / (ideal || 1)
        loss += delta * softplus(-(s[i] - s[j]))
        const g = -delta * sigmoid(-(s[i] - s[j]))
        grad[i] += g
        grad[j] -= g
      }
      break
    }
    case 'softmax': {
      // Cross-entropy between the labels normalised to sum to one and the softmax of the scores.
      const total = rel.reduce((a, b) => a + b, 0)
      const target = rel.map((r) => (total > 0 ? r / total : 1 / n))
      const p = softmax(s)
      for (let i = 0; i < n; i++) {
        loss -= target[i] * Math.log(p[i])
        grad[i] = p[i] - target[i]
      }
      break
    }
    case 'listnet': {
      // ListNet's top-one probabilities: the target is the softmax of the labels themselves.
      const target = softmax(rel)
      const p = softmax(s)
      for (let i = 0; i < n; i++) {
        loss -= target[i] * Math.log(p[i])
        grad[i] = p[i] - target[i]
      }
      break
    }
    case 'listmle': {
      // Negative log-likelihood, under Plackett–Luce, of the permutation that sorts by relevance (ties by index).
      const order = rel.map((_, i) => i).sort((a, b) => rel[b] - rel[a] || a - b)
      for (let k = 0; k < n; k++) {
        const rest = order.slice(k)
        const p = softmax(rest.map((i) => s[i]))
        loss += -Math.log(p[0])
        rest.forEach((i, m) => (grad[i] += p[m] - (m === 0 ? 1 : 0)))
      }
      break
    }
  }
  return { loss, grad }
}
