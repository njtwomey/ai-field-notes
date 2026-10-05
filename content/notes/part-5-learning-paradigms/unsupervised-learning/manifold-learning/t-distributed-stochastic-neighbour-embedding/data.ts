import { normal, stream } from 'aifn-compute/foundation/random'

export const DIM = 10
export const PER_CLUSTER = 40
export const CLUSTER_NAMES = ['A (tight)', 'B (tight, near A)', 'C (wide)', 'D (tight, far)']

/**
 * Four Gaussian clusters in 10 dimensions. A and B are tight and 3 apart; C is spread 4 times wider; D is tight and
 * 12 away from the others. PCA shows these differences; t-SNE largely hides them.
 */
export function clusters(): { x: number[][]; labels: number[] } {
  const r = stream(8)
  const unit = (k: number) => Array.from({ length: DIM }, (_, m) => (m === k ? 1 : 0))
  const centres = [
    new Array<number>(DIM).fill(0),
    unit(0).map((v) => 3 * v),
    unit(1).map((v) => 6 * v),
    unit(2).map((v) => 12 * v),
  ]
  const sd = [0.5, 0.5, 2, 0.5]
  const x: number[][] = []
  const labels: number[] = []
  centres.forEach((c, j) => {
    for (let i = 0; i < PER_CLUSTER; i++) {
      x.push(c.map((v) => v + sd[j] * normal(r)))
      labels.push(j)
    }
  })
  return { x, labels }
}
