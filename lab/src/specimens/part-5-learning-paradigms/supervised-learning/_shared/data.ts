import { blobs, circles, moons, xor } from 'aifn-methods/data/synthetic'
import { type Dataset } from 'aifn-methods/data'
import { stream } from 'aifn-compute/foundation/random'

/** The 2-D datasets the classify and cluster pages share, all from `aifn-methods/data` with fixed streams. */
export const DATASETS = {
  blobs: {
    label: 'three blobs',
    make: (): Dataset =>
      blobs(stream('lab/classify/blobs'), {
        n: 150,
        centers: [
          [-2, -1],
          [2, -1],
          [0, 2],
        ],
        sd: 0.9,
      }),
  },
  moons: { label: 'two moons', make: (): Dataset => moons(stream('lab/classify/moons'), { n: 160, noise: 0.15 }) },
  circles: {
    label: 'two circles',
    make: (): Dataset => circles(stream('lab/classify/circles'), { n: 160, noise: 0.08, factor: 0.45 }),
  },
  xor: { label: 'XOR', make: (): Dataset => xor(stream('lab/classify/xor'), { n: 160, kind: 'gaussian', sd: 0.45 }) },
} as const

export type DatasetName = keyof typeof DATASETS

export const DATASET_OPTIONS = (Object.keys(DATASETS) as DatasetName[]).map((value) => ({
  value,
  label: DATASETS[value].label,
}))
