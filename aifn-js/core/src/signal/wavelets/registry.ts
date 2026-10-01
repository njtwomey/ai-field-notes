/**
 * The registry of wavelets (design S §2.13): the orthogonal Daubechies family (Haar = db1 to db10), each as a function
 * returning its four filters with its vanishing moments and filter length, and the continuous Morlet wavelet with its
 * centre frequency. The "wavelet families" note and the lab's wavelet picker enumerate this table.
 */

import { definer, entries, type Entry, type WaveletInfo } from 'aifn/foundation/registry'
import { real, space } from 'aifn/foundation/space'
import { morlet, waveletFilters, type WaveletName } from './wavelets'

const define = definer<WaveletInfo>('wavelet', 'signal/wavelets')
const none = space({})
const orthogonalNotes = ['wavelet-families', 'discrete-wavelet-transform']

const daubechies = (n: number): Entry<() => ReturnType<typeof waveletFilters>, WaveletInfo> => {
  const key = `db${n}` as WaveletName
  return define(
    {
      key,
      name: `Daubechies ${n}`,
      family: 'daubechies',
      continuous: false,
      orthogonal: true,
      vanishingMoments: n,
      taps: 2 * n,
      params: none,
      cite: ['daubechies1988', 'mallat1989'],
      notes: orthogonalNotes,
    },
    () => waveletFilters(key),
  )
}

/** Every wavelet, keyed by the name the transforms take (`haar`, `db1` … `db10`, `morlet`). */
export const waveletRegistry: Readonly<Record<string, Entry<(...args: never[]) => unknown, WaveletInfo>>> =
  entries<WaveletInfo>('wavelet', {
    haar: define(
      {
        key: 'haar',
        name: 'Haar',
        family: 'haar',
        continuous: false,
        orthogonal: true,
        vanishingMoments: 1,
        taps: 2,
        params: none,
        cite: ['mallat1989'],
        notes: orthogonalNotes,
      },
      () => waveletFilters('haar'),
    ),
    ...Object.fromEntries([1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((n) => [`db${n}`, daubechies(n)])),
    morlet: define(
      {
        key: 'morlet',
        name: 'Morlet',
        family: 'morlet',
        continuous: true,
        orthogonal: false,
        params: space({
          omega0: real(4, 12, { default: 6, label: '\\omega_0', doc: 'centre frequency (radians per unit scale)' }),
        }),
        cite: ['torrence1998'],
        notes: ['continuous-wavelet-transform', 'wavelet-families'],
      },
      morlet,
    ),
  }) as Readonly<Record<string, Entry<(...args: never[]) => unknown, WaveletInfo>>>
