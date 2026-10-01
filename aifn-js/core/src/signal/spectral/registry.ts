/**
 * The functions of `aifn/signal/spectral`, registered with the notes they serve.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn/foundation/registry'
import * as inverse from './inverse'
import * as spectral from './spectral'

const fn = definer<FunctionInfo>('function', 'signal/spectral')
const STFT = ['short-time-fourier-transform', 'time-frequency-uncertainty']

fn(
  {
    key: 'periodogram',
    name: 'Periodogram',
    summary: 'The squared magnitude of the windowed DFT, scaled to a power spectral density.',
    role: 'estimator',
    returns: 'spectrum',
    notes: ['periodogram', 'spectral-leakage-and-windows', 'autocorrelation-and-wiener-khinchin'],
  },
  spectral.periodogram,
)
fn(
  {
    key: 'welch',
    name: "Welch's method",
    summary: 'The average of windowed periodograms of overlapping segments.',
    role: 'estimator',
    returns: 'spectrum',
    notes: ['welch-method', 'periodogram'],
    cite: ['welch1967'],
  },
  spectral.welch,
)
fn(
  {
    key: 'spectrogram',
    name: 'Spectrogram',
    role: 'estimator',
    returns: 'time-frequency',
    notes: [...STFT, 'mel-spectrogram'],
  },
  spectral.spectrogram,
)
fn(
  {
    key: 'stft',
    name: 'Short-time Fourier transform',
    role: 'transform',
    returns: 'time-frequency',
    notes: STFT,
    cite: ['allen1977', 'gabor1946'],
  },
  spectral.stft,
)
fn(
  {
    key: 'istft',
    name: 'Inverse STFT (overlap-add)',
    role: 'transform',
    returns: 'signal',
    notes: ['short-time-fourier-transform', 'overlap-add-and-overlap-save'],
    cite: ['griffin1984'],
  },
  inverse.istft,
)
fn(
  { key: 'checkCola', name: 'Constant overlap-add check', role: 'property', notes: ['overlap-add-and-overlap-save'] },
  inverse.checkCola,
)
fn(
  { key: 'checkNola', name: 'Nonzero overlap-add check', role: 'property', notes: ['overlap-add-and-overlap-save'] },
  inverse.checkNola,
)
fn(
  {
    key: 'dpss',
    name: 'Discrete prolate spheroidal sequences',
    summary: 'The Slepian tapers: the sequences most concentrated in a band.',
    role: 'construction',
    notes: ['multitaper-spectral-estimation'],
    cite: ['slepian1978'],
  },
  spectral.dpss,
)
fn(
  {
    key: 'multitaper',
    name: 'Multitaper spectral estimate',
    summary: 'The average of periodograms under orthogonal Slepian tapers.',
    role: 'estimator',
    returns: 'spectrum',
    notes: ['multitaper-spectral-estimation'],
    cite: ['thomson1982'],
  },
  spectral.multitaper,
)

/** The functions of the module, keyed by name. */
export const spectralFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', spectral, inverse) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
