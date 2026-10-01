/**
 * The functions of `aifn/signal/time-frequency`, registered with the notes they serve.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn/foundation/registry'
import * as cqt from './cqt'
import * as hilbert from './hilbert'

const fn = definer<FunctionInfo>('function', 'signal/time-frequency')

fn(
  {
    key: 'hilbert',
    name: 'Analytic signal (Hilbert transform)',
    summary: 'x + iH{x}, by zeroing the negative frequencies of the DFT.',
    role: 'transform',
    notes: ['hilbert-transform', 'instantaneous-frequency'],
  },
  hilbert.hilbert,
)
fn(
  {
    key: 'instantaneous',
    name: 'Instantaneous amplitude, phase and frequency',
    role: 'transform',
    notes: ['instantaneous-frequency', 'hilbert-transform'],
    cite: ['boashash1992'],
  },
  hilbert.instantaneous,
)
fn(
  { key: 'envelope', name: 'Envelope', role: 'transform', notes: ['hilbert-transform', 'instantaneous-frequency'] },
  hilbert.envelope,
)
fn(
  {
    key: 'hilbertSpectrum',
    name: 'Hilbert spectrum',
    summary: 'The instantaneous frequencies and amplitudes of a decomposition’s modes on a time–frequency grid.',
    role: 'transform',
    returns: 'time-frequency',
    notes: ['hilbert-huang-transform'],
    cite: ['huang1998'],
  },
  hilbert.hilbertSpectrum,
)
fn(
  {
    key: 'cqt',
    name: 'Constant-Q transform',
    role: 'transform',
    returns: 'time-frequency',
    notes: ['constant-q-transform', 'chroma-features'],
    cite: ['brown1991'],
  },
  cqt.cqt,
)

/** The functions of the module, keyed by name. */
export const timeFrequencyFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', hilbert, cqt) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
