/**
 * The registry of filter design methods (design S §2.13): each design function with its family (FIR or IIR), the band
 * types it designs, the specification fields it honours and its parameters. The FIR and IIR design notes and the
 * lab's filter picker enumerate this table.
 */

import { definer, entries, type Entry, type FilterDesignInfo, type FunctionInfo } from 'aifn/foundation/registry'
import { int, oneOf, real, space } from 'aifn/foundation/space'
import * as filters from './filters'

const define = definer<FilterDesignInfo>('filter-design', 'signal/filters')
const bands = ['lowpass', 'highpass', 'bandpass', 'bandstop'] as const
const order = int(1, 12, { default: 4, label: 'n', doc: 'filter order' })
const cutoff = real(0.01, 0.99, { default: 0.25, label: 'W_n', doc: 'edge, as a fraction of the Nyquist frequency' })
const btype = oneOf(bands, { doc: 'band type' })
const iirNotes = ['infinite-impulse-response-filter-design']
const cite = ['oppenheim2010']

define(
  {
    key: 'butter',
    name: 'Butterworth',
    summary: 'A maximally flat passband; the edge is the −3 dB point.',
    family: 'iir',
    bands,
    honours: ['order', 'cutoff'],
    params: space({ order, cutoff, btype }),
    cite,
    notes: iirNotes,
  },
  filters.butter,
)
define(
  {
    key: 'cheby1',
    name: 'Chebyshev type I',
    summary: 'An equiripple passband of a given ripple and a monotone stopband.',
    family: 'iir',
    bands,
    honours: ['order', 'cutoff', 'passRippleDb'],
    params: space({
      order,
      passRippleDb: real(0.01, 6, { default: 1, label: 'r_p', unit: 'dB', doc: 'passband ripple' }),
      cutoff,
      btype,
    }),
    cite,
    notes: iirNotes,
  },
  filters.cheby1,
)
define(
  {
    key: 'cheby2',
    name: 'Chebyshev type II',
    summary: 'A monotone passband and an equiripple stopband a given attenuation down.',
    family: 'iir',
    bands,
    honours: ['order', 'cutoff', 'stopAttenDb'],
    params: space({
      order,
      stopAttenDb: real(10, 120, { default: 40, label: 'r_s', unit: 'dB', doc: 'stopband attenuation' }),
      cutoff,
      btype,
    }),
    cite,
    notes: iirNotes,
  },
  filters.cheby2,
)
define(
  {
    key: 'firwin',
    name: 'FIR window method',
    summary: 'A linear-phase FIR filter: the ideal impulse response truncated by a window.',
    family: 'fir',
    bands,
    honours: ['numtaps', 'cutoff', 'window'],
    params: space({
      numtaps: int(3, 255, { default: 31, label: 'N', doc: 'number of taps (odd for high-pass and band-stop)' }),
      cutoff,
      window: oneOf(['hamming', 'hann', 'blackman', 'rectangular', 'kaiser'], {
        doc: 'window (see the window registry)',
      }),
    }),
    cite,
    notes: ['finite-impulse-response-filter-design'],
  },
  filters.firwin,
)

/** Every filter design method, keyed by function name. */
export const filterDesignRegistry: Readonly<Record<string, Entry<(...args: never[]) => unknown, FilterDesignInfo>>> =
  entries<FilterDesignInfo>('filter-design', filters) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FilterDesignInfo>>
  >

const fn = definer<FunctionInfo>('function', 'signal/filters')
const IIR = ['infinite-impulse-response-filters', 'difference-equations']

fn(
  {
    key: 'iirfilter',
    name: 'IIR filter design',
    role: 'construction',
    returns: 'lti',
    notes: ['infinite-impulse-response-filter-design'],
  },
  filters.iirfilter,
)
fn(
  {
    key: 'kaiserBeta',
    name: 'Kaiser β for an attenuation',
    role: 'construction',
    notes: ['finite-impulse-response-filter-design', 'spectral-leakage-and-windows'],
  },
  filters.kaiserBeta,
)
fn(
  {
    key: 'kaiserAttenuation',
    name: 'Kaiser window attenuation',
    role: 'property',
    notes: ['finite-impulse-response-filter-design'],
  },
  filters.kaiserAttenuation,
)
fn(
  {
    key: 'kaiserOrder',
    name: 'Kaiser window order',
    role: 'construction',
    notes: ['finite-impulse-response-filter-design'],
  },
  filters.kaiserOrder,
)
fn(
  {
    key: 'lfilter',
    name: 'Linear filter (difference equation)',
    summary: 'y = (b/a) * x by the transposed direct form II, with initial conditions.',
    role: 'transform',
    notes: [...IIR, 'finite-impulse-response-filters', 'linear-time-invariant-systems'],
    cite: ['oppenheim2010'],
  },
  filters.lfilter,
)
fn({ key: 'sosfilt', name: 'Filter by second-order sections', role: 'transform', notes: IIR }, filters.sosfilt)
fn(
  { key: 'lfilterZi', name: 'Steady-state initial conditions', role: 'construction', notes: ['zero-phase-filtering'] },
  filters.lfilterZi,
)
fn(
  {
    key: 'sosfiltZi',
    name: 'Steady-state initial conditions (sections)',
    role: 'construction',
    notes: ['zero-phase-filtering'],
  },
  filters.sosfiltZi,
)
fn(
  {
    key: 'filtfilt',
    name: 'Zero-phase filtering',
    summary: 'Filter forwards then backwards: the magnitude response squared and zero phase.',
    role: 'transform',
    notes: ['zero-phase-filtering'],
  },
  filters.filtfilt,
)
fn(
  {
    key: 'freqz',
    name: 'Digital frequency response',
    role: 'property',
    returns: 'spectrum',
    notes: ['frequency-response', 'discrete-time-fourier-transform'],
  },
  filters.freqz,
)
fn(
  {
    key: 'groupDelay',
    name: 'Group delay',
    tex: '\\tau_g(\\omega)',
    role: 'property',
    notes: ['linear-phase-and-group-delay'],
  },
  filters.groupDelay,
)

/** The functions of the module, keyed by name. */
export const filtersFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', filters) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
