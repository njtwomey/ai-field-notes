/** The registry of `aifn-applied/information/coding`: source codes and code bounds. */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn/foundation/registry'
import * as coding from './coding'

type Table<I extends AlgorithmInfo | FunctionInfo> = Readonly<Record<string, Entry<(...args: never[]) => unknown, I>>>
const fn = definer<FunctionInfo>('function', 'information/coding')
const SRC = ['source-coding-theorem', 'entropy']

definer<AlgorithmInfo>('algorithm', 'information/coding')(
  {
    key: 'huffmanSteps',
    name: 'Huffman coding',
    summary: 'Merge the two least probable nodes per step until one tree remains.',
    problem: 'graph',
    state: { iterate: 'nodes', flags: [] },
    notes: SRC,
    cite: ['huffman1952'],
  },
  coding.huffmanSteps,
)
fn(
  {
    key: 'huffmanTree',
    name: 'Huffman tree',
    role: 'construction',
    returns: 'tree',
    notes: SRC,
    cite: ['huffman1952'],
  },
  coding.huffmanTree,
)
fn(
  { key: 'huffmanCode', name: 'Huffman code', role: 'construction', notes: SRC, cite: ['huffman1952'] },
  coding.huffmanCode,
)
fn(
  {
    key: 'shannonCode',
    name: 'Shannon code',
    summary: 'Codeword lengths ⌈−log₂ p⌉.',
    role: 'construction',
    notes: SRC,
    cite: ['shannon1948'],
  },
  coding.shannonCode,
)
fn({ key: 'shannonFanoCode', name: 'Shannon–Fano code', role: 'construction', notes: SRC }, coding.shannonFanoCode)
fn(
  { key: 'arithmeticInterval', name: 'Arithmetic-coding interval', role: 'transform', notes: SRC },
  coding.arithmeticInterval,
)
fn(
  {
    key: 'kraftSum',
    name: 'Kraft sum',
    tex: '\\sum_i 2^{-\\ell_i}',
    role: 'property',
    notes: SRC,
    cite: ['cover2006'],
  },
  coding.kraftSum,
)
fn(
  { key: 'hammingDistance', name: 'Hamming distance', role: 'property', cite: ['hamming1950'] },
  coding.hammingDistance,
)
fn({ key: 'hammingWeight', name: 'Hamming weight', role: 'property' }, coding.hammingWeight)
fn({ key: 'minimumDistance', name: 'Minimum distance of a code', role: 'property' }, coding.minimumDistance)
fn(
  { key: 'hammingBound', name: 'Hamming (sphere-packing) bound', role: 'property', cite: ['hamming1950'] },
  coding.hammingBound,
)
fn({ key: 'singletonBound', name: 'Singleton bound', role: 'property' }, coding.singletonBound)
fn({ key: 'plotkinBound', name: 'Plotkin bound', role: 'property' }, coding.plotkinBound)

/** The algorithms of the module, keyed by factory name. */
export const codingAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>(
  'algorithm',
  coding,
) as Table<AlgorithmInfo>
/** The functions of the module, keyed by name. */
export const codingFunctions: Table<FunctionInfo> = entries<FunctionInfo>('function', coding) as Table<FunctionInfo>
