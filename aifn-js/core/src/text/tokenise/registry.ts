/** The functions of `aifn/text/tokenise`, registered with the notes that define them. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn/foundation/registry'
import * as t from './tokenise'

const fn = definer<FunctionInfo>('function', 'text/tokenise')
const notes = ['tokenisation']

fn(
  {
    key: 'tokenise',
    name: 'Regular-expression tokeniser',
    role: 'transform',
    returns: 'tokens',
    summary: 'Tokens and their offsets by a regular expression: words, words and punctuation, white space, GPT-2.',
    notes,
    cite: ['jurafsky2025', 'radford2019'],
  },
  t.tokenise,
)
fn(
  {
    key: 'whitespaceTokenise',
    name: 'White-space tokeniser',
    role: 'transform',
    returns: 'tokens',
    notes,
  },
  t.whitespaceTokenise,
)
fn(
  {
    key: 'characterTokenise',
    name: 'Character tokeniser',
    role: 'transform',
    returns: 'tokens',
    summary: 'One token per code point or per grapheme cluster, with offsets.',
    notes: ['tokenisation', 'character-n-grams-and-shingles'],
  },
  t.characterTokenise,
)
fn(
  {
    key: 'detokenise',
    name: 'Detokeniser',
    role: 'transform',
    summary: 'Tokens back to text: exactly from offsets, or by spacing rules for punctuation and clitics.',
    notes,
  },
  t.detokenise,
)

/** The functions of the module, keyed by name. */
export const tokeniseFunctions = entries<FunctionInfo>('function', t) as Readonly<
  Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
>
