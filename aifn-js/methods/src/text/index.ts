/**
 * `aifn-applied/text`: applications of text processing on `aifn/text`. Children: corpora (toy and worked-example
 * corpora) and tokenisers (a trained suite of named tokenisers to compare).
 */

export { namedCorpus, toyCorpus, type Corpus } from './corpora'
export { tokeniserSuite, TOKENISER_KINDS, type TokeniserKind } from './tokenisers'
