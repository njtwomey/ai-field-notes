/**
 * `aifn/text`: text processing, from characters to features. Children: normalise (Unicode forms, case folding,
 * accents), tokenise (regular-expression and character tokenisers with offsets), stem (Porter, stop words),
 * vocabulary (token ↔ id), subword (BPE, WordPiece and unigram-LM training as step-through algorithms, and encoding),
 * features (n-grams, bag of words, TF-IDF, BM25, feature hashing) and cooccurrence (windowed counts, PMI, PPMI, SVD
 * word vectors).
 */

export { normalise } from './normalise'
export { tokenise, detokenise, type Tokenisation } from './tokenise'
export { buildVocabulary, encodeTokens, decodeTokens, type Vocabulary } from './vocabulary'
export { bagOfWords, tfidf, bm25 } from './features'
