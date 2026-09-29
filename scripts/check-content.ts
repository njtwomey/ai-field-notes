/** `npm run check:content`: validate content/ without starting Vite. */
import path from 'node:path'
import { buildIndex } from '../plugins/content-index.ts'

const { notes, references, glossary } = buildIndex(path.resolve(import.meta.dirname, '..', 'content'))
console.log(
  `✓ ${notes.length} notes, ${Object.keys(references).length} references, ${Object.keys(glossary).length} glossary entries`,
)
