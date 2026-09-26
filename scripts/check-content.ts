/** `npm run check:content`: validate content/ without starting Vite. */
import path from 'node:path'
import { buildIndex } from '../plugins/content-index.ts'

const { notes, references } = buildIndex(path.resolve(import.meta.dirname, '..', 'content'))
console.log(`✓ ${notes.length} notes, ${Object.keys(references).length} references`)
