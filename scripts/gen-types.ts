/**
 * `make contracts`: compile site/src/generated/contracts.schema.json (written by `uv run mlc schema` from the pydantic
 * contracts) into site/src/generated/contracts.ts. Python is the source of truth; never edit the output by hand.
 */
import fs from 'node:fs'
import path from 'node:path'
import { compile, type JSONSchema } from 'json-schema-to-typescript'

const dir = path.resolve(import.meta.dirname, '..', 'site', 'src', 'generated')
const schema = JSON.parse(fs.readFileSync(path.join(dir, 'contracts.schema.json'), 'utf8')) as JSONSchema

const ts = await compile(schema, 'Contracts', {
  bannerComment:
    '/* Generated from python/mlc/core/contracts.py via contracts.schema.json by `make contracts`. Do not edit. */',
  additionalProperties: false,
  unreachableDefinitions: true,
  style: { printWidth: 120, semi: false, singleQuote: true, trailingComma: 'all' },
})
// json-schema-to-typescript capitalises letters after digits (PointCloud2d → PointCloud2D). Restore pydantic names.
let out = ts
for (const name of Object.keys((schema.$defs ?? {}) as Record<string, unknown>)) {
  const mangled = name.replace(/(\d)([a-z])/g, (_, d: string, c: string) => d + c.toUpperCase())
  if (mangled !== name) out = out.replace(new RegExp(`\\b${mangled}\\b`, 'g'), name)
}
fs.writeFileSync(path.join(dir, 'contracts.ts'), out)
console.log('✓ wrote site/src/generated/contracts.ts')
