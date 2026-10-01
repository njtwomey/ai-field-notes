/**
 * Vendor the manifold-of-fonts glyph data into aifn as a dataset (`aifn-applied/data/real/fonts`).
 *
 * The source of truth is the site's Python builder, `python/mlc/figures/fonts.py` (figure `manifold-of-fonts/glyphs`),
 * which writes `site/public/generated/figures/manifold-of-fonts/glyphs.json`. aifn must not import the site, so this
 * script copies that JSON into `aifn-js/applications/src/data/real/fonts/glyphs.ts` with a provenance header: the
 * pinned google/fonts commit, the licence of every font (from the path of its file in `fonts.py`: `ofl/` is the SIL
 * Open Font License 1.1, `apache/` the Apache License 2.0), the builder's cache hash and the repository commit of the
 * JSON. The integer outline vectors are stored as one base64 string of little-endian int16 (every value fits), which
 * keeps the module small and quick to parse; the rest is copied as is.
 *
 * To regenerate:
 *
 *   uv run mlc figures --only manifold-of-fonts
 *   node scripts/aifn-fonts.ts
 */
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

const root = path.resolve(import.meta.dirname, '..')
const dir = path.join(root, 'site', 'public', 'generated', 'figures', 'manifold-of-fonts')
const source = path.join(dir, 'glyphs.json')
const builder = path.join(root, 'python', 'mlc', 'figures', 'fonts.py')
const out = path.join(root, 'aifn-js', 'applications', 'src', 'data', 'real', 'fonts', 'glyphs.ts')

type Font = { family: string; style: string; cls: string; weight: number; width: number; italic: boolean }
type Data = {
  source: string
  cap_height: number
  display: string
  fonts: Font[]
  glyphs: { char: string; contours: number[]; offset: number; advance: number }[]
  vectors: number[][]
  handle: number
  anchor: number
  dropped: { font: string; reason: string }[]
}

const data = JSON.parse(fs.readFileSync(source, 'utf8')) as Data
const hash = fs.existsSync(path.join(dir, 'glyphs.hash'))
  ? fs.readFileSync(path.join(dir, 'glyphs.hash'), 'utf8').trim()
  : 'unknown'
let commit = 'uncommitted'
try {
  commit =
    execFileSync('git', ['log', '-1', '--format=%h', '--', path.relative(root, source)], { cwd: root })
      .toString()
      .trim() || 'uncommitted'
} catch {
  // Not a git checkout: leave the commit unknown.
}

// Licences by family: the google/fonts path of each file in the builder, e.g. "ofl/robotoslab/…" or "apache/…".
const py = fs.readFileSync(builder, 'utf8')
const fullCommit = /COMMIT = "([0-9a-f]+)"/.exec(py)?.[1] ?? data.source
const licenceOf = new Map<string, string>()
for (const m of py.matchAll(/"(ofl|apache|ufl)\/([a-z0-9]+)\//g))
  licenceOf.set(m[2], { ofl: 'OFL-1.1', apache: 'Apache-2.0', ufl: 'UFL-1.0' }[m[1]] ?? m[1])
const licence = (family: string) => {
  const l = licenceOf.get(family.toLowerCase().replace(/[^a-z0-9]/g, ''))
  if (!l) throw new Error(`aifn-fonts: no licence found for ${family} in fonts.py`)
  return l
}

const width = data.vectors[0].length
const bytes = new Uint8Array(data.vectors.length * width * 2)
const view = new DataView(bytes.buffer)
data.vectors.forEach((row, i) => {
  if (row.length !== width) throw new Error('aifn-fonts: ragged vectors')
  row.forEach((v, j) => {
    if (!Number.isInteger(v) || v < -32768 || v > 32767) throw new Error(`aifn-fonts: ${v} does not fit int16`)
    view.setInt16((i * width + j) * 2, v, true)
  })
})
const base64 = Buffer.from(bytes).toString('base64')

const fonts = data.fonts.map((f) => ({ ...f, licence: licence(f.family) }))
const json = (v: unknown) => JSON.stringify(v)

const text = `/**
 * Glyph outlines of ${data.fonts.length} real font instances in dense correspondence: the data of the manifold-of-fonts note,
 * vendored by \`scripts/aifn-fonts.ts\`. Do not edit by hand; regenerate with
 *
 *   uv run mlc figures --only manifold-of-fonts
 *   node scripts/aifn-fonts.ts
 *
 * Source: python/mlc/figures/fonts.py (figure manifold-of-fonts/glyphs), built from github.com/google/fonts at commit
 * ${fullCommit}; builder cache hash ${hash}; glyphs.json at repository commit ${commit}.
 * Licences: every font is under the SIL Open Font License 1.1 (OFL-1.1) or the Apache License 2.0, per font below.
 * The outlines are derived coordinates, not font software; the licences are recorded for attribution.
 *
 * \`VECTORS\` holds ${data.vectors.length} rows of ${width} int16 values (little-endian, base64): x, y of every sample of every
 * contour of every glyph, in font units with the cap height at ${data.cap_height}, then one advance width per glyph.
 */

/** The pinned google/fonts commit the fonts come from. */
export const SOURCE = ${json(`github.com/google/fonts@${fullCommit}`)}
export const CAP_HEIGHT = ${data.cap_height}
/** The word the note's figure sets. */
export const DISPLAY = ${json(data.display)}
/** Values per font vector. */
export const WIDTH = ${width}
/** Sample index (the display word's P, outer contour) of the foot-serif tip, and of a point held fixed while it moves. */
export const HANDLE = ${data.handle}
export const ANCHOR = ${data.anchor}

export const FONTS = ${json(fonts)} as const

export const GLYPHS: readonly { char: string; contours: readonly number[]; offset: number; advance: number }[] = ${json(data.glyphs)}

/** Fonts the builder dropped, and why. */
export const DROPPED: readonly { font: string; reason: string }[] = ${json(data.dropped)}

export const VECTORS = '${base64}'
`
fs.mkdirSync(path.dirname(out), { recursive: true })
fs.writeFileSync(out, text)
// The repository's formatter owns the layout of committed TypeScript.
execFileSync('npx', ['prettier', '--write', '--log-level', 'warn', out], { cwd: root, stdio: 'inherit' })
console.log(
  `aifn-fonts: wrote ${path.relative(root, out)} (${data.fonts.length} fonts × ${width} values, ${(text.length / 1024).toFixed(0)} KiB)`,
)
