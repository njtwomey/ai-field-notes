/**
 * Build `aifn` (aifn-js/core) as a publishable package in `aifn-js/core/dist`, without touching the workspace: the
 * repo keeps importing the TypeScript source through the workspace `exports`, and `dist/` holds what a consumer
 * outside the repo installs.
 *
 * - JavaScript: one ES module entry per node of the module tree (the root, every family and every module, from
 *   aifn-js/modules.json), bundled by Vite's library mode with shared chunks, so a consumer that imports several
 *   modules loads each primitive (and its registration) once.
 * - Declarations: `tsc` with `emitDeclarationOnly` over the source, into `dist/types`.
 * - `dist/package.json`: the source package.json with `private` removed, `files`, `sideEffects` (importing a module
 *   registers its primitives, so every file counts) and `exports` mapping each entry to `{ types, import }`.
 *
 * Usage: `node scripts/aifn-package.ts [--version x.y.z]` (or `make aifn-package`). Publish with
 * `npm publish aifn-js/core/dist` once the version is set.
 */

import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import { build } from 'vite'

const root = path.join(import.meta.dirname, '..')
const core = path.join(root, 'aifn-js', 'core')
const src = path.join(core, 'src')
const dist = path.join(core, 'dist')

type Module = { module: string }
type Family = { family: string; modules: Module[] }
const tree = JSON.parse(fs.readFileSync(path.join(root, 'aifn-js', 'modules.json'), 'utf8')) as {
  core: { families: Family[] }
}

// Entry names are the import paths without the package name ('' for the root).
const entries: Record<string, string> = { index: path.join(src, 'index.ts') }
for (const f of tree.core.families) {
  const family = path.join(src, f.family, 'index.ts')
  if (fs.existsSync(family)) entries[`${f.family}/index`] = family
  for (const m of f.modules) {
    const file = path.join(src, f.family, m.module, 'index.ts')
    if (fs.existsSync(file)) entries[`${f.family}/${m.module}/index`] = file
  }
}

const version = process.argv.includes('--version') ? process.argv[process.argv.indexOf('--version') + 1] : null

fs.rmSync(dist, { recursive: true, force: true })

await build({
  configFile: false,
  logLevel: 'warn',
  build: {
    outDir: dist,
    emptyOutDir: false,
    target: 'es2023',
    minify: false,
    sourcemap: true,
    lib: { entry: entries, formats: ['es'] },
    rollupOptions: {
      output: { entryFileNames: '[name].js', chunkFileNames: 'chunks/[name]-[hash].js' },
    },
  },
})

// Declarations from the source, with the workspace's compiler options.
const tsconfig = path.join(dist, 'tsconfig.types.json')
fs.writeFileSync(
  tsconfig,
  JSON.stringify(
    {
      extends: path.join(root, 'tsconfig.aifn.json'),
      compilerOptions: {
        noEmit: false,
        emitDeclarationOnly: true,
        declaration: true,
        declarationDir: path.join(dist, 'types'),
        rootDir: src,
        tsBuildInfoFile: null,
        incremental: false,
        composite: false,
      },
      include: [src],
      exclude: [],
    },
    null,
    2,
  ),
)
execFileSync('npx', ['tsc', '-p', tsconfig], { cwd: root, stdio: 'inherit' })
fs.rmSync(tsconfig)

const pkg = JSON.parse(fs.readFileSync(path.join(core, 'package.json'), 'utf8')) as Record<string, unknown>
const exportsMap: Record<string, { types: string; import: string }> = {}
for (const name of Object.keys(entries)) {
  const sub = name === 'index' ? '.' : `./${name.replace(/\/index$/, '')}`
  exportsMap[sub] = { types: `./types/${name}.d.ts`, import: `./${name}.js` }
}
const { private: _private, ...rest } = pkg
const out = {
  ...rest,
  ...(version ? { version } : {}),
  sideEffects: true,
  files: ['**/*.js', '**/*.js.map', 'types'],
  exports: exportsMap,
}
fs.writeFileSync(path.join(dist, 'package.json'), JSON.stringify(out, null, 2) + '\n')
console.log(`aifn-package: ${Object.keys(entries).length} entries → ${path.relative(root, dist)}`)
