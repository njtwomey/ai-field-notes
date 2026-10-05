import path from 'node:path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import { engineDeps } from '../plugins/engine-deps.ts'

const src = path.join(import.meta.dirname, 'src')

/**
 * The aifn lab: a standalone app for exploring the aifn engine. It imports the released packages by name
 * (aifn-compute, aifn-methods, aifn-render, from node_modules), its own folder through `@lab/*`, and third-party
 * packages; nothing is imported from the site.
 */
export default defineConfig({
  root: import.meta.dirname,
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: [{ find: /^@lab\//, replacement: `${src}/` }],
  },
  // The engine packages are pre-bundled whole at start-up (plugins/engine-deps.ts says why).
  optimizeDeps: { include: engineDeps(path.resolve(import.meta.dirname, '..')) },
  // A single-page app with path URLs (/<module>/<specimen>): dev and preview serve index.html for any unknown path.
  appType: 'spa',
  server: { port: 5190 },
  preview: { port: 5190 },
})
