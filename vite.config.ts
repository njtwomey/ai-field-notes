import path from 'node:path'
import mdx from '@mdx-js/rollup'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import { contentIndex } from './plugins/content-index.ts'
import { mdxOptions } from './plugins/mdx-options.ts'

const contentDir = path.resolve(import.meta.dirname, 'content')

export default defineConfig({
  root: 'site',
  // Served from https://<user>.github.io/ai-field-notes/. Change here only; the router and asset URLs read import.meta.env.BASE_URL.
  base: '/ai-field-notes/',
  build: {
    outDir: '../dist',
    emptyOutDir: true,
    // ECharts alone is ~640 kB minified (215 kB gzip) and is split into its own cached chunk.
    chunkSizeWarningLimit: 700,
    rolldownOptions: {
      output: {
        // Vendor chunks change rarely, so browsers keep them cached across content updates.
        codeSplitting: {
          groups: [
            { name: 'echarts', test: /node_modules[\\/](echarts|zrender)/ },
            { name: 'react', test: /node_modules[\\/](react|react-dom|react-router|scheduler)[\\/]/ },
            { name: 'ui', test: /node_modules[\\/](@base-ui|cmdk|lucide-react|@floating-ui)/ },
          ],
        },
      },
    },
  },
  plugins: [
    {
      enforce: 'pre',
      ...mdx(mdxOptions),
    },
    react({ include: /\.(mdx|tsx|ts)$/ }),
    tailwindcss(),
    contentIndex({ contentDir }),
    {
      // The Code tab globs python/mlc/examples, which lies outside Vite's root and so is not watched by default: without
      // this, a new example needs a dev-server restart before its files appear.
      name: 'watch-python-examples',
      configureServer(server) {
        server.watcher.add(path.resolve(import.meta.dirname, 'python/mlc/examples'))
      },
    },
  ],
  resolve: {
    alias: {
      '@': path.resolve(import.meta.dirname, 'site/src'),
      '@content': contentDir,
      '@design': path.resolve(import.meta.dirname, 'design'),
      '@python': path.resolve(import.meta.dirname, 'python'),
      '@render': path.resolve(import.meta.dirname, 'aifn-js/render/src'),
      'aifn-render': path.resolve(import.meta.dirname, 'aifn-js/render/src'),
      'aifn-applied': path.resolve(import.meta.dirname, 'aifn-js/methods/src'),
      'aifn': path.resolve(import.meta.dirname, 'aifn-js/core/src'),
    },
  },
  server: { fs: { allow: [import.meta.dirname] } },
})
