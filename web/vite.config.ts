import path from "path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"
import { VitePWA } from "vite-plugin-pwa"
import { viteStaticCopy } from "vite-plugin-static-copy"
import { Readable } from "stream"

// Custom plugin to reliably proxy GitHub Releases during local dev
// Bypasses `http-proxy` socket hang up issues with cross-domain redirects
const localModelProxy = () => ({
  name: "local-model-proxy",
  configureServer(server: any) {
    server.middlewares.use(async (req: any, res: any, next: any) => {
      if (req.url?.startsWith("/api/model-proxy")) {
        try {
          const url = new URL(`http://localhost${req.url}`)
          const file = url.searchParams.get("file")
          if (!file) {
            res.statusCode = 400
            return res.end("Missing file param")
          }

          const githubUrl = `https://github.com/fishcareyolo/fishcareyolo/releases/download/prod/${file}`
          const response = await fetch(githubUrl)
          
          if (!response.ok) {
            res.statusCode = response.status
            return res.end(`Failed to fetch from GitHub: ${response.status}`)
          }

          res.setHeader("Access-Control-Allow-Origin", "*")
          res.setHeader("Content-Type", "application/octet-stream")
          
          // Stream using Node.js Readable from Web API ReadableStream
          if (response.body) {
            // @ts-ignore
            Readable.fromWeb(response.body).pipe(res)
          } else {
            res.end()
          }
        } catch (e: any) {
          res.statusCode = 500
          res.end(e.message)
        }
      } else {
        next()
      }
    })
  },
})

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    localModelProxy(),
    viteStaticCopy({
      targets: [
        {
          // Copy ONNX Runtime WASM and MJS files to build root
          // The worker sets ort.env.wasm.wasmPaths = "/" so these must be at root level
          src: "node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded*.{wasm,mjs}",
          dest: "",
          rename: { stripBase: true },
        },
      ],
    }),
    VitePWA({
      registerType: "autoUpdate",
      includeAssets: ["favicon.ico", "apple-touch-icon.png", "masked-icon.svg"],
      manifest: {
        name: "mina",
        short_name: "mina",
        description: "fish disease detection using yolo model",
        theme_color: "#ffffff",
        background_color: "#ffffff",
        display: "standalone",
        start_url: "/",
        icons: [
          {
            src: "pwa-192x192.png",
            sizes: "192x192",
            type: "image/png",
          },
          {
            src: "pwa-512x512.png",
            sizes: "512x512",
            type: "image/png",
          },
          {
            src: "pwa-512x512.png",
            sizes: "512x512",
            type: "image/png",
            purpose: "any maskable",
          },
        ],
      },
      devOptions: {
        enabled: true,
      },
      workbox: {
        maximumFileSizeToCacheInBytes: 30 * 1024 * 1024,
        globPatterns: ["**/*.{js,mjs,css,html,ico,png,svg,woff2,wasm}"],
      },
    }),
  ],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./app"),
    },
  },
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
  },
})
