import { StrictMode } from "react"
import { createRoot } from "react-dom/client"

import "./index.css"
import App from "./routes"
import { ThemeProvider } from "@/components/theme-provider"
import { CameraProvider } from "@/lib/camera/context"
import { DetectionProvider } from "@/lib/detection/context"
import { ModelLoaderProvider } from "@/lib/inference/model-loader-context"
import { initHistoryDB } from "@/lib/history"

// Initialize IndexedDB for history storage
initHistoryDB().catch((err) => {
  console.error("Failed to initialize history database:", err)
})

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ThemeProvider>
      <CameraProvider>
        <DetectionProvider>
          <ModelLoaderProvider>
            <App />
          </ModelLoaderProvider>
        </DetectionProvider>
      </CameraProvider>
    </ThemeProvider>
  </StrictMode>,
)
