import type { ComponentType } from "react"
import { BrowserRouter as Router, Routes, Route } from "react-router-dom"
import { FileSystemRouter } from "file-system-router"
import { AppShell } from "@/components/layout/app-shell"

const pages = import.meta.glob("./pages/**/*.tsx", { eager: true }) as Record<
  string,
  { default: ComponentType }
>

export default function App() {
  return (
    <Router>
      <Routes>
        <Route element={<AppShell />}>
          <Route path="*" element={<FileSystemRouter pages={pages} />} />
        </Route>
      </Routes>
    </Router>
  )
}
