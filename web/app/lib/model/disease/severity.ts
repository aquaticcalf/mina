/**
 * Utilities for working with severity levels and metadata.
 */

import type { Severity } from "@/lib/model/types"
import type { Detection } from "@/lib/model/types"
import { AlertCircle, AlertTriangle, CheckCircle } from "lucide-react"
import { DISEASE_INFO } from "./info"

export interface SeverityMeta {
  label: string
  color: string
  bg: string
  border: string
  icon: typeof CheckCircle | typeof AlertTriangle | typeof AlertCircle
}

/**
 * Get display metadata for a severity level.
 * Returns color, background, border, icon, and label for UI rendering.
 */
export function getSeverityMeta(severity: Severity): SeverityMeta {
  switch (severity) {
    case "healthy":
      return {
        label: "Healthy",
        color: "var(--healthy)",
        bg: "var(--healthy-bg)",
        border: "var(--healthy-border)",
        icon: CheckCircle,
      }
    case "low":
      return {
        label: "Low",
        color: "var(--low)",
        bg: "var(--low-bg)",
        border: "var(--low-border)",
        icon: AlertTriangle,
      }
    case "medium":
      return {
        label: "Medium",
        color: "var(--medium)",
        bg: "var(--medium-bg)",
        border: "var(--medium-border)",
        icon: AlertTriangle,
      }
    case "high":
      return {
        label: "High",
        color: "var(--high)",
        bg: "var(--high-bg)",
        border: "var(--high-border)",
        icon: AlertCircle,
      }
  }
}

/**
 * Get the worst (highest) severity from an array of detections.
 * Severity order: high > medium > low > healthy
 */
export function getWorstSeverity(detections: Detection[]): Severity {
  const order: Severity[] = ["high", "medium", "low", "healthy"]
  for (const severity of order) {
    const hasSeverity = detections.some((det) => {
      const info = DISEASE_INFO[det.diseaseClass]
      return info.severity === severity
    })
    if (hasSeverity) return severity
  }
  return "healthy"
}
