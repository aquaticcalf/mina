import React, { createContext, useContext, useState, useEffect } from "react"

interface CameraState {
  capturedImage: Blob | null
  setCapturedImage: (img: Blob | null) => void
  capturedImageUrl: string | null
  cameraFacingMode: "environment" | "user"
  setCameraFacingMode: (mode: "environment" | "user") => void
}

const CameraContext = createContext<CameraState | null>(null)

export function CameraProvider({ children }: { children: React.ReactNode }) {
  const [capturedImage, setCapturedImageState] = useState<Blob | null>(null)
  const [capturedImageUrl, setCapturedImageUrl] = useState<string | null>(null)
  const [cameraFacingMode, setCameraFacingModeState] = useState<"environment" | "user">(() => {
    return (localStorage.getItem("mina_camera_facing") as "environment" | "user") || "environment"
  })

  // Create/revoke object URLs when captured image changes
  useEffect(() => {
    if (capturedImage) {
      const url = URL.createObjectURL(capturedImage)
      setCapturedImageUrl(url)
      return () => URL.revokeObjectURL(url)
    } else {
      setCapturedImageUrl(null)
    }
  }, [capturedImage])

  const setCapturedImage = (img: Blob | null) => {
    setCapturedImageState(img)
  }

  const setCameraFacingMode = (mode: "environment" | "user") => {
    setCameraFacingModeState(mode)
    localStorage.setItem("mina_camera_facing", mode)
  }

  return (
    <CameraContext.Provider
      value={{
        capturedImage,
        setCapturedImage,
        capturedImageUrl,
        cameraFacingMode,
        setCameraFacingMode,
      }}
    >
      {children}
    </CameraContext.Provider>
  )
}

export function useCameraContext() {
  const ctx = useContext(CameraContext)
  if (!ctx) throw new Error("useCameraContext must be inside CameraProvider")
  return ctx
}
