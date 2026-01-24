import React, { createContext, useContext, useState } from "react"
import type { NavigationStateType } from "@/lib/navigation/types"

const NavigationContext = createContext<NavigationStateType | undefined>(
    undefined,
)

export function NavigationProvider({
    children,
}: {
    children: React.ReactNode
}) {
    const [originalImageUri, setOriginalImageUri] = useState<string | null>(
        null,
    )

    return (
        <NavigationContext.Provider
            value={{
                originalImageUri,
                setOriginalImageUri,
            }}
        >
            {children}
        </NavigationContext.Provider>
    )
}

export function useNavigationState() {
    const context = useContext(NavigationContext)
    if (!context) {
        throw new Error(
            "useNavigationState must be used within NavigationProvider",
        )
    }
    return context
}
