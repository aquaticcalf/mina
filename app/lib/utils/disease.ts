export const getDiseaseLabel = (diseaseClass: string): string => {
    return diseaseClass
        .split("_")
        .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
        .join(" ")
}

export const getDiseaseColor = (diseaseClass: string): string => {
    switch (diseaseClass) {
        case "healthy":
            return "#22c55e" // green-500
        case "bacterial_infection":
            return "#ef4444" // red-500
        case "fungal_infection":
            return "#f97316" // orange-500
        case "parasite":
            return "#eab308" // yellow-500
        case "white_tail":
            return "#a855f7" // purple-500
        default:
            return "#6b7280" // gray-500
    }
}
