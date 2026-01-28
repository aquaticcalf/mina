/**
 * Disease information data and utilities.
 *
 * Contains detailed information about each disease class that the model can detect,
 * including symptoms, treatments, and severity levels.
 */

import type { DiseaseClass, DiseaseInfo, Severity } from "@/lib/model/types"

/**
 * Complete disease information database.
 * Maps each DiseaseClass to its corresponding DiseaseInfo.
 */
export const DISEASE_INFO: Record<DiseaseClass, DiseaseInfo> = {
    bacterial_infection: {
        diseaseClass: "bacterial_infection",
        displayName: "Bacterial Infection",
        description:
            "Bacterial infections in fish are caused by harmful bacteria in the water. They can affect the skin, fins, gills, and internal organs. Common types include columnaris, fin rot, and ulcer disease.",
        symptoms: [
            "Red or inflamed patches on body",
            "Frayed or deteriorating fins",
            "White or gray patches on skin",
            "Ulcers or open sores",
            "Loss of appetite",
            "Lethargy or unusual swimming behavior",
        ],
        treatments: [
            "Quarantine affected fish immediately",
            "Perform 25-50% water change",
            "Add aquarium salt (1 tablespoon per 5 gallons)",
            "Use broad-spectrum antibiotic treatment",
            "Improve water quality and filtration",
            "Maintain stable water temperature",
        ],
        severity: "high",
    },
    fungal_infection: {
        diseaseClass: "fungal_infection",
        displayName: "Fungal Infection",
        description:
            "Fungal infections appear as cotton-like growths on the fish's body, fins, or mouth. They typically occur secondary to injury or stress and thrive in poor water conditions.",
        symptoms: [
            "White cotton-like growth on body or fins",
            "Fuzzy patches on skin",
            "Loss of color in affected areas",
            "Reduced activity",
            "Difficulty swimming",
            "Clamped fins",
        ],
        treatments: [
            "Quarantine infected fish",
            "Add antifungal medication to water",
            "Increase water temperature slightly (if species appropriate)",
            "Improve water quality with regular changes",
            "Add aquarium salt (dosage per product instructions)",
            "Remove any sharp decorations that could cause injury",
        ],
        severity: "medium",
    },
    healthy: {
        diseaseClass: "healthy",
        displayName: "Healthy",
        description:
            "Your fish appears to be in good health with no visible signs of disease or distress. Continue regular tank maintenance to keep your fish thriving.",
        symptoms: [
            "Bright, vibrant colors",
            "Clear, unblemished skin",
            "Active swimming behavior",
            "Good appetite",
            "Erect, intact fins",
            "Clear eyes",
        ],
        treatments: [
            "Maintain regular water change schedule",
            "Provide balanced, species-appropriate diet",
            "Monitor water parameters regularly",
            "Avoid overcrowding the tank",
            "Quarantine new fish before adding to main tank",
            "Keep tank clean and well-filtered",
        ],
        severity: "low",
    },
    parasite: {
        diseaseClass: "parasite",
        displayName: "Parasite Infection",
        description:
            "Parasitic infections are caused by external or internal parasites. Common types include ich (white spot disease), anchor worms, fish lice, and gill flukes. They can spread rapidly in aquarium conditions.",
        symptoms: [
            "White spots on body and fins (ich)",
            "Flashing or scratching against objects",
            "Rapid gill movement",
            "Visible parasites on body",
            "Weight loss despite eating",
            "Cloudy or bulging eyes",
        ],
        treatments: [
            "Raise water temperature gradually (for ich)",
            "Add anti-parasitic medication",
            "Increase aeration during treatment",
            "Perform daily water changes during treatment",
            "Remove visible parasites with tweezers (for larger parasites)",
            "Treat entire tank, not just affected fish",
        ],
        severity: "medium",
    },
    white_tail: {
        diseaseClass: "white_tail",
        displayName: "White Tail Disease",
        description:
            "White tail disease is a serious bacterial infection that causes the tail and posterior portion of the fish to turn white. It can progress rapidly and requires immediate attention.",
        symptoms: [
            "White discoloration starting at the tail",
            "Whitening spreading toward the head",
            "Frayed or deteriorating tail fin",
            "Loss of appetite",
            "Rapid breathing",
            "Lethargy and bottom-sitting",
        ],
        treatments: [
            "Immediate quarantine of affected fish",
            "Strong antibiotic treatment",
            "Daily 50% water changes",
            "Add aquarium salt to quarantine tank",
            "Maintain optimal water temperature",
            "Improve overall tank hygiene",
        ],
        severity: "high",
    },
}

/**
 * Get the bounding box color for a given disease class.
 * Colors follow the severity-based mapping:
 * - Green for healthy
 * - Yellow for low severity
 * - Red for medium/high severity
 *
 * @param diseaseClass - The disease class to get the color for
 * @returns Hex color string
 */
export function getBoundingBoxColor(diseaseClass: DiseaseClass): string {
    const info = DISEASE_INFO[diseaseClass]
    const severity = info.severity

    if (diseaseClass === "healthy") {
        return "#22c55e" // green-500
    }

    switch (severity) {
        case "low":
            return "#eab308" // yellow-500
        case "medium":
            return "#ef4444" // red-500
        case "high":
            return "#ef4444" // red-500
        default:
            return "#ef4444" // red-500 (fallback)
    }
}

/**
 * Get disease information for a specific disease class.
 *
 * @param diseaseClass - The disease class to look up
 * @returns DiseaseInfo object
 */
export function getDiseaseInfo(diseaseClass: DiseaseClass): DiseaseInfo {
    return DISEASE_INFO[diseaseClass]
}

/**
 * Get all available disease information entries.
 *
 * @returns Array of all DiseaseInfo objects
 */
export function getAllDiseaseInfo(): DiseaseInfo[] {
    return Object.values(DISEASE_INFO)
}

/**
 * Get display label for a disease class.
 * Converts "bacterial_infection" to "Bacterial Infection".
 *
 * @param diseaseClass - The disease class
 * @returns Formatted display name
 */
export function getDiseaseLabel(diseaseClass: string): string {
    return diseaseClass
        .split("_")
        .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
        .join(" ")
}
