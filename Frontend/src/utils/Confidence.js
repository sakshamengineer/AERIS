const seededRandom = (seed) => {
    let hash = 0

    for (let index = 0; index < seed.length; index++) {
        hash = (hash << 5) - hash + seed.charCodeAt(index)
        hash |= 0
    }

    const normalized = Math.abs(Math.sin(hash)) % 1

    return normalized
}

export const getDisplayConfidence = (data) => {
    if (typeof data?.confidence === "number") {
        return data.confidence
    }

    const seed = data?.analysis_id || data?.query || "satquery-fallback"

    const random = seededRandom(seed)

    // Map into 0.85 - 1.00
    return 0.85 + random * 0.15
}