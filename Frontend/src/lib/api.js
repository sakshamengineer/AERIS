export const API_BASE_URL =
    (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "")

async function handleResponse(response) {
    let data = null

    try {
        data = await response.json()
    } catch (error) {
        data = null
    }

    if (!response.ok) {
        const message =
            data?.detail ||
            data?.error ||
            `Request failed (${response.status})`

        throw new Error(message)
    }

    return data
}


export async function startAnalysis({ query, files, modalities, task }) {
    const formData = new FormData()

    formData.append("query", query)

    files.forEach((file) => {
        formData.append("images", file)
    })

    formData.append(
        "modalities",
        JSON.stringify(modalities || [])
    )

    // Forward-compatible: the backend doesn't read this field yet
    // (task is auto-routed server-side), but sending it now means
    // no frontend change is needed once manual task override lands.
    if (task) {
        formData.append("task", task)
    }

    const response = await fetch(`${API_BASE_URL}/analysis`, {
        method: "POST",
        body: formData
    })

    return handleResponse(response)
}

export async function inspectAnalysis({ query, files, modalities }) {
    const formData = new FormData()

    formData.append("query", query || "")

    files.forEach((file) => {
        formData.append("images", file)
    })

    formData.append(
        "modalities",
        JSON.stringify(modalities || [])
    )

    const response = await fetch(`${API_BASE_URL}/analysis/inspect`, {
        method: "POST",
        body: formData
    })

    return handleResponse(response)
}

export async function getAnalysisStatus(analysisId) {
    const response = await fetch(
        `${API_BASE_URL}/analysis/${analysisId}/status`
    )

    return handleResponse(response)
}

export async function getAnalysisResult(analysisId) {
    const response = await fetch(
        `${API_BASE_URL}/analysis/${analysisId}`
    )

    return handleResponse(response)
}

export async function getDashboard() {
    const response = await fetch(`${API_BASE_URL}/dashboard`)

    return handleResponse(response)
}

export async function getModelMetrics() {
    const response = await fetch(`${API_BASE_URL}/models/metrics`)

    return handleResponse(response)
}
export async function runModelMetrics() {
    const response = await fetch(`${API_BASE_URL}/models/metrics/run`, {
        method: "POST"
    })

    return handleResponse(response)
}

export async function getModelMetricsStatus() {
    const response = await fetch(`${API_BASE_URL}/models/metrics/status`)

    return handleResponse(response)
}

