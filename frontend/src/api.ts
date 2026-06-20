import type {
  BatchEvaluationResult,
  ProviderId,
  ProviderResponse,
  RecognitionResult,
  RemoteSettings,
} from "./types";

export const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const text = await response.text();
    let detail = text || response.statusText;
    try {
      const parsed = JSON.parse(text) as { detail?: string };
      detail = parsed.detail ?? detail;
    } catch {
      detail = text || response.statusText;
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

export function outputUrl(path: string | null): string | null {
  if (!path) {
    return null;
  }
  if (path.startsWith("http")) {
    return path;
  }
  return `${API_BASE}${path}`;
}

export async function fetchProviders(): Promise<ProviderResponse> {
  const response = await fetch(`${API_BASE}/api/providers`);
  return parseResponse<ProviderResponse>(response);
}

export async function recognizeImage(
  file: File,
  provider: ProviderId,
  returnIntermediate: boolean,
): Promise<RecognitionResult> {
  const form = new FormData();
  form.append("file", file);
  form.append("provider", provider);
  form.append("return_intermediate", String(returnIntermediate));
  const response = await fetch(`${API_BASE}/api/recognize`, {
    method: "POST",
    body: form,
  });
  return parseResponse<RecognitionResult>(response);
}

export async function evaluateBatch(
  files: File[],
  provider: ProviderId,
  groundTruth: string,
  datasetDir: string,
  returnIntermediate: boolean,
): Promise<BatchEvaluationResult> {
  const form = new FormData();
  files.forEach((file) => form.append("files", file));
  form.append("provider", provider);
  form.append("return_intermediate", String(returnIntermediate));
  if (groundTruth.trim()) {
    form.append("ground_truth", groundTruth.trim());
  }
  if (datasetDir.trim()) {
    form.append("dataset_dir", datasetDir.trim());
  }
  const response = await fetch(`${API_BASE}/api/batch/evaluate`, {
    method: "POST",
    body: form,
  });
  return parseResponse<BatchEvaluationResult>(response);
}

export async function saveRemoteSettings(settings: RemoteSettings): Promise<ProviderResponse> {
  const response = await fetch(`${API_BASE}/api/settings/remote`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(settings),
  });
  const payload = await parseResponse<{ remote_settings: RemoteSettings }>(response);
  const providers = await fetchProviders();
  return {
    providers: providers.providers,
    remote_settings: payload.remote_settings,
  };
}
