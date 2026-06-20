export type ProviderId = "opencv_baseline" | "local_model" | "remote_server";

export interface ProviderInfo {
  id: ProviderId;
  name: string;
  description: string;
  available: boolean;
  mode: string;
}

export interface CharacterResult {
  text: string;
  confidence: number;
  bbox: number[] | null;
}

export interface RecognitionImages {
  detected: string | null;
  plate_crop: string | null;
  mask: string | null;
  binary: string | null;
  segmented: string | null;
}

export interface RecognitionResult {
  request_id: string;
  provider: string;
  provider_used: string;
  plate_text: string;
  plate_type: string;
  confidence: number;
  bbox: number[] | null;
  chars: CharacterResult[];
  images: RecognitionImages;
  timing_ms: Record<string, number>;
  messages: string[];
}

export interface RemoteSettings {
  enabled: boolean;
  endpoint: string;
  timeout_seconds: number;
}

export interface ProviderResponse {
  providers: ProviderInfo[];
  remote_settings: RemoteSettings;
}

export interface BatchMetrics {
  total: number;
  labeled: number;
  exact_accuracy: number | null;
  char_precision: number | null;
  char_recall: number | null;
  char_f1: number | null;
}

export interface BatchItemResult {
  file_name: string;
  ground_truth: string | null;
  prediction: string;
  correct: boolean | null;
  confidence: number;
  plate_type: string;
  output_image: string | null;
  message: string | null;
}

export interface BatchEvaluationResult {
  evaluation_id: string;
  provider: string;
  metrics: BatchMetrics;
  results: BatchItemResult[];
  errors: BatchItemResult[];
  report_file: string | null;
}

