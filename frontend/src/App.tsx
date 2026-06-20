import {
  Activity,
  BarChart3,
  CheckCircle2,
  Cpu,
  Database,
  ImagePlus,
  Loader2,
  RadioTower,
  Save,
  Settings,
  Upload,
  XCircle,
} from "lucide-react";
import { ChangeEvent, FormEvent, useEffect, useMemo, useState } from "react";
import {
  evaluateBatch,
  fetchProviders,
  outputUrl,
  recognizeImage,
  saveRemoteSettings,
} from "./api";
import type {
  BatchEvaluationResult,
  ProviderId,
  ProviderInfo,
  RecognitionResult,
  RemoteSettings,
} from "./types";

type TabId = "recognize" | "experiments" | "settings";
type ImageTileModel = { label: string; note?: string; src: string | null };

const tabs: Array<{ id: TabId; label: string; icon: typeof Activity }> = [
  { id: "recognize", label: "单图识别", icon: Activity },
  { id: "experiments", label: "批量实验", icon: BarChart3 },
  { id: "settings", label: "推理设置", icon: Settings },
];

const defaultRemote: RemoteSettings = {
  enabled: false,
  endpoint: "",
  timeout_seconds: 12,
};

function formatPercent(value: number | null): string {
  if (value === null) {
    return "-";
  }
  return `${(value * 100).toFixed(1)}%`;
}

function formatMs(value: number | undefined): string {
  if (typeof value !== "number") {
    return "-";
  }
  return `${value.toFixed(1)} ms`;
}

function providerBadge(provider: ProviderInfo): string {
  if (provider.mode === "remote") {
    return provider.available ? "SERVER" : "OFFLINE";
  }
  if (provider.mode === "local_trained_onnx") {
    return "ONNX";
  }
  if (provider.mode === "local_stub") {
    return "STUB";
  }
  return "LOCAL";
}

export default function App() {
  const [activeTab, setActiveTab] = useState<TabId>("recognize");
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [selectedProvider, setSelectedProvider] = useState<ProviderId>("local_model");
  const [returnIntermediate, setReturnIntermediate] = useState(true);
  const [remoteSettings, setRemoteSettings] = useState<RemoteSettings>(defaultRemote);
  const [status, setStatus] = useState("Ready");
  const [error, setError] = useState("");

  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [recognition, setRecognition] = useState<RecognitionResult | null>(null);
  const [recognizing, setRecognizing] = useState(false);

  const [batchFiles, setBatchFiles] = useState<File[]>([]);
  const [groundTruth, setGroundTruth] = useState("");
  const [datasetDir, setDatasetDir] = useState("");
  const [batchResult, setBatchResult] = useState<BatchEvaluationResult | null>(null);
  const [evaluating, setEvaluating] = useState(false);

  useEffect(() => {
    fetchProviders()
      .then((payload) => {
        setProviders(payload.providers);
        setRemoteSettings(payload.remote_settings);
        if (payload.providers.some((provider) => provider.id === "local_model")) {
          setSelectedProvider("local_model");
        }
        setStatus("API connected");
      })
      .catch((caught: Error) => {
        setError(caught.message);
        setStatus("API offline");
      });
  }, []);

  useEffect(() => {
    if (!imageFile) {
      setImagePreview(null);
      return;
    }
    const url = URL.createObjectURL(imageFile);
    setImagePreview(url);
    return () => URL.revokeObjectURL(url);
  }, [imageFile]);

  const selectedProviderInfo = useMemo(
    () => providers.find((provider) => provider.id === selectedProvider),
    [providers, selectedProvider],
  );

  function onImageChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;
    setImageFile(file);
    setRecognition(null);
    setError("");
  }

  async function onRecognize(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!imageFile) {
      setError("请选择图片。");
      return;
    }
    setRecognizing(true);
    setError("");
    setStatus("Recognizing");
    try {
      const result = await recognizeImage(imageFile, selectedProvider, returnIntermediate);
      setRecognition(result);
      setStatus("Recognition done");
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "识别失败。";
      setError(message);
      setStatus("Recognition failed");
    } finally {
      setRecognizing(false);
    }
  }

  function onBatchFilesChange(event: ChangeEvent<HTMLInputElement>) {
    setBatchFiles(Array.from(event.target.files ?? []));
    setBatchResult(null);
  }

  async function onEvaluate(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setEvaluating(true);
    setError("");
    setStatus("Evaluating");
    try {
      const result = await evaluateBatch(
        batchFiles,
        selectedProvider,
        groundTruth,
        datasetDir,
        returnIntermediate,
      );
      setBatchResult(result);
      setStatus("Evaluation done");
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "评测失败。";
      setError(message);
      setStatus("Evaluation failed");
    } finally {
      setEvaluating(false);
    }
  }

  async function onSaveRemote(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setStatus("Saving remote settings");
    try {
      const payload = await saveRemoteSettings(remoteSettings);
      setProviders(payload.providers);
      setRemoteSettings(payload.remote_settings);
      setStatus("Settings saved");
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "设置保存失败。";
      setError(message);
      setStatus("Settings failed");
    }
  }

  const images = recognition?.images;
  const isOnnxPipeline =
    selectedProviderInfo?.mode === "local_trained_onnx" || recognition?.provider_used === "trained_onnx";
  const pipelineSteps = isOnnxPipeline
    ? ["输入图像", "YOLOv8n-pose 定位与角点", "透视校正", "CRNN-CTC 整牌识别", "结果输出"]
    : ["输入图像", "颜色阈值定位", "车牌裁剪", "二值化与字符分割", "模板/启发式识别"];
  const imageTiles: ImageTileModel[] = isOnnxPipeline
    ? [
        { label: "定位与角点", note: "YOLOv8n-pose 输出", src: outputUrl(images?.detected ?? null) },
        { label: "透视校正裁剪", note: "CRNN-CTC 输入", src: outputUrl(images?.plate_crop ?? null) },
        { label: "二值化辅助图", note: "仅用于展示", src: outputUrl(images?.binary ?? null) },
        { label: "字符框辅助图", note: "非 CTC 必需步骤", src: outputUrl(images?.segmented ?? null) },
      ]
    : [
        { label: "定位结果", src: outputUrl(images?.detected ?? null) },
        { label: "车牌裁剪", src: outputUrl(images?.plate_crop ?? null) },
        { label: "颜色掩膜", src: outputUrl(images?.mask ?? null) },
        { label: "二值化", src: outputUrl(images?.binary ?? null) },
        { label: "字符分割", src: outputUrl(images?.segmented ?? null) },
      ];

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <img alt="Northeastern University at Qinhuangdao" src="/neu-logo-cropped.png" />
          </div>
          <div>
            <h1>NEUQ PlateVision Lab</h1>
            <p>东北大学秦皇岛分校 · 车牌识别课程设计</p>
          </div>
        </div>

        <nav className="nav-tabs" aria-label="主导航">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                className={activeTab === tab.id ? "nav-tab active" : "nav-tab"}
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                type="button"
                title={tab.label}
              >
                <Icon size={18} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="runtime-strip">
          <span className="pulse" />
          <span>{status}</span>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div>
            <span className="eyebrow">NEUQ · Digital Image Processing Laboratory</span>
            <h2>{tabs.find((tab) => tab.id === activeTab)?.label}</h2>
          </div>
          <div className="provider-select">
            <Cpu size={18} />
            <select
              value={selectedProvider}
              onChange={(event) => setSelectedProvider(event.target.value as ProviderId)}
              title="Provider"
            >
              {providers.map((provider) => (
                <option key={provider.id} value={provider.id}>
                  {provider.name}
                </option>
              ))}
            </select>
          </div>
        </header>

        {error && (
          <div className="alert" role="alert">
            <XCircle size={18} />
            <span>{error}</span>
          </div>
        )}

        {activeTab === "recognize" && (
          <section className="pipeline-strip" aria-label="系统流程">
            {pipelineSteps.map((step, index) => (
              <div className="pipeline-step" key={step}>
                <span>{index + 1}</span>
                <strong>{step}</strong>
              </div>
            ))}
          </section>
        )}

        {activeTab === "recognize" && (
          <section className="workspace">
            <form className="input-panel" onSubmit={onRecognize}>
              <label className={imagePreview ? "dropzone filled" : "dropzone"}>
                <input accept="image/*" onChange={onImageChange} type="file" />
                {imagePreview ? (
                  <img alt="uploaded preview" src={imagePreview} />
                ) : (
                  <span className="drop-placeholder">
                    <ImagePlus size={34} />
                    <strong>选择车辆图片</strong>
                  </span>
                )}
              </label>

              <div className="control-row">
                <label className="switch">
                  <input
                    checked={returnIntermediate}
                    onChange={(event) => setReturnIntermediate(event.target.checked)}
                    type="checkbox"
                  />
                  <span />
                  中间图
                </label>
                <button className="primary-button" disabled={recognizing} type="submit">
                  {recognizing ? <Loader2 className="spin" size={18} /> : <Upload size={18} />}
                  <span>{recognizing ? "处理中" : "识别"}</span>
                </button>
              </div>
            </form>

            <section className="result-panel">
              <div className="result-header">
                <div>
                  <span className="eyebrow">Prediction</span>
                  <div className="plate-number">{recognition?.plate_text ?? "等待输入"}</div>
                </div>
                <div className="confidence-ring">
                  <span>{recognition ? `${Math.round(recognition.confidence * 100)}%` : "--"}</span>
                </div>
              </div>

              <div className="metric-grid">
                <div className="metric">
                  <span>类型</span>
                  <strong>{recognition?.plate_type ?? "-"}</strong>
                </div>
                <div className="metric">
                  <span>检测框</span>
                  <strong>{recognition?.bbox?.join(", ") ?? "-"}</strong>
                </div>
                <div className="metric">
                  <span>Provider</span>
                  <strong>{recognition?.provider_used ?? selectedProviderInfo?.mode ?? "-"}</strong>
                </div>
                <div className="metric">
                  <span>总耗时</span>
                  <strong>{formatMs(recognition?.timing_ms.total)}</strong>
                </div>
              </div>

              <div className="char-strip">
                {(recognition?.chars ?? []).map((char, index) => (
                  <div className="char-cell" key={`${char.text}-${index}`}>
                    <strong>{char.text}</strong>
                    <span>{Math.round(char.confidence * 100)}%</span>
                  </div>
                ))}
              </div>

              {recognition?.messages.map((message) => (
                <div className="note-line" key={message}>
                  {message}
                </div>
              ))}
            </section>
          </section>
        )}

        {activeTab === "recognize" && (
          <section className="image-board">
            {imageTiles.map((tile) => (
              <ImageTile key={tile.label} label={tile.label} note={tile.note} src={tile.src} />
            ))}
          </section>
        )}

        {activeTab === "experiments" && (
          <section className="experiment-layout">
            <form className="surface" onSubmit={onEvaluate}>
              <div className="surface-title">
                <Database size={20} />
                <h3>Batch Evaluation</h3>
              </div>
              <label className="field">
                <span>图片文件</span>
                <input accept="image/*" multiple onChange={onBatchFilesChange} type="file" />
              </label>
              <label className="field">
                <span>样本目录</span>
                <input
                  onChange={(event) => setDatasetDir(event.target.value)}
                  placeholder="例如 demo_set"
                  value={datasetDir}
                />
              </label>
              <label className="field">
                <span>Ground Truth JSON</span>
                <textarea
                  onChange={(event) => setGroundTruth(event.target.value)}
                  placeholder='{"car1.jpg":"冀A12345"}'
                  value={groundTruth}
                />
              </label>
              <button className="primary-button wide" disabled={evaluating} type="submit">
                {evaluating ? <Loader2 className="spin" size={18} /> : <BarChart3 size={18} />}
                <span>{evaluating ? "评测中" : "开始评测"}</span>
              </button>
            </form>

            <section className="surface">
              <div className="surface-title">
                <Activity size={20} />
                <h3>Metrics</h3>
              </div>
              <div className="metric-grid two">
                <div className="metric">
                  <span>Total</span>
                  <strong>{batchResult?.metrics.total ?? 0}</strong>
                </div>
                <div className="metric">
                  <span>Labeled</span>
                  <strong>{batchResult?.metrics.labeled ?? 0}</strong>
                </div>
                <div className="metric">
                  <span>Accuracy</span>
                  <strong>{formatPercent(batchResult?.metrics.exact_accuracy ?? null)}</strong>
                </div>
                <div className="metric">
                  <span>F1</span>
                  <strong>{formatPercent(batchResult?.metrics.char_f1 ?? null)}</strong>
                </div>
              </div>
              <div className="result-table">
                <div className="table-row head">
                  <span>文件</span>
                  <span>预测</span>
                  <span>置信度</span>
                  <span>状态</span>
                </div>
                {(batchResult?.results ?? []).slice(0, 12).map((item) => (
                  <div className="table-row" key={item.file_name}>
                    <span>{item.file_name}</span>
                    <span>{item.prediction || "-"}</span>
                    <span>{Math.round(item.confidence * 100)}%</span>
                    <span className={item.correct === false ? "bad" : "good"}>
                      {item.correct === null ? "-" : item.correct ? "OK" : "ERR"}
                    </span>
                  </div>
                ))}
              </div>
            </section>
          </section>
        )}

        {activeTab === "settings" && (
          <section className="settings-layout">
            <section className="provider-list">
              {providers.map((provider) => (
                <button
                  className={selectedProvider === provider.id ? "provider-card selected" : "provider-card"}
                  key={provider.id}
                  onClick={() => setSelectedProvider(provider.id)}
                  type="button"
                >
                  <span className="provider-icon">
                    {provider.mode === "remote" ? <RadioTower size={20} /> : <Cpu size={20} />}
                  </span>
                  <span>
                    <strong>{provider.name}</strong>
                    <small>{provider.description}</small>
                  </span>
                  <em>{providerBadge(provider)}</em>
                </button>
              ))}
            </section>

            <form className="surface remote-form" onSubmit={onSaveRemote}>
              <div className="surface-title">
                <RadioTower size={20} />
                <h3>Remote Server</h3>
              </div>
              <label className="switch">
                <input
                  checked={remoteSettings.enabled}
                  onChange={(event) =>
                    setRemoteSettings((current) => ({ ...current, enabled: event.target.checked }))
                  }
                  type="checkbox"
                />
                <span />
                启用
              </label>
              <label className="field">
                <span>Endpoint</span>
                <input
                  onChange={(event) =>
                    setRemoteSettings((current) => ({ ...current, endpoint: event.target.value }))
                  }
                  placeholder="http://server:8000/api/recognize"
                  value={remoteSettings.endpoint}
                />
              </label>
              <label className="field">
                <span>Timeout</span>
                <input
                  max={120}
                  min={1}
                  onChange={(event) =>
                    setRemoteSettings((current) => ({
                      ...current,
                      timeout_seconds: Number(event.target.value),
                    }))
                  }
                  type="number"
                  value={remoteSettings.timeout_seconds}
                />
              </label>
              <button className="primary-button wide" type="submit">
                <Save size={18} />
                <span>保存</span>
              </button>
            </form>
          </section>
        )}
      </main>
    </div>
  );
}

function ImageTile({ label, note, src }: { label: string; note?: string; src: string | null }) {
  return (
    <figure className="image-tile">
      <figcaption>
        <strong>{label}</strong>
        {note && <span>{note}</span>}
      </figcaption>
      {src ? (
        <img alt={label} src={src} />
      ) : (
        <div className="empty-image">
          <CheckCircle2 size={20} />
        </div>
      )}
    </figure>
  );
}
