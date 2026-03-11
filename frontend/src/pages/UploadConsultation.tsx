import { useCallback, useRef, useState } from "react";
import "../styles/MedScribe.css";
import { PatientContextCard, SOAPExtras } from "./MedScribeComponents";
import type { SOAPNote, EvaluationScores, PatientContextData } from "./MedScribeComponents";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8020/api/v1";

type UploadState = "idle" | "uploading" | "processing" | "done" | "error";

interface AudioUploadResponse {
  session_id: string;
  transcript: string;
  soap: SOAPNote;
  scores: EvaluationScores;
  patient_context: PatientContextData;
  missing_fields: Array<{ field: string; message: string }>;
}

const ACCEPTED = ["audio/wav", "audio/mp3", "audio/mpeg", "audio/webm", "audio/ogg", "audio/m4a"];
const ACCEPTED_EXT = ".wav,.mp3,.webm,.ogg,.m4a";

function FileIcon() {
  return (
    <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
      <rect width="40" height="40" rx="10" fill="var(--ms-surface2)" />
      <path d="M12 10h11l7 7v13a2 2 0 01-2 2H12a2 2 0 01-2-2V12a2 2 0 012-2z" stroke="var(--ms-accent)" strokeWidth="1.5" fill="none" />
      <path d="M23 10v7h7" stroke="var(--ms-accent)" strokeWidth="1.5" fill="none" />
      <path d="M16 22h8M16 26h5" stroke="var(--ms-muted)" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

function UploadIcon() {
  return (
    <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
      <circle cx="24" cy="24" r="23" stroke="var(--ms-border)" strokeWidth="1.5" strokeDasharray="4 3" />
      <path d="M24 32V20M18 26l6-6 6 6" stroke="var(--ms-accent)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export default function UploadConsultation() {
  const [uploadState, setUploadState] = useState<UploadState>("idle");
  const [file, setFile] = useState<File | null>(null);
  const [patientId, setPatientId] = useState("P001");
  const [result, setResult] = useState<AudioUploadResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [activeTab, setActiveTab] = useState<"soap" | "details" | "transcript">("soap");
  const [approveStatus, setApproveStatus] = useState<"idle" | "loading" | "done" | "error">("idle");
  const [ehrId, setEhrId] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const progressRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const validateFile = (f: File) => {
    if (!ACCEPTED.includes(f.type) && !f.name.match(/\.(wav|mp3|webm|ogg|m4a)$/i)) {
      return "Unsupported file type. Please upload a WAV, MP3, WebM, OGG, or M4A file.";
    }
    if (f.size > 100 * 1024 * 1024) return "File too large. Max 100MB.";
    return null;
  };

  const pickFile = (f: File) => {
    const err = validateFile(f);
    if (err) { setError(err); return; }
    setError(null);
    setFile(f);
    setResult(null);
    setApproveStatus("idle");
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f) pickFile(f);
  }, []);

  const simulateProgress = () => {
    setProgress(0);
    let p = 0;
    progressRef.current = setInterval(() => {
      p += Math.random() * 8;
      if (p >= 90) { clearInterval(progressRef.current!); p = 90; }
      setProgress(Math.min(p, 90));
    }, 400);
  };

  const upload = async () => {
    if (!file) return;
    setError(null);
    setResult(null);
    setUploadState("uploading");
    simulateProgress();

    const form = new FormData();
    form.append("file", file);
    form.append("patient_id", patientId);

    try {
      setUploadState("processing");
      const res = await fetch(`${API_BASE}/Scribe/upload`, { method: "POST", body: form });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `HTTP ${res.status}`);
      }
      const data: AudioUploadResponse = await res.json();
      clearInterval(progressRef.current!);
      setProgress(100);
      setResult(data);
      setUploadState("done");
    } catch (err: any) {
      clearInterval(progressRef.current!);
      setError(err.message || "Upload failed");
      setUploadState("error");
    }
  };

  const approve = async () => {
    if (!result) return;
    setApproveStatus("loading");
    try {
      const res = await fetch(`${API_BASE}/Scribe/approve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: result.session_id, soap: result.soap, patient_id: patientId }),
      });
      const data = await res.json();
      if (data.success) { setEhrId(data.ehr_record_id); setApproveStatus("done"); }
      else setApproveStatus("error");
    } catch { setApproveStatus("error"); }
  };

  const reset = () => {
    setUploadState("idle");
    setFile(null);
    setResult(null);
    setError(null);
    setProgress(0);
    setApproveStatus("idle");
    setEhrId(null);
  };

  const fmt = (n?: number) => (typeof n === "number" ? `${Math.round(n)}%` : "—");
  const scoreColor = (val?: number) => {
    if (val === undefined || val === null) return "var(--ms-muted)";
    if (val >= 85) return "var(--ms-green)";
    if (val >= 60) return "var(--ms-amber)";
    return "var(--ms-red)";
  };

  const fmtSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const isLoading = uploadState === "uploading" || uploadState === "processing";

  return (
    <div className="ms-page">
      <div className="ms-page-header">
        <div className="ms-breadcrumb">MedScribe+ <span>/</span> Upload</div>
        <h1 className="ms-page-title">Upload Consultation</h1>
        <p className="ms-page-sub">Process a pre-recorded consultation audio file</p>
      </div>

      <div className="ms-layout">
        {/* ── Left: upload form ─────────────────────────────────── */}
        <div className="ms-left">
          <div className="ms-card">
            <div className="ms-card-header">
              <span className="ms-card-title">Audio File</span>
            </div>

            {/* Drop zone */}
            {!file ? (
              <div
                className={`ms-dropzone ${dragOver ? "drag-over" : ""}`}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={onDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <UploadIcon />
                <p className="ms-drop-title">Drop your audio file here</p>
                <p className="ms-drop-sub">or click to browse · WAV, MP3, WebM, OGG, M4A · max 100 MB</p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept={ACCEPTED_EXT}
                  style={{ display: "none" }}
                  onChange={(e) => { const f = e.target.files?.[0]; if (f) pickFile(f); }}
                />
              </div>
            ) : (
              <div className="ms-file-preview">
                <FileIcon />
                <div className="ms-file-info">
                  <span className="ms-file-name">{file.name}</span>
                  <span className="ms-file-meta">{fmtSize(file.size)} · {file.type || "audio"}</span>
                </div>
                {!isLoading && (
                  <button className="ms-btn-icon-only" onClick={reset} title="Remove">✕</button>
                )}
              </div>
            )}

            {/* Patient ID */}
            <div className="ms-field">
              <label className="ms-field-label">Patient ID</label>
              <input
                className="ms-input"
                value={patientId}
                onChange={(e) => setPatientId(e.target.value)}
                placeholder="e.g. P001"
                disabled={isLoading}
              />
            </div>

            {/* Progress */}
            {isLoading && (
              <div className="ms-progress-wrap">
                <div className="ms-progress-bar">
                  <div className="ms-progress-fill" style={{ width: `${progress}%` }} />
                </div>
                <span className="ms-progress-label">
                  {uploadState === "uploading" ? "Uploading…" : "Processing pipeline…"}
                </span>
              </div>
            )}

            {error && <div className="ms-error-banner">⚠ {error}</div>}

            <div className="ms-recorder-actions">
              {!isLoading && uploadState !== "done" && (
                <button
                  className="ms-btn ms-btn-primary"
                  onClick={upload}
                  disabled={!file || !!error}
                >
                  Process Consultation
                </button>
              )}
              {uploadState === "done" && (
                <button className="ms-btn ms-btn-ghost" onClick={reset}>
                  Upload Another
                </button>
              )}
            </div>
          </div>

          {/* Transcript preview */}
          {result && (
            <div className="ms-card ms-transcript-card">
              <div className="ms-card-header">
                <span className="ms-card-title">Transcript</span>
                <span className="ms-badge-done">Processed</span>
              </div>
              <div className="ms-transcript-body">
                <p className="ms-transcript-text">{result.transcript}</p>
              </div>
            </div>
          )}
        </div>

        {/* ── Right: results ───────────────────────────────────── */}
        <div className="ms-right">
          {isLoading && (
            <div className="ms-card ms-processing-card">
              <div className="ms-spinner" />
              <p className="ms-processing-label">Running evaluation pipeline</p>
              <div className="ms-processing-steps">
                {["Transcribing audio", "Generating SOAP note", "Checking drug interactions", "Scoring compliance"].map(
                  (step, i) => (
                    <div key={i} className="ms-step" style={{ animationDelay: `${i * 0.5}s` }}>
                      <div className="ms-step-dot" /> {step}
                    </div>
                  )
                )}
              </div>
            </div>
          )}

          {result && (
            <>
              {/* Scores */}
              <div className="ms-card ms-scores-card">
                <div className="ms-card-header">
                  <span className="ms-card-title">Evaluation Scores</span>
                  {result.scores?.overall_ready && (
                    <span className="ms-ready-badge">✓ Ready</span>
                  )}
                </div>
                <div className="ms-scores-grid">
                  {[
                    { label: "Completeness", val: result.scores?.completeness },
                    { label: "Drug Safety", val: result.scores?.drug_safety },
                    { label: "Guideline Align.", val: result.scores?.guideline_alignment },
                  ].map(({ label, val }) => (
                    <div key={label} className="ms-score-cell">
                      <div className="ms-score-val" style={{ color: scoreColor(val) }}>
                        {fmt(val)}
                      </div>
                      <div className="ms-score-label">{label}</div>
                      <div className="ms-score-bar-bg">
                        <div
                          className="ms-score-bar-fill"
                          style={{ width: val ? `${val}%` : "0%", backgroundColor: scoreColor(val) }}
                        />
                      </div>
                    </div>
                  ))}
                  <div className="ms-score-cell">
                    <div
                      className="ms-score-val"
                      style={{
                        color: result.scores?.hallucination_risk === "low" ? "var(--ms-green)" : "var(--ms-red)",
                      }}
                    >
                      {result.scores?.hallucination_risk?.toUpperCase() || "—"}
                    </div>
                    <div className="ms-score-label">Hallucination</div>
                  </div>
                </div>
              </div>

              {/* Drug alert */}
              {result.scores?.drug_interactions && result.scores.drug_interactions.length > 0 && (
                <div className="ms-card ms-alert-card">
                  <div className="ms-alert-header">⚠ Drug Interaction Alert</div>
                  {result.scores.drug_interactions.map((d, i) => (
                    <div key={i} className="ms-alert-item">
                      <span className={`ms-severity ms-sev-${d.severity}`}>{d.severity.toUpperCase()}</span>
                      <span className="ms-alert-text">{d.drug} — {d.description}</span>
                    </div>
                  ))}
                </div>
              )}

              {/* Tabbed note */}
              <div className="ms-card ms-results-card">
                <div className="ms-tabs">
                  {([["soap", "SOAP Note"], ["details", "Details"], ["transcript", "Transcript"]] as const).map(
                    ([tab, label]) => (
                      <button
                        key={tab}
                        className={`ms-tab ${activeTab === tab ? "active" : ""}`}
                        onClick={() => setActiveTab(tab)}
                      >
                        {label}
                      </button>
                    )
                  )}
                </div>

                {activeTab === "soap" && result.soap && (
                  <div className="ms-soap">
                    {(["subjective", "objective", "assessment", "plan"] as const).map((sec) => (
                      <div key={sec} className="ms-soap-section">
                        <div className="ms-soap-label">{sec[0].toUpperCase() + sec.slice(1)}</div>
                        <p className="ms-soap-text">{result.soap[sec] || "—"}</p>
                      </div>
                    ))}
                    {result.soap.icd10_codes && result.soap.icd10_codes.length > 0 && (
                      <div className="ms-codes-row">
                        {result.soap.icd10_codes.map((c) => (
                          <span key={c} className="ms-code-badge ms-icd">{c}</span>
                        ))}
                      </div>
                    )}
                    <SOAPExtras soap={result.soap} />
                  </div>
                )}

                {activeTab === "details" && (
                  <div className="ms-details">
                    {result.scores?.guideline_suggestions?.map((s, i) => (
                      <div key={i} className="ms-suggestion">💡 {s}</div>
                    ))}
                    {result.missing_fields?.map((f, i) => (
                      <div key={i} className="ms-missing">
                        ⚠ Missing: <strong>{f.field}</strong> — {f.message}
                      </div>
                    ))}
                    {result.scores?.cpt_codes && result.scores.cpt_codes.length > 0 && (
                      <div className="ms-codes-row" style={{ marginTop: 12 }}>
                        {result.scores.cpt_codes.map((c) => (
                          <span key={c} className="ms-code-badge ms-cpt">{c}</span>
                        ))}
                      </div>
                    )}
                    <PatientContextCard ctx={result.patient_context} />
                  </div>
                )}

                {activeTab === "transcript" && (
                  <div className="ms-transcript-full">
                    <p>{result.transcript || "No transcript available."}</p>
                  </div>
                )}
              </div>

              {/* Approve bar */}
              <div className="ms-card ms-approve-bar">
                <div className="ms-approve-left">
                  <span className="ms-approve-label">Physician Approval</span>
                  <span className="ms-approve-sub">Review and insert into EHR</span>
                </div>
                <div className="ms-approve-right">
                  {approveStatus === "idle" && (
                    <button className="ms-btn ms-btn-approve" onClick={approve}>
                      Approve &amp; Insert →
                    </button>
                  )}
                  {approveStatus === "loading" && (
                    <span className="ms-status-label amber">Inserting…</span>
                  )}
                  {approveStatus === "done" && (
                    <span className="ms-status-label green">
                      ✓ Inserted {ehrId && `#${ehrId}`}
                    </span>
                  )}
                  {approveStatus === "error" && (
                    <button className="ms-btn ms-btn-danger" onClick={approve}>
                      Retry
                    </button>
                  )}
                </div>
              </div>
            </>
          )}

          {!result && !isLoading && (
            <div className="ms-card ms-empty-state">
              <div className="ms-empty-icon">📁</div>
              <p className="ms-empty-text">
                Upload a consultation recording
                <br />
                to generate SOAP notes &amp; scores
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}