import { useEffect, useRef, useState, useCallback } from "react";
import "../styles/MedScribe.css";
import { PatientContextCard, SOAPExtras } from "./MedScribeComponents";
import type { SOAPNote, EvaluationScores, PatientContextData } from "./MedScribeComponents";

const WS_URL = import.meta.env.VITE_WS_URL || "ws://localhost:8020/api/transcribe/";
const CHUNK_DURATION_MS = 5000;

type SessionState = "idle" | "connecting" | "recording" | "processing" | "done" | "error";





interface ConsultationResult {
  soap: SOAPNote;
  scores: EvaluationScores;
  patient_context: PatientContextData;
  missing_fields: Array<{ field: string; message: string }>;
  transcript: string;
}

function ApproveBar({
  sessionId,
  soap,
  patientId,
}: {
  sessionId: string;
  soap: SOAPNote;
  patientId: string;
}) {
  const [status, setStatus] = useState<"idle" | "loading" | "done" | "error">("idle");
  const [ehrId, setEhrId] = useState<string | null>(null);

  const approve = async () => {
    setStatus("loading");
    try {
      const res = await fetch(
        // `${import.meta.env.VITE_API_URL || "http://localhost:8000"}/Scribe/approve`,
        `${import.meta.env.VITE_API_URL || "http://localhost:8020"}/api/v1/Scribe/approve`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sessionId, soap, patient_id: patientId }),
        }
      );
      const data = await res.json();
      if (data.success) {
        setEhrId(data.ehr_record_id);
        setStatus("done");
      } else {
        setStatus("error");
      }
    } catch {
      setStatus("error");
    }
  };

  return (
    <div className="ms-card ms-approve-bar">
      <div className="ms-approve-left">
        <span className="ms-approve-label">Physician Approval</span>
        <span className="ms-approve-sub">Review and insert into EHR</span>
      </div>
      <div className="ms-approve-right">
        {status === "idle" && (
          <button className="ms-btn ms-btn-approve" onClick={approve}>
            Approve &amp; Insert →
          </button>
        )}
        {status === "loading" && <span className="ms-status-label amber">Inserting…</span>}
        {status === "done" && (
          <span className="ms-status-label green">✓ Inserted {ehrId && `#${ehrId}`}</span>
        )}
        {status === "error" && (
          <button className="ms-btn ms-btn-danger" onClick={approve}>
            Retry
          </button>
        )}
      </div>
    </div>
  );
}

export default function LiveRecording() {
  const [sessionState, setSessionState] = useState<SessionState>("idle");
  const [transcript, setTranscript] = useState("");
  const [result, setResult] = useState<ConsultationResult | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [activeTab, setActiveTab] = useState<"soap" | "details" | "transcript">("soap");

  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const patientId = "P001";

  const cleanup = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    if (timerRef.current) clearInterval(timerRef.current);
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
  }, []);

  useEffect(() => {
    return () => {
      cleanup();
      wsRef.current?.close();
    };
  }, [cleanup]);

  const recordChunk = useCallback((stream: MediaStream) => {
    const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
      ? "audio/webm;codecs=opus"
      : MediaRecorder.isTypeSupported("audio/webm")
      ? "audio/webm"
      : null;

    if (!mimeType) return;

    const recorder = new MediaRecorder(stream, { mimeType });
    const chunks: BlobPart[] = [];

    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunks.push(e.data);
    };

    recorder.onstop = async () => {
      if (
        chunks.length === 0 ||
        !wsRef.current ||
        wsRef.current.readyState !== WebSocket.OPEN
      )
        return;

      const blob = new Blob(chunks, { type: mimeType });
      const arrayBuffer = await blob.arrayBuffer();
      const uint8 = new Uint8Array(arrayBuffer);
      let binary = "";
      for (let i = 0; i < uint8.length; i++) binary += String.fromCharCode(uint8[i]);
      const base64 = btoa(binary);

      wsRef.current.send(
        JSON.stringify({
          event: "transcribe",
          audio: base64,
          patient_id: patientId,
          session_id: sessionIdRef.current,
        })
      );
    };

    recorder.start();
    setTimeout(() => {
      if (recorder.state !== "inactive") recorder.stop();
    }, CHUNK_DURATION_MS);
  }, []);

  const connect = useCallback((): Promise<void> => {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      const timeout = setTimeout(() => reject(new Error("Connection timeout")), 8000);

      ws.onopen = () => {
        clearTimeout(timeout);
        resolve();
      };

      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.event === "scribe.connected") {
            sessionIdRef.current = msg.session_id;
            setSessionId(msg.session_id);
          } else if (msg.event === "scribe.transcript_chunk") {
            setTranscript((p) => p + " " + (msg.chunk || ""));
          } else if (msg.event === "scribe.processing") {
            setSessionState("processing");
          } else if (msg.event === "scribe.consultation_result") {
            setResult(msg.data);
            setSessionState("done");
            cleanup();
          } else if (msg.event === "scribe.error") {
            setError(msg.message || "An error occurred");
            setSessionState("error");
            cleanup();
          }
        } catch {}
      };

      ws.onerror = () => {
        clearTimeout(timeout);
        reject(new Error("WebSocket connection failed"));
      };

      ws.onclose = () => {
        clearTimeout(timeout);
      };
    });
  }, [cleanup]);

  const startRecording = useCallback(async () => {
    setError(null);
    setTranscript("");
    setResult(null);
    setElapsed(0);
    setSessionState("connecting");

    try {
      await connect();
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      setSessionState("recording");

      timerRef.current = setInterval(() => setElapsed((e) => e + 1), 1000);
      // Start first chunk immediately, then every CHUNK_DURATION_MS
      recordChunk(stream);
      intervalRef.current = setInterval(() => recordChunk(stream), CHUNK_DURATION_MS);
    } catch (err: any) {
      setError(err.message || "Failed to start recording");
      setSessionState("error");
    }
  }, [connect, recordChunk]);

  const stopRecording = useCallback(() => {
    cleanup();
    setSessionState("processing");
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          event: "end",
          patient_id: patientId,
          session_id: sessionIdRef.current,
        })
      );
    }
  }, [cleanup]);

  const reset = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
    sessionIdRef.current = null;
    setSessionState("idle");
    setTranscript("");
    setResult(null);
    setError(null);
    setElapsed(0);
    setSessionId(null);
  }, []);

  const fmt = (n?: number) => (typeof n === "number" ? `${Math.round(n)}%` : "—");
  const fmtTime = (s: number) =>
    `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;

  const scoreColor = (val?: number) => {
    if (val === undefined || val === null) return "var(--ms-muted)";
    if (val >= 85) return "var(--ms-green)";
    if (val >= 60) return "var(--ms-amber)";
    return "var(--ms-red)";
  };

  return (
    <div className="ms-page">
      <div className="ms-page-header">
        <div className="ms-breadcrumb">
          MedScribe+ <span>/</span> Live Session
        </div>
        <h1 className="ms-page-title">Live Consultation</h1>
        <p className="ms-page-sub">Real-time transcription &amp; automated note generation</p>
      </div>

      <div className="ms-layout">
        {/* ── Left: recorder + transcript ─────────────────────────── */}
        <div className="ms-left">
          <div className="ms-card ms-recorder-card">
            <div className="ms-recorder-vis">
              <div className={`ms-pulse-ring ${sessionState === "recording" ? "active" : ""}`}>
                <div className={`ms-pulse-dot ${sessionState === "recording" ? "active" : ""}`} />
              </div>
              {sessionState === "recording" && (
                <div className="ms-waveform">
                  {Array.from({ length: 24 }).map((_, i) => (
                    <div
                      key={i}
                      className="ms-wave-bar"
                      style={{ animationDelay: `${i * 0.06}s` }}
                    />
                  ))}
                </div>
              )}
            </div>

            <div className="ms-recorder-status">
              {sessionState === "idle" && (
                <span className="ms-status-label">Ready to record</span>
              )}
              {sessionState === "connecting" && (
                <span className="ms-status-label amber">Connecting…</span>
              )}
              {sessionState === "recording" && (
                <span className="ms-status-label green">
                  Recording &nbsp;<span className="ms-timer">{fmtTime(elapsed)}</span>
                </span>
              )}
              {sessionState === "processing" && (
                <span className="ms-status-label amber">Generating notes…</span>
              )}
              {sessionState === "done" && (
                <span className="ms-status-label green">Complete</span>
              )}
              {sessionState === "error" && (
                <span className="ms-status-label red">Error</span>
              )}
            </div>

            {error && <div className="ms-error-banner">⚠ {error}</div>}

            <div className="ms-recorder-actions">
              {(sessionState === "idle" || sessionState === "error") && (
                <button className="ms-btn ms-btn-primary" onClick={startRecording}>
                  <span className="ms-rec-dot" /> Start Recording
                </button>
              )}
              {sessionState === "recording" && (
                <button className="ms-btn ms-btn-danger" onClick={stopRecording}>
                  <span className="ms-stop-sq" /> End Session
                </button>
              )}
              {sessionState === "connecting" && (
                <button className="ms-btn ms-btn-ghost" disabled>
                  Connecting…
                </button>
              )}
              {(sessionState === "done" || sessionState === "error") && (
                <button className="ms-btn ms-btn-ghost" onClick={reset}>
                  New Session
                </button>
              )}
            </div>

            {sessionId && (
              <div className="ms-session-id">
                Session <code>{sessionId.slice(0, 8)}…</code>
              </div>
            )}
          </div>

          <div className="ms-card ms-transcript-card">
            <div className="ms-card-header">
              <span className="ms-card-title">Live Transcript</span>
              {sessionState === "recording" && (
                <span className="ms-live-badge">● LIVE</span>
              )}
            </div>
            <div className="ms-transcript-body">
              {transcript ? (
                <p className="ms-transcript-text">{transcript}</p>
              ) : (
                <p className="ms-transcript-empty">
                  Transcript will appear here as you speak…
                </p>
              )}
            </div>
          </div>
        </div>

        {/* ── Right: results ───────────────────────────────────────── */}
        <div className="ms-right">
          {sessionState === "processing" && (
            <div className="ms-card ms-processing-card">
              <div className="ms-spinner" />
              <p className="ms-processing-label">Running evaluation pipeline</p>
              <div className="ms-processing-steps">
                {[
                  "Generating SOAP note",
                  "Detecting hallucinations",
                  "Checking drug interactions",
                  "Scoring compliance",
                ].map((step, i) => (
                  <div
                    key={i}
                    className="ms-step"
                    style={{ animationDelay: `${i * 0.5}s` }}
                  >
                    <div className="ms-step-dot" />
                    {step}
                  </div>
                ))}
              </div>
            </div>
          )}

          {result && (
            <>
              {/* Scores overview */}
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
                    {
                      label: "Guideline Align.",
                      val: result.scores?.guideline_alignment,
                    },
                  ].map(({ label, val }) => (
                    <div key={label} className="ms-score-cell">
                      <div
                        className="ms-score-val"
                        style={{ color: scoreColor(val) }}
                      >
                        {fmt(val)}
                      </div>
                      <div className="ms-score-label">{label}</div>
                      <div className="ms-score-bar-bg">
                        <div
                          className="ms-score-bar-fill"
                          style={{
                            width: val ? `${val}%` : "0%",
                            backgroundColor: scoreColor(val),
                          }}
                        />
                      </div>
                    </div>
                  ))}
                  <div className="ms-score-cell">
                    <div
                      className="ms-score-val"
                      style={{
                        color:
                          result.scores?.hallucination_risk === "low"
                            ? "var(--ms-green)"
                            : "var(--ms-red)",
                      }}
                    >
                      {result.scores?.hallucination_risk?.toUpperCase() || "—"}
                    </div>
                    <div className="ms-score-label">Hallucination</div>
                  </div>
                </div>
              </div>

              {/* Drug alert */}
              {result.scores?.drug_interactions &&
                result.scores.drug_interactions.length > 0 && (
                  <div className="ms-card ms-alert-card">
                    <div className="ms-alert-header">⚠ Drug Interaction Alert</div>
                    {result.scores.drug_interactions.map((d, i) => (
                      <div key={i} className="ms-alert-item">
                        <span className={`ms-severity ms-sev-${d.severity}`}>
                          {d.severity.toUpperCase()}
                        </span>
                        <span className="ms-alert-text">
                          {d.drug} — {d.description}
                        </span>
                      </div>
                    ))}
                  </div>
                )}

              {/* Tabbed results */}
              <div className="ms-card ms-results-card">
                <div className="ms-tabs">
                  {(
                    [
                      ["soap", "SOAP Note"],
                      ["details", "Details"],
                      ["transcript", "Transcript"],
                    ] as const
                  ).map(([tab, label]) => (
                    <button
                      key={tab}
                      className={`ms-tab ${activeTab === tab ? "active" : ""}`}
                      onClick={() => setActiveTab(tab)}
                    >
                      {label}
                    </button>
                  ))}
                </div>

                {activeTab === "soap" && result.soap && (
                  <div className="ms-soap">
                    {(
                      [
                        "subjective",
                        "objective",
                        "assessment",
                        "plan",
                      ] as const
                    ).map((sec) => (
                      <div key={sec} className="ms-soap-section">
                        <div className="ms-soap-label">{sec[0].toUpperCase() + sec.slice(1)}</div>
                        <p className="ms-soap-text">{result.soap[sec] || "—"}</p>
                      </div>
                    ))}
                    {result.soap.icd10_codes && result.soap.icd10_codes.length > 0 && (
                      <div className="ms-codes-row">
                        {result.soap.icd10_codes.map((c) => (
                          <span key={c} className="ms-code-badge ms-icd">
                            {c}
                          </span>
                        ))}
                      </div>
                    )}
                    <SOAPExtras soap={result.soap} />
                  </div>
                )}

                {activeTab === "details" && (
                  <div className="ms-details">
                    {result.scores?.guideline_suggestions?.map((s, i) => (
                      <div key={i} className="ms-suggestion">
                        💡 {s}
                      </div>
                    ))}
                    {result.missing_fields?.map((f, i) => (
                      <div key={i} className="ms-missing">
                        ⚠ Missing: <strong>{f.field}</strong> — {f.message}
                      </div>
                    ))}
                    {result.scores?.cpt_codes && result.scores.cpt_codes.length > 0 && (
                      <div className="ms-codes-row" style={{ marginTop: 12 }}>
                        {result.scores.cpt_codes.map((c) => (
                          <span key={c} className="ms-code-badge ms-cpt">
                            {c}
                          </span>
                        ))}
                      </div>
                    )}
                    <PatientContextCard ctx={result.patient_context} />
                  </div>
                )}

                {activeTab === "transcript" && (
                  <div className="ms-transcript-full">
                    <p>{result.transcript || transcript || "No transcript available."}</p>
                  </div>
                )}
              </div>

              <ApproveBar
                sessionId={sessionId!}
                soap={result.soap}
                patientId={patientId}
              />
            </>
          )}

          {!result && sessionState === "idle" && (
            <div className="ms-card ms-empty-state">
              <div className="ms-empty-icon">🎙</div>
              <p className="ms-empty-text">
                Start a recording to generate
                <br />
                SOAP notes &amp; evaluation scores
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}