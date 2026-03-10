import { useMemo, useState } from "react";
import "../index.css";

type SignalItem = { ngram: string; contrib: number };

type ViralitySignals = {
  top_push_viral?: SignalItem[];
  top_push_nonviral?: SignalItem[];
};

type DebugProbs = {
  p_bal?: number;
  p_imbal?: number;
  alpha?: number;
  beta?: number;
  threshold?: number;
};

type ViralityPayload = {
  prob_viral?: number;
  pred_label?: string;
  debug_probs?: DebugProbs;
  signals?: ViralitySignals;
};

type ApiResponse = {
  success: boolean;
  message?: string;
  virality?: ViralityPayload | Record<string, any>;
  rewrite?: boolean;
  new_story?: string;
  rewrite_details?: string;
};

function fmt(n?: number, digits = 4) {
  if (typeof n !== "number" || Number.isNaN(n)) return "—";
  return n.toFixed(digits);
}

function parseMarkdownTable(text: string): { before: string; tables: Array<{ headers: string[]; rows: string[][] }>; after: string } | null {
  const tableRegex = /\|(.+)\|\n\|[-:\s|]+\|\n((?:\|.+\|\n?)+)/g;
  const matches = [...text.matchAll(tableRegex)];
  
  if (matches.length === 0) return null;

  const tables: Array<{ headers: string[]; rows: string[][] }> = [];
  let lastIndex = 0;
  let before = "";
  let after = "";

  matches.forEach((match, idx) => {
    if (idx === 0) {
      before = text.substring(0, match.index);
    }
    
    const headerLine = match[1];
    const headers = headerLine.split('|').map(h => h.trim()).filter(Boolean);
    
    const bodyLines = match[2].trim().split('\n');
    const rows = bodyLines.map(line => 
      line.split('|').map(cell => cell.trim()).filter(Boolean)
    );
    
    tables.push({ headers, rows });
    lastIndex = (match.index || 0) + match[0].length;
  });

  after = text.substring(lastIndex);

  return { before, tables, after };
}

function renderInsightsContent(content: string) {
  const parsed = parseMarkdownTable(content);
  
  if (!parsed) {
    return <div className="rewrite-box">{content}</div>;
  }

  return (
    <div className="rewrite-box">
      {parsed.before && <div style={{ marginBottom: '16px' }}>{parsed.before}</div>}
      
      {parsed.tables.map((table, tableIdx) => (
        <table key={tableIdx} className="insights-table">
          <thead>
            <tr>
              {table.headers.map((header, idx) => (
                <th key={idx}>{header}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.rows.map((row, rowIdx) => (
              <tr key={rowIdx}>
                {row.map((cell, cellIdx) => (
                  <td key={cellIdx} dangerouslySetInnerHTML={{ __html: cell.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>') }} />
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      ))}
      
      {parsed.after && <div style={{ marginTop: '16px' }}>{parsed.after}</div>}
    </div>
  );
}

function isViralLabel(label?: string) {
  if (!label) return false;
  const x = label.trim().toLowerCase();
  return x === "viral" || (x.includes("viral") && !x.includes("non"));
}

function safeSignals(virality: any): ViralitySignals {
  const s = virality?.signals ?? {};
  return {
    top_push_viral: Array.isArray(s.top_push_viral) ? s.top_push_viral : [],
    top_push_nonviral: Array.isArray(s.top_push_nonviral) ? s.top_push_nonviral : [],
  };
}

export default function StoryPredictorPage() {
  const [scriptText, setScriptText] = useState("");
  const [rewriteStory, setRewriteStory] = useState(false);

  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [resp, setResp] = useState<ApiResponse | null>(null);
  
  const [copiedInsights, setCopiedInsights] = useState(false);
  const [copiedStory, setCopiedStory] = useState(false);

  const API_URL =
    `${import.meta.env.VITE_FRONTEND_WRITER_URL}predict_story_virality_with_rewrite` || 
    "http://localhost:8000/api/virality/predict";

  const virality = useMemo(() => {
    return (resp?.virality as ViralityPayload) ?? null;
  }, [resp]);

  const signals = useMemo(() => safeSignals(virality), [virality]);

  const predLabel = (virality?.pred_label ?? "").toString();
  const viral = isViralLabel(predLabel);

  const badgeClass = viral ? "badge badge-viral" : "badge badge-nonviral";

  async function onSubmit() {
    setErr(null);
    setResp(null);
    setCopiedInsights(false);
    setCopiedStory(false);

    if (!scriptText.trim()) {
      setErr("Please paste a story (script_text) before submitting.");
      return;
    }

    setLoading(true);
    try {
      const payload = {
        script_text: scriptText,
        rewrite_story: rewriteStory,
      };

      const r = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const json = (await r.json()) as ApiResponse;

      if (!r.ok || json?.success === false) {
        const msg = json?.message || `Request failed (${r.status})`;
        throw new Error(msg);
      }

      setResp(json);
    } catch (e: any) {
      setErr(e?.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  function copyText(text: string, type: 'insights' | 'story') {
    if (!text) return;
    
    navigator.clipboard.writeText(text).then(() => {
      if (type === 'insights') {
        setCopiedInsights(true);
        setTimeout(() => setCopiedInsights(false), 2000);
      } else {
        setCopiedStory(true);
        setTimeout(() => setCopiedStory(false), 2000);
      }
    }).catch(() => {});
  }

  const leftList = signals.top_push_viral ?? [];
  const rightList = signals.top_push_nonviral ?? [];

  return (
    <div className="page-wrap">
      <div className="header">
        <h1 className="title">Story Virality Predictor</h1>
        <p className="subtitle">
          Analyze your story's viral potential with AI-powered insights and optimization suggestions
        </p>
      </div>

      <div className="card">
        <div className="controls-row">
          <label className="toggle">
            <input
              type="checkbox"
              checked={rewriteStory}
              onChange={(e) => setRewriteStory(e.target.checked)}
              disabled={loading}
            />
            <span className="toggle-label">
              Enable AI rewrite & optimization insights
            </span>
          </label>

          <button className="btn" onClick={onSubmit} disabled={loading}>
            {loading ? "Analyzing" : "Predict Virality"}
          </button>
        </div>

        <label className="field-label">Original Story</label>
        <textarea
          className="textarea"
          value={scriptText}
          onChange={(e) => setScriptText(e.target.value)}
          placeholder="Paste your story here to analyze its viral potential..."
          rows={12}
          disabled={loading}
        />

        {err && <div className="alert alert-error">⚠️ {err}</div>}
      </div>

      {resp && (
        <div className="results">
          <div className="results-container">
            {/* Left Sidebar - Sticky Summary */}
            <div className="results-sidebar">
              <div className={`card result-summary ${viral ? "glow-viral" : "glow-nonviral"}`}>
                <div className="label-row">
                  <span className={badgeClass}>
                    {viral ? "✅ VIRAL" : "⚠️ NON-VIRAL"}
                  </span>
                  {resp?.message && (
                    <span className="muted small">{resp.message}</span>
                  )}
                </div>

                <div className="summary-row">
                  <div className="summary-left">
                    <div className="metrics">
                      <div className="metric">
                        <div className="metric-name">Virality Score</div>
                        <div className="metric-val" style={{ color: viral ? '#22c55e' : '#ef4444' }}>
                          {fmt((virality as any)?.prob_viral, 4)}
                        </div>
                      </div>
                      <div className="metric">
                        <div className="metric-name">Prediction</div>
                        <div className="metric-val">{predLabel || "—"}</div>
                      </div>
                    </div>
                  </div>

                  <div className="summary-right">
                    <div className="debug">
                      <div className="debug-title">Debug Probabilities</div>
                      <div className="debug-grid">
                        <div className="debug-item">
                          <span className="muted">p_bal</span>
                          <span>{fmt((virality as any)?.debug_probs?.p_bal, 6)}</span>
                        </div>
                        <div className="debug-item">
                          <span className="muted">p_imbal</span>
                          <span>{fmt((virality as any)?.debug_probs?.p_imbal, 6)}</span>
                        </div>
                        <div className="debug-item">
                          <span className="muted">alpha</span>
                          <span>{fmt((virality as any)?.debug_probs?.alpha, 3)}</span>
                        </div>
                        <div className="debug-item">
                          <span className="muted">beta</span>
                          <span>{fmt((virality as any)?.debug_probs?.beta, 3)}</span>
                        </div>
                        <div className="debug-item">
                          <span className="muted">threshold</span>
                          <span>{fmt((virality as any)?.debug_probs?.threshold, 3)}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Right Main Content - Scrollable */}
            <div className="results-main">

          {/* Signals Analysis */}
          <div className="card">
            <h2 className="section-title">📊 Viral Signals Analysis</h2>

            <div className="signals-grid">
              <div className="signals-col">
                <div className="signals-head signals-viral">
                  ⬆️ Top Viral Indicators
                </div>
                {leftList.length === 0 ? (
                  <div className="muted small">No viral signals detected.</div>
                ) : (
                  <ul className="signals-list">
                    {leftList.map((it, idx) => (
                      <li key={`v-${idx}`} className="signals-item">
                        <span className="signals-ngram">"{it.ngram}"</span>
                        <span className="signals-contrib" style={{ color: '#22c55e' }}>
                          +{fmt(it.contrib, 6)}
                        </span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>

              <div className="signals-col">
                <div className="signals-head signals-nonviral">
                  ⬇️ Top Non-Viral Indicators
                </div>
                {rightList.length === 0 ? (
                  <div className="muted small">No non-viral signals detected.</div>
                ) : (
                  <ul className="signals-list">
                    {rightList.map((it, idx) => (
                      <li key={`nv-${idx}`} className="signals-item">
                        <span className="signals-ngram">"{it.ngram}"</span>
                        <span className="signals-contrib" style={{ color: '#ef4444' }}>
                          {fmt(it.contrib, 6)}
                        </span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          </div>

          {/* AI Insights & Rewrite */}
          {(typeof resp.rewrite !== "undefined" ||
            resp.new_story ||
            resp.rewrite_details) && (
            <>
              {/* Agent Insights */}
              {resp.rewrite_details && resp.rewrite_details.trim() !== "" && (
                <div className="card">
                  <div className="rewrite-header">
                    <div className="rewrite-title">💡 AI Optimization Insights</div>
                    <button
                      className="btn btn-secondary"
                      onClick={() => copyText(resp.rewrite_details || "", 'insights')}
                      type="button"
                    >
                      {copiedInsights ? '✓ Copied!' : '📋 Copy Insights'}
                    </button>
                  </div>
                  {renderInsightsContent(resp.rewrite_details)}
                </div>
              )}

              {/* Optimized Story */}
              {resp.new_story && resp.new_story.trim() !== "" && (
                <div className="card">
                  <div className="new-story">
                    <div className="rewrite-header">
                      <div className="rewrite-title">✨ Optimized Story</div>
                      <button
                        className="btn btn-secondary"
                        onClick={() => copyText(resp.new_story || "", 'story')}
                        type="button"
                      >
                        {copiedStory ? '✓ Copied!' : '📋 Copy Story'}
                      </button>
                    </div>
                    <textarea 
                      className="textarea" 
                      value={resp.new_story} 
                      readOnly 
                      rows={14}
                    />
                  </div>
                </div>
              )}
            </>
          )}
          </div>
          </div>
        </div>
      )}
    </div>
  );
}