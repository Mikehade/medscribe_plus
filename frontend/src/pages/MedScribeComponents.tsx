/**
 * Shared rendering helpers for MedScribe+ pages.
 * Handles graceful display of nested patient context data.
 */

interface Medication {
  name?: string;
  dose?: string;
  frequency?: string;
  [key: string]: any;
}

interface PriorNote {
  date?: string;
  summary?: string;
  [key: string]: any;
}

interface PatientContextData {
  name?: string;
  dob?: string;
  allergies?: string[];
  medications?: Medication[];
  conditions?: string[];
  last_visit?: string;
  prior_notes?: PriorNote[];
  [key: string]: any;
}

interface SOAPNote {
  subjective?: string;
  objective?: string;
  assessment?: string;
  plan?: string;
  icd10_codes?: string[];
  cpt_codes?: string[];
  medications_mentioned?: string[];
  follow_up?: string;
  conditions_detected?: string[];
  [key: string]: any;
}

interface EvaluationScores {
  completeness?: number;
  hallucination_risk?: string;
  drug_safety?: number;
  guideline_alignment?: number;
  drug_interactions?: Array<{ drug: string; severity: string; description: string }>;
  guideline_suggestions?: string[];
  overall_ready?: boolean;
  cpt_codes?: string[];
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function renderValue(val: any): string {
  if (val === null || val === undefined) return "—";
  if (typeof val === "string") return val || "—";
  if (typeof val === "number" || typeof val === "boolean") return String(val);
  if (Array.isArray(val)) {
    if (val.length === 0) return "—";
    // Array of primitives
    if (typeof val[0] !== "object") return val.join(", ");
  }
  return ""; // signals "render structured"
}

// ── PatientContextCard ────────────────────────────────────────────────────────

export function PatientContextCard({ ctx }: { ctx: PatientContextData }) {
  if (!ctx || Object.keys(ctx).length === 0) return null;

  const simpleFields: [string, string][] = [];
  const medications = ctx.medications as Medication[] | undefined;
  const priorNotes = ctx.prior_notes as PriorNote[] | undefined;
  const allergies = ctx.allergies as string[] | undefined;
  const conditions = ctx.conditions as string[] | undefined;

  // Collect simple key-value fields
  for (const [k, v] of Object.entries(ctx)) {
    if (["medications", "prior_notes", "allergies", "conditions"].includes(k)) continue;
    const rendered = renderValue(v);
    if (rendered) simpleFields.push([k, rendered]);
  }

  const labelMap: Record<string, string> = {
    name: "Name",
    dob: "Date of Birth",
    last_visit: "Last Visit",
  };

  return (
    <div className="ms-patient-ctx">
      <div className="ms-soap-label" style={{ marginBottom: 10 }}>Patient Context</div>

      {/* Simple fields */}
      {simpleFields.map(([k, v]) => (
        <div key={k} className="ms-ctx-row">
          <span className="ms-ctx-key">{labelMap[k] || k.replace(/_/g, " ")}</span>
          <span className="ms-ctx-val">{v}</span>
        </div>
      ))}

      {/* Allergies */}
      {allergies && allergies.length > 0 && (
        <div className="ms-ctx-row ms-ctx-row-block">
          <span className="ms-ctx-key">Allergies</span>
          <div className="ms-ctx-tags">
            {allergies.map((a, i) => (
              <span key={i} className="ms-ctx-tag ms-ctx-tag-red">{a}</span>
            ))}
          </div>
        </div>
      )}

      {/* Conditions */}
      {conditions && conditions.length > 0 && (
        <div className="ms-ctx-row ms-ctx-row-block">
          <span className="ms-ctx-key">Conditions</span>
          <div className="ms-ctx-tags">
            {conditions.map((c, i) => (
              <span key={i} className="ms-ctx-tag ms-ctx-tag-amber">{c}</span>
            ))}
          </div>
        </div>
      )}

      {/* Medications */}
      {medications && medications.length > 0 && (
        <div className="ms-ctx-section">
          <span className="ms-ctx-key" style={{ marginBottom: 6, display: "block" }}>Medications</span>
          <div className="ms-med-list">
            {medications.map((med, i) => (
              <div key={i} className="ms-med-item">
                <span className="ms-med-name">{med.name || "Unknown"}</span>
                {med.dose && <span className="ms-med-detail">{med.dose}</span>}
                {med.frequency && <span className="ms-med-freq">{med.frequency}</span>}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Prior notes */}
      {priorNotes && priorNotes.length > 0 && (
        <div className="ms-ctx-section">
          <span className="ms-ctx-key" style={{ marginBottom: 6, display: "block" }}>Prior Notes</span>
          {priorNotes.map((note, i) => (
            <div key={i} className="ms-prior-note">
              {note.date && <span className="ms-prior-date">{note.date}</span>}
              {note.summary && <p className="ms-prior-summary">{note.summary}</p>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── SOAPExtras ────────────────────────────────────────────────────────────────
// Renders medications_mentioned, conditions_detected, follow_up from SOAP note

export function SOAPExtras({ soap }: { soap: SOAPNote }) {
  const hasMeds = soap.medications_mentioned && soap.medications_mentioned.length > 0;
  const hasConditions = soap.conditions_detected && soap.conditions_detected.length > 0;
  const hasFollowUp = !!soap.follow_up;

  if (!hasMeds && !hasConditions && !hasFollowUp) return null;

  return (
    <div className="ms-soap-extras">
      {hasFollowUp && (
        <div className="ms-soap-section ms-followup-section">
          <div className="ms-soap-label">Follow-up</div>
          <p className="ms-soap-text">{soap.follow_up}</p>
        </div>
      )}
      {hasConditions && (
        <div className="ms-soap-section">
          <div className="ms-soap-label">Conditions Detected</div>
          <div className="ms-codes-row" style={{ marginTop: 4 }}>
            {soap.conditions_detected!.map((c) => (
              <span key={c} className="ms-code-badge ms-condition-badge">{c}</span>
            ))}
          </div>
        </div>
      )}
      {hasMeds && (
        <div className="ms-soap-section">
          <div className="ms-soap-label">Medications Mentioned</div>
          <div className="ms-codes-row" style={{ marginTop: 4 }}>
            {soap.medications_mentioned!.map((m) => (
              <span key={m} className="ms-code-badge ms-med-badge">{m}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export type { SOAPNote, EvaluationScores, PatientContextData, Medication, PriorNote };