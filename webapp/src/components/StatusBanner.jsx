import React from "react";

const CONFIG = {
  UPLOADING:{ color: "var(--blue)",  bg: "var(--blue-bg)",  border: "rgba(0,112,243,0.18)",   label: "Uploading scan…",   animate: true  },
  PENDING:  { color: "var(--amber)", bg: "var(--amber-bg)", border: "rgba(245,158,11,0.18)",  label: "Queued…",           animate: false },
  PROGRESS: { color: "var(--blue)",  bg: "var(--blue-bg)",  border: "rgba(0,112,243,0.18)",   label: "Running inference…", animate: true  },
  SUCCESS:  { color: "var(--green)", bg: "var(--green-bg)", border: "rgba(34,197,94,0.18)",   label: "Complete",           animate: false },
  FAILURE:  { color: "var(--red)",   bg: "var(--red-bg)",   border: "rgba(239,68,68,0.18)",   label: "Failed",             animate: false },
};

function formatProgressStep(step) {
  if (!step) return null;
  if (step === "detection") return "Running detection…";
  if (step === "rendering") return "Rendering slices…";
  return step;
}

function formatElapsed(elapsedSeconds) {
  if (elapsedSeconds == null) return null;
  const total = Math.max(0, Math.floor(elapsedSeconds));
  const minutes = String(Math.floor(total / 60)).padStart(2, "0");
  const seconds = String(total % 60).padStart(2, "0");
  return `${minutes}:${seconds}`;
}

export default function StatusBanner({ status, step, error, elapsedSeconds }) {
  if (!status) return null;

  const cfg = CONFIG[status] ?? {
    color: "var(--text-2)",
    bg: "transparent",
    border: "var(--border)",
    label: status,
    animate: false,
  };
  const label = status === "PROGRESS" ? formatProgressStep(step) ?? cfg.label : cfg.label;
  const elapsed = formatElapsed(elapsedSeconds);

  return (
    <div style={{
      display: "flex",
      alignItems: "flex-start",
      gap: 8,
      padding: "7px 10px",
      borderRadius: 5,
      background: cfg.bg,
      border: `1px solid ${cfg.border}`,
      marginBottom: 12,
    }}>
      <div style={{
        width: 6,
        height: 6,
        borderRadius: "50%",
        background: cfg.color,
        flexShrink: 0,
        marginTop: 4,
        animation: cfg.animate ? "pulse 1.4s ease-in-out infinite" : "none",
      }} />
      <div style={{ minWidth: 0 }}>
        <div style={{ fontSize: 11, color: cfg.color, fontWeight: 500, letterSpacing: "0.01em" }}>
          {label}
        </div>
        {elapsed && (
          <div style={{
            marginTop: 4,
            fontSize: 10,
            color: "var(--text-2)",
            fontFamily: "var(--mono)",
            letterSpacing: "0.04em",
          }}>
            ELAPSED {elapsed}
          </div>
        )}
        {status === "FAILURE" && error && (
          <div style={{
            marginTop: 4,
            fontSize: 10,
            color: "var(--text-2)",
            fontFamily: "var(--mono)",
            lineHeight: 1.5,
            whiteSpace: "pre-wrap",
            wordBreak: "break-word",
          }}>
            {error}
          </div>
        )}
      </div>
    </div>
  );
}
