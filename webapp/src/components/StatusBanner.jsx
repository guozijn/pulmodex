import React from "react";

const COLORS = {
  PENDING: "#d69e2e",
  PROGRESS: "#3182ce",
  SUCCESS: "#38a169",
  FAILURE: "#e53e3e",
};

const LABELS = {
  PENDING: "Queued…",
  PROGRESS: "Running inference…",
  SUCCESS: "Complete",
  FAILURE: "Failed",
};

export default function StatusBanner({ status }) {
  if (!status) return null;
  return (
    <div
      style={{
        padding: "6px 10px",
        borderRadius: 6,
        background: COLORS[status] + "22",
        border: `1px solid ${COLORS[status]}`,
        color: COLORS[status],
        fontSize: 12,
        marginBottom: 12,
        fontWeight: 600,
      }}
    >
      {LABELS[status] ?? status}
    </div>
  );
}
