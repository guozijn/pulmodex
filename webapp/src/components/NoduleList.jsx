import React from "react";

function riskLevel(prob) {
  if (prob >= 0.7) return { label: "HIGH", color: "var(--red)",   bg: "var(--red-bg)"   };
  if (prob >= 0.4) return { label: "MOD",  color: "var(--amber)", bg: "var(--amber-bg)" };
  return              { label: "LOW",  color: "var(--green)", bg: "var(--green-bg)" };
}

export default function NoduleList({ candidates, selected, onSelect }) {
  if (!candidates || candidates.length === 0) return null;

  return (
    <div>
      {candidates.map((c, i) => {
        const isSelected = selected === c;
        const prob = c.fp_prob ?? c.prob ?? 0;
        const { label, color, bg } = riskLevel(prob);

        return (
          <div
            key={i}
            data-testid="nodule-item"
            onClick={() => onSelect(c, i)}
            style={{
              padding: "10px 12px",
              borderRadius: "var(--radius)",
              marginBottom: 4,
              cursor: "pointer",
              background: isSelected ? "var(--bg-3)" : "var(--bg-2)",
              border: `1px solid ${isSelected ? "var(--border-1)" : "var(--border)"}`,
              transition: "all 0.12s",
            }}
          >
            {/* Header row */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 7 }}>
              <span style={{ fontSize: 12, fontWeight: 500, color: "var(--text)" }}>Nodule {i + 1}</span>
              <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
                <span style={{
                  fontSize: 9,
                  fontWeight: 600,
                  letterSpacing: "0.08em",
                  color,
                  background: bg,
                  padding: "1px 5px",
                  borderRadius: 2,
                  fontFamily: "var(--mono)",
                }}>
                  {label}
                </span>
                <span style={{ fontSize: 12, fontWeight: 600, color, fontFamily: "var(--mono)" }}>
                  {(prob * 100).toFixed(0)}%
                </span>
              </div>
            </div>

            {/* Confidence bar */}
            <div style={{
              height: 2,
              background: "var(--bg-hover)",
              borderRadius: 1,
              marginBottom: 8,
              overflow: "hidden",
            }}>
              <div style={{
                height: "100%",
                width: `${prob * 100}%`,
                background: color,
                borderRadius: 1,
                transition: "width 0.3s ease",
              }} />
            </div>

            {/* Coordinates & diameter */}
            <div style={{
              display: "flex",
              justifyContent: "space-between",
              fontSize: 10,
              fontFamily: "var(--mono)",
              color: "var(--text-3)",
            }}>
              <span>dia {c.diameter_mm?.toFixed(1)} mm</span>
              <span>
                ({c.coordX?.toFixed(0)}, {c.coordY?.toFixed(0)}, {c.coordZ?.toFixed(0)})
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
