import React from "react";

export default function NoduleList({ candidates, selected, onSelect }) {
  if (!candidates || candidates.length === 0) return null;

  return (
    <div>
      <h3 style={{ fontSize: 13, color: "#a0aec0", marginBottom: 8, textTransform: "uppercase", letterSpacing: 1 }}>
        Detected Nodules
      </h3>
      {candidates.map((c, i) => {
        const isSelected = selected === c;
        const prob = c.fp_prob ?? c.prob ?? 0;
        const isConfident = prob >= 0.5;
        return (
          <div
            key={i}
            onClick={() => onSelect(c, i)}
            style={{
              padding: "8px 10px",
              borderRadius: 6,
              marginBottom: 6,
              cursor: "pointer",
              background: isSelected ? "#2d3748" : "transparent",
              border: `1px solid ${isSelected ? "#4a5568" : "transparent"}`,
              transition: "background 0.15s",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: 13, fontWeight: 600 }}>Nodule {i + 1}</span>
              <span
                style={{
                  fontSize: 11,
                  fontWeight: 700,
                  color: isConfident ? "#fc8181" : "#f6ad55",
                  background: isConfident ? "#fc818122" : "#f6ad5522",
                  padding: "2px 6px",
                  borderRadius: 4,
                }}
              >
                {(prob * 100).toFixed(0)}%
              </span>
            </div>
            <div style={{ fontSize: 11, color: "#718096", marginTop: 3 }}>
              ⌀ {c.diameter_mm?.toFixed(1)} mm &nbsp;·&nbsp;
              ({c.coordX?.toFixed(1)}, {c.coordY?.toFixed(1)}, {c.coordZ?.toFixed(1)})
            </div>
          </div>
        );
      })}
    </div>
  );
}
