import React, { useRef } from "react";

function CTScanIcon({ size = 32, color = "currentColor" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" fill="none" aria-hidden="true">
      <circle cx="16" cy="16" r="12" stroke={color} strokeWidth="1" opacity="0.2"/>
      <circle cx="16" cy="16" r="8"  stroke={color} strokeWidth="1" opacity="0.4"/>
      <circle cx="16" cy="16" r="3.5" stroke={color} strokeWidth="1.2" opacity="0.8"/>
      <line x1="4"  y1="16" x2="13" y2="16" stroke={color} strokeWidth="0.8" opacity="0.35"/>
      <line x1="19" y1="16" x2="28" y2="16" stroke={color} strokeWidth="0.8" opacity="0.35"/>
      <line x1="16" y1="4"  x2="16" y2="13" stroke={color} strokeWidth="0.8" opacity="0.35"/>
      <line x1="16" y1="19" x2="16" y2="28" stroke={color} strokeWidth="0.8" opacity="0.35"/>
    </svg>
  );
}

export default function UploadZone({ onUpload, disabled, uploading = false }) {
  const inputRef = useRef(null);

  const handleDrop = (e) => {
    e.preventDefault();
    if (disabled) return;
    const file = e.dataTransfer.files[0];
    if (file) onUpload(file);
  };

  const handleChange = (e) => {
    const file = e.target.files[0];
    if (file) onUpload(file);
    e.target.value = "";
  };

  return (
    <div
      onDrop={handleDrop}
      onDragOver={(e) => e.preventDefault()}
      onClick={() => !disabled && inputRef.current?.click()}
      style={{
        border: `1px dashed ${disabled ? "rgba(255,255,255,0.05)" : "rgba(255,255,255,0.11)"}`,
        borderRadius: 8,
        padding: "18px 14px",
        textAlign: "center",
        cursor: disabled ? "not-allowed" : "pointer",
        marginBottom: 12,
        opacity: disabled ? 0.45 : 1,
        background: "rgba(255,255,255,0.01)",
        transition: "border-color 0.15s, opacity 0.15s",
      }}
    >
      <div style={{ marginBottom: 10, color: "var(--teal)", opacity: 0.65, display: "flex", justifyContent: "center" }}>
        <div style={{ animation: uploading ? "spin 1s linear infinite" : "none" }}>
          <CTScanIcon size={28} />
        </div>
      </div>
      <p style={{ fontSize: 12, color: "var(--text-2)", lineHeight: 1.5 }}>
        {uploading ? "Uploading scan…" : disabled ? "Processing…" : "Drop a .zip DICOM series or .nii.gz volume here or click to upload"}
      </p>
      {!disabled && !uploading && (
        <div style={{ marginTop: 9, display: "flex", justifyContent: "center", gap: 4 }}>
          {[".ZIP", ".NII.GZ", "CT"].map((fmt) => (
            <span
              key={fmt}
              style={{
                fontSize: 9,
                fontFamily: "var(--mono)",
                color: "var(--text-3)",
                border: "1px solid var(--border)",
                padding: "1px 5px",
                borderRadius: 2,
                letterSpacing: "0.06em",
              }}
            >
              {fmt}
            </span>
          ))}
        </div>
      )}
      <input
        ref={inputRef}
        type="file"
        accept=".zip,.nii.gz,application/gzip"
        style={{ display: "none" }}
        onChange={handleChange}
      />
    </div>
  );
}
