import React, { useRef } from "react";

export default function UploadZone({ onUpload, disabled }) {
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
  };

  return (
    <div
      onDrop={handleDrop}
      onDragOver={(e) => e.preventDefault()}
      onClick={() => !disabled && inputRef.current?.click()}
      style={{
        border: "2px dashed #4a5568",
        borderRadius: 8,
        padding: 20,
        textAlign: "center",
        cursor: disabled ? "not-allowed" : "pointer",
        marginBottom: 16,
        opacity: disabled ? 0.5 : 1,
        transition: "border-color 0.2s",
      }}
    >
      <p style={{ fontSize: 13, color: "#a0aec0" }}>
        {disabled ? "Processing…" : "Drop .mhd file here or click to upload"}
      </p>
      <input
        ref={inputRef}
        type="file"
        accept=".mhd,.raw"
        style={{ display: "none" }}
        onChange={handleChange}
      />
    </div>
  );
}
