/**
 * CT slice viewer component.
 *
 * Uses a simple <img> PNG fallback (Cornerstone.js requires DICOM buffers;
 * pre-rendered PNGs from the API are already windowed and annotated).
 * The saliency overlay is already baked into the PNG by the renderer.
 */
import React, { useState, useEffect, useRef } from "react";

export default function Viewer({ sliceUrl, sliceIdx, onSliceChange, saliencyAlpha }) {
  const [imgSrc, setImgSrc] = useState(null);
  const [error, setError] = useState(false);
  const containerRef = useRef(null);

  useEffect(() => {
    if (!sliceUrl) { setImgSrc(null); return; }
    setError(false);
    setImgSrc(sliceUrl);
  }, [sliceUrl, sliceIdx, saliencyAlpha]);

  const handleWheel = (e) => {
    e.preventDefault();
    onSliceChange((prev) => Math.max(0, prev + (e.deltaY > 0 ? 1 : -1)));
  };

  return (
    <div
      ref={containerRef}
      onWheel={handleWheel}
      style={{
        flex: 1,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "#000",
        overflow: "hidden",
        position: "relative",
        userSelect: "none",
      }}
    >
      {imgSrc && !error ? (
        <img
          src={imgSrc}
          alt={`slice ${sliceIdx}`}
          onError={() => setError(true)}
          style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain" }}
        />
      ) : (
        <div style={{ color: "#4a5568", fontSize: 14 }}>
          {sliceUrl ? (error ? "Slice not available" : "Loading…") : "Upload a scan to begin"}
        </div>
      )}
      {imgSrc && (
        <div style={{
          position: "absolute", bottom: 8, right: 12,
          color: "#718096", fontSize: 11,
        }}>
          Slice {sliceIdx} · scroll to navigate
        </div>
      )}
    </div>
  );
}
