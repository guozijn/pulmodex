/**
 * CT slice viewer component.
 *
 * Uses PNG layers from the API: a base CT slice and an optional transparent overlay.
 */
import React, { useState, useEffect, useRef } from "react";

const ORIENTATION = {
  axial:    { top: "A", bottom: "P", left: "R", right: "L" },
  coronal:  { top: "S", bottom: "I", left: "R", right: "L" },
  sagittal: { top: "S", bottom: "I", left: "A", right: "P" },
};

function LungPlaceholder() {
  return (
    <svg width="64" height="64" viewBox="0 0 64 64" fill="none" aria-hidden="true" opacity="0.12">
      <path d="M32 8v36" stroke="white" strokeWidth="2.5" strokeLinecap="round"/>
      <path
        d="M32 16c-5 0-10 3.5-13 9-3 5.5-3 11-6 17-2.5 5-4 9-3 11.5 1 2.5 5 2.5 9.5 0 3.5-2 5.5-5.5 7-10 1.5-4.5 1.5-9 2-14"
        stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
      />
      <path
        d="M32 16c5 0 10 3.5 13 9 3 5.5 3 11 6 17 2.5 5 4 9 3 11.5-1 2.5-5 2.5-9.5 0-3.5-2-5.5-5.5-7-10-1.5-4.5-1.5-9-2-14"
        stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
      />
    </svg>
  );
}

export default function Viewer({
  baseSliceUrl,
  overlaySliceUrl,
  sliceIdx,
  onSliceChange,
  view = "axial",
  maxSliceIdx = null,
  showOverlay = false,
  overlayOpacity = 0.30,
}) {
  const [baseImgSrc, setBaseImgSrc] = useState(null);
  const [overlayImgSrc, setOverlayImgSrc] = useState(null);
  const [error, setError] = useState(false);
  const containerRef = useRef(null);

  useEffect(() => {
    if (!baseSliceUrl) {
      setBaseImgSrc(null);
      setOverlayImgSrc(null);
      return;
    }
    setError(false);
    setBaseImgSrc(baseSliceUrl);
    setOverlayImgSrc(overlaySliceUrl);
  }, [baseSliceUrl, overlaySliceUrl, sliceIdx]);

  const handleWheel = (e) => {
    e.preventDefault();
    onSliceChange((prev) => {
      const next = Math.max(0, prev + (e.deltaY > 0 ? 1 : -1));
      if (typeof maxSliceIdx === "number") {
        return Math.min(maxSliceIdx, next);
      }
      return next;
    });
  };

  const orient = ORIENTATION[view] ?? ORIENTATION.axial;

  return (
    <div
      ref={containerRef}
      onWheel={handleWheel}
      style={{
        flex: 1,
        position: "relative",
        background: "#000",
        overflow: "hidden",
        userSelect: "none",
      }}
    >
      {baseImgSrc && !error ? (
        <>
          <div style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}>
            <img
              src={baseImgSrc}
              alt={`slice ${sliceIdx}`}
              onError={() => setError(true)}
              style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain" }}
            />
            {showOverlay && overlayImgSrc ? (
              <img
                src={overlayImgSrc}
                alt=""
                aria-hidden="true"
                style={{
                  position: "absolute",
                  maxWidth: "100%",
                  maxHeight: "100%",
                  objectFit: "contain",
                  opacity: overlayOpacity,
                }}
              />
            ) : null}
          </div>

          <div style={{ position: "absolute", inset: 0, pointerEvents: "none" }}>
            {/* Crosshair lines */}
            <div style={{
              position: "absolute",
              top: "50%", left: 0, right: 0,
              height: 1,
              background: "rgba(0,180,216,0.12)",
              transform: "translateY(-0.5px)",
            }} />
            <div style={{
              position: "absolute",
              left: "50%", top: 0, bottom: 0,
              width: 1,
              background: "rgba(0,180,216,0.12)",
              transform: "translateX(-0.5px)",
            }} />

            {/* Orientation labels */}
            {[
              { label: orient.top,    style: { top: 14,   left: "50%", transform: "translateX(-50%)" } },
              { label: orient.bottom, style: { bottom: 14, left: "50%", transform: "translateX(-50%)" } },
              { label: orient.left,   style: { left: 14,  top: "50%",  transform: "translateY(-50%)" } },
              { label: orient.right,  style: { right: 14, top: "50%",  transform: "translateY(-50%)" } },
            ].map(({ label, style }) => (
              <div key={label + JSON.stringify(style)} style={{
                position: "absolute",
                ...style,
                fontSize: 10,
                fontFamily: "var(--mono)",
                color: "rgba(0,180,216,0.4)",
                letterSpacing: "0.1em",
                fontWeight: 600,
              }}>
                {label}
              </div>
            ))}

            {/* Top-left scan metadata */}
            <div style={{
              position: "absolute",
              top: 10, left: 10,
              fontSize: 9,
              fontFamily: "var(--mono)",
              color: "rgba(255,255,255,0.18)",
              lineHeight: 1.8,
              letterSpacing: "0.04em",
            }}>
              <div>PULMODEX v1.0</div>
              <div>WIN -1000/400 HU</div>
              <div>{showOverlay ? `OVERLAY ${Math.round(overlayOpacity * 100)}%` : "OVERLAY OFF"}</div>
            </div>

            {/* Slice index — bottom left */}
            <div style={{
              position: "absolute",
              bottom: 10, left: 10,
              fontSize: 10,
              fontFamily: "var(--mono)",
              color: "rgba(255,255,255,0.3)",
              letterSpacing: "0.05em",
            }}>
              Slice {sliceIdx} · scroll to navigate
            </div>
          </div>
        </>
      ) : (
        <div style={{
          position: "absolute",
          inset: 0,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: 14,
        }}>
          <LungPlaceholder />
          <div style={{ fontSize: 13, color: "rgba(255,255,255,0.18)", letterSpacing: "0.02em" }}>
            {baseSliceUrl ? (error ? "Slice not available" : "Loading…") : "Upload a scan to begin"}
          </div>
          {!baseSliceUrl && (
            <div style={{
              fontSize: 9,
              fontFamily: "var(--mono)",
              color: "rgba(255,255,255,0.08)",
              letterSpacing: "0.12em",
              textTransform: "uppercase",
            }}>
              LUNA16 · DICOM · MHD
            </div>
          )}
        </div>
      )}
    </div>
  );
}
