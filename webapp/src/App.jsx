import React, { useState, useCallback, useEffect, useRef } from "react";
import Viewer from "./components/Viewer";
import NoduleList from "./components/NoduleList";
import UploadZone from "./components/UploadZone";
import StatusBanner from "./components/StatusBanner";

const API = "/api";
const VIEW_NAMES = ["axial", "coronal", "sagittal"];

function LungIcon({ size = 18 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 20 20" fill="none" aria-hidden="true">
      <path
        d="M10 2.5v11"
        stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"
      />
      <path
        d="M10 5c-1.5 0-3 1-4 3-1 2-1 4-2 6-.8 1.5-1.2 2.8-.8 3.5.4.7 1.8.7 3 0 1.2-.7 1.8-2 2.2-3.5.4-1.5.4-3 .6-4.5"
        stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round"
      />
      <path
        d="M10 5c1.5 0 3 1 4 3 1 2 1 4 2 6 .8 1.5 1.2 2.8.8 3.5-.4.7-1.8.7-3 0-1.2-.7-1.8-2-2.2-3.5-.4-1.5-.4-3-.6-4.5"
        stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round"
      />
    </svg>
  );
}

function SectionLabel({ children }) {
  return (
    <div style={{
      fontSize: 10,
      fontWeight: 600,
      letterSpacing: "0.1em",
      textTransform: "uppercase",
      color: "var(--text-3)",
      marginBottom: 8,
    }}>
      {children}
    </div>
  );
}

function ViewTabs({ active, onChange }) {
  const subtitles = { axial: "Transverse", coronal: "Frontal", sagittal: "Lateral" };
  return (
    <div style={{
      display: "flex",
      alignItems: "center",
      height: 40,
      padding: "0 14px",
      gap: 2,
      background: "var(--bg-1)",
      borderBottom: "1px solid var(--border)",
      flexShrink: 0,
    }}>
      {["axial", "coronal", "sagittal"].map((v) => (
        <button
          key={v}
          onClick={() => onChange(v)}
          style={{
            padding: "3px 11px",
            background: active === v ? "var(--bg-3)" : "transparent",
            color: active === v ? "var(--text)" : "var(--text-2)",
            border: `1px solid ${active === v ? "var(--border-1)" : "transparent"}`,
            borderRadius: "var(--radius)",
            cursor: "pointer",
            fontSize: 11,
            fontFamily: "inherit",
            fontWeight: active === v ? 500 : 400,
            letterSpacing: "0.06em",
            textTransform: "uppercase",
            transition: "all 0.12s",
          }}
        >
          {v}
        </button>
      ))}
      <div style={{ flex: 1 }} />
      {active && (
        <span style={{
          fontSize: 9,
          color: "var(--text-3)",
          fontFamily: "var(--mono)",
          letterSpacing: "0.1em",
          textTransform: "uppercase",
        }}>
          {subtitles[active]}
        </span>
      )}
    </div>
  );
}

function OverlayControls({ showOverlay, onToggleOverlay, overlayOpacity, onOpacityChange }) {
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 7 }}>
        <label style={{ fontSize: 11, color: "var(--text-2)" }}>Heatmap overlay</label>
        <button
          type="button"
          onClick={onToggleOverlay}
          style={{
            fontSize: 10,
            fontFamily: "var(--mono)",
            color: showOverlay ? "var(--teal)" : "var(--text-3)",
            background: "transparent",
            border: "1px solid var(--border)",
            borderRadius: 4,
            padding: "2px 6px",
            cursor: "pointer",
          }}
        >
          {showOverlay ? "ON" : "OFF"}
        </button>
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 7 }}>
        <label style={{ fontSize: 11, color: "var(--text-2)" }}>Opacity</label>
        <span style={{ fontSize: 11, color: "var(--teal)", fontFamily: "var(--mono)" }}>
          {Math.round(overlayOpacity * 100)}%
        </span>
      </div>
      <input
        type="range"
        min={0}
        max={1}
        step={0.05}
        value={overlayOpacity}
        onChange={(e) => onOpacityChange(Number(e.target.value))}
        disabled={!showOverlay}
        style={{
          width: "100%",
          opacity: showOverlay ? 1 : 0.4,
          cursor: showOverlay ? "pointer" : "not-allowed",
        }}
      />
    </div>
  );
}

export default function App() {
  const [jobId, setJobId] = useState(null);
  const [seriesuid, setSeriesuid] = useState(null);
  const [status, setStatus] = useState(null);
  const [progressStep, setProgressStep] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);
  const [startedAt, setStartedAt] = useState(null);
  const [finishedAt, setFinishedAt] = useState(null);
  const [clockNow, setClockNow] = useState(Date.now());
  const [report, setReport] = useState(null);
  const [activeView, setActiveView] = useState("axial");
  const [sliceIdx, setSliceIdx] = useState(0);
  const [showOverlay, setShowOverlay] = useState(true);
  const [overlayOpacity, setOverlayOpacity] = useState(0.30);
  const [selectedNodule, setSelectedNodule] = useState(null);
  const [sliceCatalog, setSliceCatalog] = useState({});
  const [history, setHistory] = useState([]);
  const pollRef = useRef(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const resetScanState = useCallback(() => {
    setSeriesuid(null);
    setReport(null);
    setSelectedNodule(null);
    setSliceCatalog({});
    setSliceIdx(0);
    setShowOverlay(true);
    setOverlayOpacity(0.30);
  }, []);

  useEffect(() => stopPolling, [stopPolling]);

  useEffect(() => {
    if (!startedAt || finishedAt || !["PENDING", "PROGRESS"].includes(status ?? "")) {
      return undefined;
    }

    const timer = window.setInterval(() => {
      setClockNow(Date.now());
    }, 1000);

    return () => window.clearInterval(timer);
  }, [finishedAt, startedAt, status]);

  const loadHistory = useCallback(async () => {
    try {
      const res = await fetch(`${API}/scans`);
      if (res.ok) { const data = await res.json(); setHistory(Array.isArray(data) ? data : []); }
    } catch (_) {}
  }, []);

  useEffect(() => { loadHistory(); }, [loadHistory]);

  const loadSliceCatalog = useCallback(async (uid) => {
    const entries = await Promise.all(
      VIEW_NAMES.map(async (view) => {
        const res = await fetch(`${API}/slices/${uid}/${view}/index`);
        if (!res.ok) return [view, { indices: [], count: 0 }];
        const data = await res.json();
        return [view, data];
      })
    );

    const nextCatalog = Object.fromEntries(entries);
    setSliceCatalog(nextCatalog);
    return nextCatalog;
  }, []);

  // Refresh history after each successful job
  const handleUploadSuccess = useCallback(() => { loadHistory(); }, [loadHistory]);

  const resolveCandidateSlice = useCallback((candidate, view, catalog = sliceCatalog) => {
    const preferred = candidate?.slice_indices?.[view];
    const indices = catalog?.[view]?.indices ?? [];
    if (typeof preferred === "number" && indices.includes(preferred)) {
      return preferred;
    }
    if (indices.length > 0) {
      return indices[Math.min(indices.length - 1, Math.max(0, Math.round(preferred ?? 0)))];
    }
    return 0;
  }, [sliceCatalog]);

  const handleUpload = useCallback(async (file) => {
    stopPolling();
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${API}/predict`, { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Upload failed");
      }

      setJobId(data.job_id);
      setSeriesuid(data.seriesuid);
      setStatus("PENDING");
      setProgressStep(null);
      setErrorMessage(null);
      const now = Date.now();
      setStartedAt(now);
      setFinishedAt(null);
      setClockNow(now);
      resetScanState();
      setSeriesuid(data.seriesuid);

      pollRef.current = setInterval(async () => {
        try {
          const r = await fetch(`${API}/status/${data.job_id}`);
          if (!r.ok) {
            throw new Error(`Status request failed (${r.status})`);
          }
          const s = await r.json();
          setStatus(s.state);
          setProgressStep(s.progress?.step ?? null);
          setErrorMessage(s.error ?? null);
          if (s.state === "SUCCESS") {
            stopPolling();
            setFinishedAt((prev) => prev ?? Date.now());
            setReport(s.result?.report ?? null);
            const rr = await fetch(`${API}/report/${data.seriesuid}`);
            if (!rr.ok) {
              throw new Error(`Report request failed (${rr.status})`);
            }
            setReport(await rr.json());
            const catalog = await loadSliceCatalog(data.seriesuid);
            const initialIndices = catalog.axial?.indices ?? [];
            setSliceIdx(initialIndices[0] ?? 0);
            handleUploadSuccess();
          } else if (s.state === "FAILURE") {
            stopPolling();
            setFinishedAt((prev) => prev ?? Date.now());
          }
        } catch (error) {
          stopPolling();
          setStatus("FAILURE");
          setProgressStep(null);
          setErrorMessage(error instanceof Error ? error.message : "Polling failed");
          setFinishedAt((prev) => prev ?? Date.now());
        }
      }, 2000);
    } catch (error) {
      stopPolling();
      setStatus("FAILURE");
      setProgressStep(null);
      setErrorMessage(error instanceof Error ? error.message : "Upload failed");
      const now = Date.now();
      setStartedAt((prev) => prev ?? now);
      setFinishedAt(now);
      setClockNow(now);
      resetScanState();
    }
  }, [handleUploadSuccess, loadSliceCatalog, resetScanState, stopPolling]);

  const openScan = useCallback(async (scan) => {
    stopPolling();
    setJobId(null);
    setSeriesuid(scan.seriesuid);
    setStatus("SUCCESS");
    setProgressStep(null);
    setErrorMessage(null);
    setStartedAt(null);
    setFinishedAt(null);
    setReport(scan.report ?? null);
    setSelectedNodule(null);
    setSliceIdx(0);
    setShowOverlay(true);
    setOverlayOpacity(0.30);
    const catalog = await loadSliceCatalog(scan.seriesuid);
    const initialIndices = catalog.axial?.indices ?? [];
    setSliceIdx(initialIndices[0] ?? 0);
  }, [loadSliceCatalog, stopPolling]);

  const elapsedSeconds = startedAt
    ? Math.max(0, ((finishedAt ?? clockNow) - startedAt) / 1000)
    : null;

  const activeSliceMeta = sliceCatalog[activeView] ?? { indices: [], count: 0 };
  const maxSliceIdx = activeSliceMeta.indices.length > 0
    ? activeSliceMeta.indices[activeSliceMeta.indices.length - 1]
    : null;

  // Jump to nodule slice only when the selected nodule or view changes — not on every scroll.
  useEffect(() => {
    if (selectedNodule?.slice_indices?.[activeView] != null) {
      setSliceIdx(resolveCandidateSlice(selectedNodule, activeView));
    }
  }, [activeView, selectedNodule]); // eslint-disable-line react-hooks/exhaustive-deps

  // When switching views with no nodule selected, clamp to a valid slice.
  useEffect(() => {
    if (selectedNodule) return;
    const indices = activeSliceMeta.indices ?? [];
    if (indices.length > 0 && !indices.includes(sliceIdx)) {
      setSliceIdx(indices[0]);
    }
  }, [activeView, activeSliceMeta.indices]); // eslint-disable-line react-hooks/exhaustive-deps

  const baseSliceUrl = seriesuid && status === "SUCCESS"
    ? `${API}/slices/${seriesuid}/${activeView}?idx=${sliceIdx}&layer=base`
    : null;
  const overlaySliceUrl = seriesuid && status === "SUCCESS"
    ? `${API}/slices/${seriesuid}/${activeView}?idx=${sliceIdx}&layer=overlay`
    : null;

  const candidates = report?.top_candidates ?? [];

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", background: "var(--bg)" }}>
      {/* Header */}
      <header style={{
        height: 48,
        display: "flex",
        alignItems: "center",
        padding: "0 20px",
        borderBottom: "1px solid var(--border)",
        background: "var(--bg-1)",
        gap: 10,
        flexShrink: 0,
      }}>
        <div style={{ color: "var(--teal)", display: "flex", alignItems: "center" }}>
          <LungIcon size={17} />
        </div>
        <span style={{ fontSize: 13, fontWeight: 600, letterSpacing: "0.01em", color: "var(--text)" }}>
          Pulmodex | Lung Nodule Detection
        </span>
        <div style={{ flex: 1 }} />
        <span style={{
          fontSize: 9,
          fontFamily: "var(--mono)",
          letterSpacing: "0.1em",
          textTransform: "uppercase",
          color: "var(--text-3)",
          border: "1px solid var(--border)",
          padding: "2px 7px",
          borderRadius: 3,
        }}>
          AI Diagnostics
        </span>
        <span style={{ fontSize: 10, color: "var(--text-3)", fontFamily: "var(--mono)" }}>v1.0</span>
      </header>

      <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>
        {/* Sidebar */}
        <aside style={{
          width: 288,
          background: "var(--bg-1)",
          borderRight: "1px solid var(--border)",
          display: "flex",
          flexDirection: "column",
          overflowY: "auto",
          flexShrink: 0,
        }}>
          <div style={{ padding: "16px 16px 12px" }}>
            <SectionLabel>Upload</SectionLabel>
            <UploadZone
              onUpload={handleUpload}
              disabled={status === "PENDING" || status === "PROGRESS"}
            />
          </div>

          {status && (
            <div style={{ padding: "0 16px 4px" }}>
              <StatusBanner
                status={status}
                step={progressStep}
                error={errorMessage}
                elapsedSeconds={elapsedSeconds}
              />
            </div>
          )}

          {seriesuid && (
            <div style={{ padding: "0 16px 16px" }}>
              <SectionLabel>Scan</SectionLabel>
              <div style={{
                background: "var(--bg-2)",
                border: "1px solid var(--border)",
                borderRadius: "var(--radius)",
                padding: "10px 12px",
              }}>
                <div style={{
                  fontSize: 9,
                  color: "var(--text-3)",
                  textTransform: "uppercase",
                  letterSpacing: "0.08em",
                  marginBottom: 5,
                  fontWeight: 600,
                }}>
                  Series UID
                </div>
                <div style={{
                  fontFamily: "var(--mono)",
                  fontSize: 9,
                  color: "var(--text-2)",
                  wordBreak: "break-all",
                  lineHeight: 1.7,
                }}>
                  {seriesuid}
                </div>
              </div>
            </div>
          )}

          {status === "SUCCESS" && (
            <>
              <div style={{ padding: "0 16px" }}>
                <OverlayControls
                  showOverlay={showOverlay}
                  onToggleOverlay={() => setShowOverlay((prev) => !prev)}
                  overlayOpacity={overlayOpacity}
                  onOpacityChange={setOverlayOpacity}
                />
              </div>
              <div style={{ padding: "0 16px 20px" }}>
                <SectionLabel>Findings</SectionLabel>
                <NoduleList
                  candidates={candidates}
                  selected={selectedNodule}
                  onSelect={(c) => {
                    setSelectedNodule(c);
                    setSliceIdx(resolveCandidateSlice(c, activeView));
                  }}
                />
                {candidates.length === 0 && (
                  <div style={{ fontSize: 11, color: "var(--text-3)", padding: "6px 0" }}>
                    No nodules detected.
                  </div>
                )}
              </div>
            </>
          )}

          {history.filter(s => s.status === "done").length > 0 && (
            <div style={{ padding: "0 16px 20px", borderTop: "1px solid var(--border)", paddingTop: 16 }}>
              <SectionLabel>History</SectionLabel>
              <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                {history.filter(s => s.status === "done").map((scan) => {
                  const isActive = scan.seriesuid === seriesuid;
                  const date = new Date(scan.uploaded_at).toLocaleDateString(undefined, { month: "short", day: "numeric" });
                  const time = new Date(scan.uploaded_at).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
                  const n = scan.report?.n_candidates_final ?? 0;
                  return (
                    <button
                      key={scan.seriesuid}
                      onClick={() => openScan(scan)}
                      style={{
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "flex-start",
                        gap: 2,
                        padding: "7px 10px",
                        background: isActive ? "var(--bg-3)" : "var(--bg-2)",
                        border: `1px solid ${isActive ? "var(--teal)" : "var(--border)"}`,
                        borderRadius: "var(--radius)",
                        cursor: "pointer",
                        textAlign: "left",
                        width: "100%",
                      }}
                    >
                      <div style={{ fontSize: 10, color: "var(--text)", fontFamily: "var(--mono)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", width: "100%" }}>
                        {scan.filename !== "unknown.zip" ? scan.filename : scan.seriesuid.slice(0, 8) + "…"}
                      </div>
                      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                        <span style={{ fontSize: 9, color: "var(--text-3)", fontFamily: "var(--mono)" }}>{date} {time}</span>
                        <span style={{ fontSize: 9, color: n > 0 ? "var(--teal)" : "var(--text-3)", fontFamily: "var(--mono)" }}>
                          {n > 0 ? `${n} nodule${n > 1 ? "s" : ""}` : "none"}
                        </span>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </aside>

        {/* Main viewer */}
        <main style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
          <ViewTabs active={activeView} onChange={setActiveView} />
          <Viewer
            baseSliceUrl={baseSliceUrl}
            overlaySliceUrl={overlaySliceUrl}
            sliceIdx={sliceIdx}
            onSliceChange={(updater) => {
              setSelectedNodule(null);
              setSliceIdx(updater);
            }}
            view={activeView}
            maxSliceIdx={maxSliceIdx}
            showOverlay={showOverlay}
            overlayOpacity={overlayOpacity}
          />
        </main>
      </div>
    </div>
  );
}
