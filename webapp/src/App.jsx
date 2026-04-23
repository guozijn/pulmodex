import React, { useState, useCallback, useEffect, useRef } from "react";
import Viewer from "./components/Viewer";
import NoduleList from "./components/NoduleList";
import UploadZone from "./components/UploadZone";
import StatusBanner from "./components/StatusBanner";

const API = "/api";
const VIEW_NAMES = ["axial", "coronal", "sagittal"];

function candidateProb(candidate) {
  return typeof candidate?.fp_prob === "number" ? candidate.fp_prob : (candidate?.prob ?? 0);
}

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

function SidebarMenu({ active, items, onChange }) {
  return (
    <div style={{
      display: "grid",
      gridTemplateColumns: "1fr 1fr",
      gap: 6,
      padding: "0 16px 12px",
    }}>
      {items.map((item) => (
        <button
          key={item.id}
          type="button"
          onClick={() => onChange(item.id)}
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 8,
            padding: "8px 9px",
            background: active === item.id ? "var(--bg-3)" : "var(--bg-2)",
            color: active === item.id ? "var(--text)" : "var(--text-2)",
            border: `1px solid ${active === item.id ? "var(--teal)" : "var(--border)"}`,
            borderRadius: "var(--radius)",
            cursor: "pointer",
            textAlign: "left",
          }}
        >
          <span style={{
            fontSize: 10,
            fontWeight: 600,
            letterSpacing: "0.08em",
            textTransform: "uppercase",
          }}>
            {item.label}
          </span>
          {item.badge ? (
            <span style={{
              fontSize: 9,
              fontFamily: "var(--mono)",
              color: active === item.id ? "var(--teal)" : "var(--text-3)",
            }}>
              {item.badge}
            </span>
          ) : null}
        </button>
      ))}
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
  const [selectedNodule, setSelectedNodule] = useState(null);
  const [minDiameterMm, setMinDiameterMm] = useState(0);
  const [minConfidence, setMinConfidence] = useState(0);
  const [sliceCatalog, setSliceCatalog] = useState({});
  const [history, setHistory] = useState([]);
  const [deletingScanId, setDeletingScanId] = useState(null);
  const [historyError, setHistoryError] = useState(null);
  const [activeSidebarPanel, setActiveSidebarPanel] = useState("upload");
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
    setMinDiameterMm(0);
    setMinConfidence(0);
    setActiveSidebarPanel("upload");
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
      if (res.ok) {
        const data = await res.json();
        setHistory(Array.isArray(data) ? data : []);
        setHistoryError(null);
      }
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
      setStatus("UPLOADING");
      setProgressStep(null);
      setErrorMessage(null);
      setFinishedAt(null);
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
      setActiveSidebarPanel("upload");
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
            setActiveSidebarPanel("findings");
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
    setHistoryError(null);
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
    setActiveSidebarPanel(scan.report?.candidates?.length ? "findings" : "scan");
    const catalog = await loadSliceCatalog(scan.seriesuid);
    const initialIndices = catalog.axial?.indices ?? [];
    setSliceIdx(initialIndices[0] ?? 0);
  }, [loadSliceCatalog, stopPolling]);

  const handleDeleteScan = useCallback(async (scan) => {
    setDeletingScanId(scan.seriesuid);
    setHistoryError(null);
    try {
      const res = await fetch(`${API}/scans/${scan.seriesuid}`, { method: "DELETE" });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.detail || "Delete failed");
      }

      setHistory((prev) => prev.filter((entry) => entry.seriesuid !== scan.seriesuid));

      if (seriesuid === scan.seriesuid) {
        stopPolling();
        setJobId(null);
        setStatus(null);
        setProgressStep(null);
        setErrorMessage(null);
        setStartedAt(null);
        setFinishedAt(null);
        resetScanState();
      }
    } catch (error) {
      setHistoryError(error instanceof Error ? error.message : "Delete failed");
    } finally {
      setDeletingScanId(null);
    }
  }, [resetScanState, seriesuid, stopPolling]);

  const elapsedSeconds = startedAt
    ? Math.max(0, ((finishedAt ?? clockNow) - startedAt) / 1000)
    : null;

  const activeSliceMeta = sliceCatalog[activeView] ?? { indices: [], count: 0 };
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
    ? `${API}/slices/${seriesuid}/${activeView}?idx=${sliceIdx}`
    : null;

  const candidates = report?.candidates ?? [];
  const maxDiameterMm = candidates.length > 0
    ? Math.ceil(Math.max(...candidates.map((c) => c.diameter_mm ?? 0)))
    : 30;
  const filteredCandidates = candidates.filter((c) => (
    (c.diameter_mm ?? 0) >= minDiameterMm && candidateProb(c) >= minConfidence
  ));

  const historyDone = history.filter((s) => s.status === "done");
  const sidebarItems = [
    { id: "upload", label: "Upload" },
    { id: "scan", label: "Scan", badge: seriesuid ? "1" : "" },
    { id: "findings", label: "Findings", badge: filteredCandidates.length ? String(filteredCandidates.length) : "" },
    { id: "history", label: "History", badge: historyDone.length ? String(historyDone.length) : "" },
  ];

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
          flexShrink: 0,
        }}>
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
          <div style={{ paddingTop: 16 }}>
            <SidebarMenu active={activeSidebarPanel} items={sidebarItems} onChange={setActiveSidebarPanel} />
          </div>

          <div style={{ flex: 1, overflowY: "auto", padding: "0 16px 20px" }}>
            {activeSidebarPanel === "upload" && (
              <div>
                <SectionLabel>Upload</SectionLabel>
                <UploadZone
                  onUpload={handleUpload}
                  disabled={status === "UPLOADING" || status === "PENDING" || status === "PROGRESS"}
                  uploading={status === "UPLOADING"}
                />
              </div>
            )}

            {activeSidebarPanel === "scan" && (
              <div>
                <SectionLabel>Scan</SectionLabel>
                {seriesuid ? (
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
                ) : (
                  <div style={{ fontSize: 11, color: "var(--text-3)" }}>
                    No scan selected.
                  </div>
                )}
              </div>
            )}

            {activeSidebarPanel === "findings" && (
              <div>
                <SectionLabel>Findings</SectionLabel>
                {status === "SUCCESS" ? (
                  <>
                    <div style={{ marginBottom: 16 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 7 }}>
                        <label style={{ fontSize: 11, color: "var(--text-2)" }}>Min diameter</label>
                        <span style={{ fontSize: 11, color: "var(--teal)", fontFamily: "var(--mono)" }}>
                          {minDiameterMm > 0 ? `>= ${minDiameterMm.toFixed(1)} mm` : "All"}
                        </span>
                      </div>
                      <input
                        type="range"
                        min={0}
                        max={maxDiameterMm}
                        step={0.5}
                        value={minDiameterMm}
                        onChange={(e) => setMinDiameterMm(Number(e.target.value))}
                        style={{ width: "100%", cursor: "pointer" }}
                      />
                    </div>
                    <div style={{ marginBottom: 16 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 7 }}>
                        <label style={{ fontSize: 11, color: "var(--text-2)" }}>Min confidence</label>
                        <span style={{ fontSize: 11, color: "var(--teal)", fontFamily: "var(--mono)" }}>
                          {minConfidence > 0 ? `>= ${(minConfidence * 100).toFixed(0)}%` : "All"}
                        </span>
                      </div>
                      <input
                        type="range"
                        min={0}
                        max={1}
                        step={0.05}
                        value={minConfidence}
                        onChange={(e) => setMinConfidence(Number(e.target.value))}
                        style={{ width: "100%", cursor: "pointer" }}
                      />
                    </div>
                    <NoduleList
                      candidates={filteredCandidates}
                      selected={selectedNodule}
                      onSelect={(c) => {
                        setSelectedNodule(c);
                        setSliceIdx(resolveCandidateSlice(c, activeView));
                      }}
                    />
                    {filteredCandidates.length === 0 && (
                      <div style={{ fontSize: 11, color: "var(--text-3)", padding: "6px 0" }}>
                        {candidates.length === 0
                          ? "No nodules detected."
                          : "No nodules match the current filter."}
                      </div>
                    )}
                  </>
                ) : (
                  <div style={{ fontSize: 11, color: "var(--text-3)" }}>
                    Findings will appear after a completed scan.
                  </div>
                )}
              </div>
            )}

            {activeSidebarPanel === "history" && (
              <div>
                <SectionLabel>History</SectionLabel>
                {historyDone.length > 0 ? (
                  <>
                    {historyError && (
                      <div style={{ fontSize: 11, color: "#ff8a80", marginBottom: 8 }}>
                        {historyError}
                      </div>
                    )}
                    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                      {historyDone.map((scan) => {
                        const isActive = scan.seriesuid === seriesuid;
                        const date = new Date(scan.uploaded_at).toLocaleDateString(undefined, { month: "short", day: "numeric" });
                        const time = new Date(scan.uploaded_at).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
                        const n = scan.report?.n_candidates_final ?? 0;
                        return (
                          <div
                            key={scan.seriesuid}
                            style={{
                              display: "flex",
                              alignItems: "stretch",
                              gap: 6,
                              background: isActive ? "var(--bg-3)" : "var(--bg-2)",
                              border: `1px solid ${isActive ? "var(--teal)" : "var(--border)"}`,
                              borderRadius: "var(--radius)",
                              width: "100%",
                            }}
                          >
                            <button
                              type="button"
                              onClick={() => openScan(scan)}
                              style={{
                                display: "flex",
                                flexDirection: "column",
                                alignItems: "flex-start",
                                gap: 2,
                                padding: "7px 10px 5px",
                                background: "transparent",
                                border: 0,
                                cursor: "pointer",
                                textAlign: "left",
                                flex: "1 1 auto",
                                minWidth: 0,
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
                            <div style={{ display: "flex", justifyContent: "flex-end", alignItems: "center", flexShrink: 0, padding: "0 10px 0 0" }}>
                              <button
                                type="button"
                                onClick={() => handleDeleteScan(scan)}
                                disabled={deletingScanId === scan.seriesuid}
                                aria-label={`Delete ${scan.filename !== "unknown.zip" ? scan.filename : scan.seriesuid}`}
                                style={{
                                  fontSize: 9,
                                  fontFamily: "var(--mono)",
                                  color: deletingScanId === scan.seriesuid ? "var(--text-3)" : "#ff8a80",
                                  background: "transparent",
                                  border: "1px solid var(--border)",
                                  borderRadius: 4,
                                  padding: "2px 6px",
                                  cursor: deletingScanId === scan.seriesuid ? "wait" : "pointer",
                                }}
                              >
                                {deletingScanId === scan.seriesuid ? "DELETING" : "DELETE"}
                              </button>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </>
                ) : (
                  <div style={{ fontSize: 11, color: "var(--text-3)" }}>
                    No saved scans yet.
                  </div>
                )}
              </div>
            )}
          </div>
        </aside>

        {/* Main viewer */}
        <main style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
          <ViewTabs active={activeView} onChange={setActiveView} />
          <Viewer
            baseSliceUrl={baseSliceUrl}
            sliceIdx={sliceIdx}
            view={activeView}
          />
        </main>
      </div>
    </div>
  );
}
