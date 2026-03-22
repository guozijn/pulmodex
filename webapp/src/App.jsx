import React, { useState, useCallback, useRef } from "react";
import Viewer from "./components/Viewer";
import NoduleList from "./components/NoduleList";
import UploadZone from "./components/UploadZone";
import StatusBanner from "./components/StatusBanner";

const API = "/api";

export default function App() {
  const [jobId, setJobId] = useState(null);
  const [seriesuid, setSeriesuid] = useState(null);
  const [status, setStatus] = useState(null);   // "PENDING"|"PROGRESS"|"SUCCESS"|"FAILURE"
  const [report, setReport] = useState(null);
  const [activeView, setActiveView] = useState("axial");
  const [sliceIdx, setSliceIdx] = useState(0);
  const [saliencyAlpha, setSaliencyAlpha] = useState(0.4);
  const [selectedNodule, setSelectedNodule] = useState(null);
  const pollRef = useRef(null);

  const handleUpload = useCallback(async (file) => {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(`${API}/predict`, { method: "POST", body: form });
    const data = await res.json();
    setJobId(data.job_id);
    setSeriesuid(data.seriesuid);
    setStatus("PENDING");
    setReport(null);

    // Poll every 2s
    pollRef.current = setInterval(async () => {
      const r = await fetch(`${API}/status/${data.job_id}`);
      const s = await r.json();
      setStatus(s.state);
      if (s.state === "SUCCESS") {
        clearInterval(pollRef.current);
        setReport(s.result?.report ?? null);
        // Load report for nodule list
        const rr = await fetch(`${API}/report/${data.seriesuid}`);
        if (rr.ok) setReport(await rr.json());
      } else if (s.state === "FAILURE") {
        clearInterval(pollRef.current);
      }
    }, 2000);
  }, []);

  const sliceUrl = seriesuid && status === "SUCCESS"
    ? `${API}/slices/${seriesuid}/${activeView}?idx=${sliceIdx}`
    : null;

  const candidates = report?.top_candidates ?? [];

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh" }}>
      <header style={{ padding: "12px 24px", background: "#1a1f2e", borderBottom: "1px solid #2d3748" }}>
        <h1 style={{ fontSize: 20, fontWeight: 700, letterSpacing: 1 }}>
          Pulmodex — Lung Nodule Detection
        </h1>
      </header>

      <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>
        {/* Left panel */}
        <div style={{ width: 280, background: "#131720", borderRight: "1px solid #2d3748", padding: 16, overflowY: "auto" }}>
          <UploadZone onUpload={handleUpload} disabled={status === "PENDING" || status === "PROGRESS"} />
          <StatusBanner status={status} />

          {status === "SUCCESS" && (
            <>
              <SaliencyControl alpha={saliencyAlpha} onChange={setSaliencyAlpha} />
              <NoduleList
                candidates={candidates}
                selected={selectedNodule}
                onSelect={(c, idx) => {
                  setSelectedNodule(c);
                  setSliceIdx(Math.round(c.coordZ));
                }}
              />
            </>
          )}
        </div>

        {/* Main viewer */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
          <ViewTabs active={activeView} onChange={setActiveView} />
          <Viewer
            sliceUrl={sliceUrl}
            sliceIdx={sliceIdx}
            onSliceChange={setSliceIdx}
            saliencyAlpha={saliencyAlpha}
          />
        </div>
      </div>
    </div>
  );
}

function ViewTabs({ active, onChange }) {
  return (
    <div style={{ display: "flex", gap: 0, background: "#1a1f2e", borderBottom: "1px solid #2d3748" }}>
      {["axial", "coronal", "sagittal"].map((v) => (
        <button
          key={v}
          onClick={() => onChange(v)}
          style={{
            padding: "8px 20px",
            background: active === v ? "#3182ce" : "transparent",
            color: "#e2e8f0",
            border: "none",
            cursor: "pointer",
            fontWeight: active === v ? 700 : 400,
            textTransform: "capitalize",
          }}
        >
          {v}
        </button>
      ))}
    </div>
  );
}

function SaliencyControl({ alpha, onChange }) {
  return (
    <div style={{ marginBottom: 16 }}>
      <label style={{ fontSize: 12, color: "#a0aec0" }}>Saliency opacity: {Math.round(alpha * 100)}%</label>
      <input
        type="range"
        min={0} max={1} step={0.05}
        value={alpha}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ width: "100%", marginTop: 4 }}
      />
    </div>
  );
}
