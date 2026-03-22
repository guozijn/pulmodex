import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, afterEach } from "vitest";
import App from "../App";

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("App", () => {
  it("renders the application header", () => {
    render(<App />);
    expect(screen.getByText(/Pulmodex — Lung Nodule Detection/)).toBeInTheDocument();
  });

  it("renders all three view tab buttons", () => {
    render(<App />);
    expect(screen.getByText("axial")).toBeInTheDocument();
    expect(screen.getByText("coronal")).toBeInTheDocument();
    expect(screen.getByText("sagittal")).toBeInTheDocument();
  });

  it("shows upload prompt in the initial state", () => {
    render(<App />);
    expect(screen.getByText(/Drop .mhd file here/)).toBeInTheDocument();
  });

  it("shows PENDING status and disables the upload zone after submitting a file", async () => {
    // Prevent the real 2 s interval from running during this test.
    vi.spyOn(window, "setInterval").mockReturnValue(99);
    vi.spyOn(window, "clearInterval").mockImplementation(() => {});
    vi.stubGlobal("fetch", vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => ({ job_id: "job-1", seriesuid: "series-1" }),
    }));

    render(<App />);
    const input = document.querySelector("input[type='file']");
    await userEvent.upload(input, new File(["data"], "scan.mhd", { type: "" }));

    await waitFor(() => expect(screen.getByText("Queued…")).toBeInTheDocument());
    expect(screen.getByText("Processing…")).toBeInTheDocument();
  });

  it(
    "transitions to SUCCESS and shows saliency control when polling resolves",
    async () => {
      // NOTE: vi.useFakeTimers() deadlocks in React 18 + jsdom because
      // advanceTimersByTimeAsync cannot resolve chained async Promises while
      // React's scheduler is also waiting on the same fake clock. Manually
      // invoking the captured setInterval callback bypasses React's internal
      // scheduler, so setState updates are never committed. The real-timer
      // approach is the only reliable method here: the first poll fires after
      // ~2 s and waitFor catches the resulting state update normally.
      vi.stubGlobal("fetch", vi.fn().mockImplementation((url) => {
        if (url.includes("/predict"))
          return Promise.resolve({ ok: true, json: () => Promise.resolve({ job_id: "job-2", seriesuid: "series-2" }) });
        if (url.includes("/status"))
          return Promise.resolve({ ok: true, json: () => Promise.resolve({ state: "SUCCESS", result: {} }) });
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ top_candidates: [] }) });
      }));

      render(<App />);
      const input = document.querySelector("input[type='file']");
      await userEvent.upload(input, new File(["d"], "scan.mhd", { type: "" }));

      await waitFor(
        () => expect(screen.getByText("Complete")).toBeInTheDocument(),
        { timeout: 4000 },
      );
      expect(screen.getByText(/Saliency opacity/)).toBeInTheDocument();
    },
    6000,
  );
});
