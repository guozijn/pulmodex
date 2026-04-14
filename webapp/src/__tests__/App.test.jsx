import React from "react";
import { render, screen, waitFor, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, afterEach } from "vitest";
import App from "../App";

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

function getHistoryOpenButton(name) {
  return screen.getAllByRole("button").find((button) => button.textContent?.includes(name) && !button.getAttribute("aria-label"));
}

function getLabeledToggle(label, name) {
  return screen.getAllByRole("button", { name }).find((button) => button.parentElement?.textContent?.includes(label));
}

describe("App", () => {
  it("renders the application header", () => {
    render(<App />);
    expect(screen.getByText(/Pulmodex | Lung Nodule Detection/)).toBeInTheDocument();
  });

  it("renders all three view tab buttons", () => {
    render(<App />);
    expect(screen.getByText("axial")).toBeInTheDocument();
    expect(screen.getByText("coronal")).toBeInTheDocument();
    expect(screen.getByText("sagittal")).toBeInTheDocument();
  });

  it("shows upload prompt in the initial state", () => {
    render(<App />);
    expect(screen.getByText(/Drop a .zip DICOM series or .nii.gz volume here/)).toBeInTheDocument();
  });

  it("shows PENDING status and disables the upload zone after submitting a file", async () => {
    // Prevent the real 2 s interval from running during this test.
    vi.spyOn(window, "setInterval").mockReturnValue(99);
    vi.spyOn(window, "clearInterval").mockImplementation(() => {});
    vi.stubGlobal("fetch", vi.fn().mockImplementation((url) => {
      if (url.includes("/predict"))
        return Promise.resolve({ ok: true, json: async () => ({ job_id: "job-1", seriesuid: "series-1" }) });
      return Promise.resolve({ ok: true, json: async () => [] });
    }));

    render(<App />);
    const input = document.querySelector("input[type='file']");
    await userEvent.upload(input, new File(["data"], "scan.zip", { type: "application/zip" }));

    await waitFor(() => expect(screen.getByText("Queued…")).toBeInTheDocument());
    expect(screen.getByText("Processing…")).toBeInTheDocument();
    expect(screen.getByText("ELAPSED 00:00")).toBeInTheDocument();
  });

  it("shows uploading feedback immediately before the queue response arrives", async () => {
    let resolvePredict;
    vi.spyOn(window, "setInterval").mockReturnValue(99);
    vi.spyOn(window, "clearInterval").mockImplementation(() => {});
    vi.stubGlobal("fetch", vi.fn().mockImplementation((url) => {
      if (url.includes("/predict")) {
        return new Promise((resolve) => {
          resolvePredict = () => resolve({ ok: true, json: async () => ({ job_id: "job-upload", seriesuid: "series-upload" }) });
        });
      }
      return Promise.resolve({ ok: true, json: async () => [] });
    }));

    render(<App />);
    const input = document.querySelector("input[type='file']");
    const uploadPromise = userEvent.upload(input, new File(["data"], "scan.zip", { type: "application/zip" }));

    await waitFor(() => expect(screen.getAllByText("Uploading scan…").length).toBeGreaterThan(0));
    resolvePredict();
    await uploadPromise;
    await waitFor(() => expect(screen.getByText("Queued…")).toBeInTheDocument());
  });

  it(
    "transitions to SUCCESS and shows overlay controls when polling resolves",
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
        if (url.includes("/slices/")) {
          return Promise.resolve({ ok: true, json: () => Promise.resolve({ view: "axial", indices: [0, 1], count: 2 }) });
        }
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ candidates: [] }) });
      }));

      render(<App />);
      const input = document.querySelector("input[type='file']");
      await userEvent.upload(input, new File(["d"], "scan.zip", { type: "application/zip" }));

      await waitFor(
        () => expect(screen.getByText("Complete")).toBeInTheDocument(),
        { timeout: 4000 },
      );
      expect(screen.getByText("Heatmap overlay")).toBeInTheDocument();
      const toggle = getLabeledToggle("Heatmap overlay", "OFF");
      expect(toggle).toBeInTheDocument();
      await userEvent.click(toggle);
      expect(getLabeledToggle("Heatmap overlay", "ON")).toBeInTheDocument();
    },
    6000,
  );

  it("switches to raw CT when annotations are hidden", async () => {
    vi.stubGlobal("fetch", vi.fn().mockImplementation((url) => {
      if (url.includes("/predict")) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ job_id: "job-raw", seriesuid: "series-raw" }) });
      }
      if (url.includes("/status")) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ state: "SUCCESS", result: {} }) });
      }
      if (url.includes("/slices/")) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ view: "axial", indices: [0, 1], count: 2 }) });
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ candidates: [] }) });
    }));

    render(<App />);
    const input = document.querySelector("input[type='file']");
    await userEvent.upload(input, new File(["d"], "scan.zip", { type: "application/zip" }));

    await waitFor(() => expect(screen.getByText("Complete")).toBeInTheDocument(), { timeout: 4000 });
    const img = await screen.findByRole("img", { name: "slice 0" });
    expect(img).toHaveAttribute("src", "/api/slices/series-raw/axial?idx=0&layer=base");

    const annotationsToggle = getLabeledToggle("Annotations", "ON");
    await userEvent.click(annotationsToggle);

    await waitFor(() => {
      expect(screen.getByRole("img", { name: "slice 0" })).toHaveAttribute("src", "/api/slices/series-raw/axial?idx=0&layer=raw");
    });
  });

  it("shows backend failure details when polling returns FAILURE", async () => {
    vi.stubGlobal("fetch", vi.fn().mockImplementation((url) => {
      if (url.includes("/predict")) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ job_id: "job-fail", seriesuid: "series-fail" }),
        });
      }
      if (url.includes("/status")) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({
            state: "FAILURE",
            error: "Please provide ground truth targets during training.",
          }),
        });
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve({}) });
    }));

    render(<App />);
    const input = document.querySelector("input[type='file']");
    await userEvent.upload(input, new File(["d"], "scan.zip", { type: "application/zip" }));

    await waitFor(() => expect(screen.getByText("Failed")).toBeInTheDocument(), { timeout: 4000 });
    expect(screen.getByText("Please provide ground truth targets during training.")).toBeInTheDocument();
  });

  it("clears previous scan metadata when a new upload request fails", async () => {
    vi.spyOn(window, "setInterval").mockReturnValue(99);
    vi.spyOn(window, "clearInterval").mockImplementation(() => {});

    const historyScan = {
      seriesuid: "existing-series",
      filename: "existing.zip",
      uploaded_at: "2026-04-01T12:00:00Z",
      status: "done",
      report: { n_candidates_final: 0, candidates: [] },
    };

    vi.stubGlobal("fetch", vi.fn().mockImplementation((url) => {
      if (url.includes("/scans")) {
        return Promise.resolve({ ok: true, json: async () => [historyScan] });
      }
      if (url.includes("/slices/existing-series/")) {
        return Promise.resolve({ ok: true, json: async () => ({ view: "axial", indices: [0, 1], count: 2 }) });
      }
      if (url.includes("/predict")) {
        return Promise.resolve({ ok: false, json: async () => ({ detail: "Only .zip or .nii.gz uploads are supported" }) });
      }
      return Promise.resolve({ ok: true, json: async () => ({}) });
    }));

    render(<App />);

    await waitFor(() => expect(screen.getByRole("button", { name: /history/i })).toBeInTheDocument());
    await userEvent.click(screen.getByRole("button", { name: /history/i }));
    await userEvent.click(getHistoryOpenButton("existing.zip"));
    await waitFor(() => expect(screen.getByText("Series UID")).toBeInTheDocument());
    expect(screen.getByText("existing-series")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /upload/i }));
    const input = document.querySelector("input[type='file']");
    await userEvent.upload(input, new File(["bad"], "bad.zip", { type: "application/zip" }));

    await waitFor(() => expect(screen.getByText("Failed")).toBeInTheDocument());
    expect(screen.queryByText("Series UID")).not.toBeInTheDocument();
    expect(screen.queryByText("existing-series")).not.toBeInTheDocument();
  });

  it("deletes a history item and clears the viewer when deleting the active scan", async () => {
    const historyScan = {
      seriesuid: "existing-series",
      filename: "existing.zip",
      uploaded_at: "2026-04-01T12:00:00Z",
      status: "done",
      report: { n_candidates_final: 0, candidates: [] },
    };

    vi.stubGlobal("fetch", vi.fn().mockImplementation((url, options = {}) => {
      if (url.includes("/scans") && (!options.method || options.method === "GET")) {
        return Promise.resolve({ ok: true, json: async () => [historyScan] });
      }
      if (url.includes("/slices/existing-series/")) {
        return Promise.resolve({ ok: true, json: async () => ({ view: "axial", indices: [0, 1], count: 2 }) });
      }
      if (url.includes("/scans/existing-series") && options.method === "DELETE") {
        return Promise.resolve({ ok: true, json: async () => ({ status: "deleted", seriesuid: "existing-series" }) });
      }
      return Promise.resolve({ ok: true, json: async () => ({}) });
    }));

    render(<App />);

    await waitFor(() => expect(screen.getByRole("button", { name: /history/i })).toBeInTheDocument());
    await userEvent.click(screen.getByRole("button", { name: /history/i }));
    await userEvent.click(getHistoryOpenButton("existing.zip"));
    await waitFor(() => expect(screen.getByText("Series UID")).toBeInTheDocument());

    await userEvent.click(screen.getByRole("button", { name: /history/i }));
    await userEvent.click(screen.getByRole("button", { name: /delete existing\.zip/i }));

    await waitFor(() => expect(screen.queryByText("existing.zip")).not.toBeInTheDocument());
    expect(screen.queryByText("Series UID")).not.toBeInTheDocument();
    expect(screen.getByText(/Drop a \.zip DICOM series or \.nii\.gz volume here/)).toBeInTheDocument();
  });

  it("surfaces polling request failures instead of hanging in progress", async () => {
    const intervals = [];
    vi.spyOn(window, "setInterval").mockImplementation((callback, delay) => {
      intervals.push({ callback, delay });
      return intervals.length;
    });
    vi.spyOn(window, "clearInterval").mockImplementation(() => {});

    vi.stubGlobal("fetch", vi.fn().mockImplementation((url) => {
      if (url.includes("/predict")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ job_id: "job-poll-error", seriesuid: "series-poll-error" }),
        });
      }
      if (url.includes("/status")) {
        return Promise.resolve({ ok: false, status: 503, json: async () => ({}) });
      }
      return Promise.resolve({ ok: true, json: async () => [] });
    }));

    render(<App />);
    const input = document.querySelector("input[type='file']");
    await userEvent.upload(input, new File(["data"], "scan.zip", { type: "application/zip" }));

    const pollInterval = intervals.find(({ delay }) => delay === 2000);
    await act(async () => {
      await pollInterval.callback();
    });

    await waitFor(() => expect(screen.getByText("Failed")).toBeInTheDocument());
    expect(screen.getByText("Status request failed (503)")).toBeInTheDocument();
  });

  it("clears the polling timer when the component unmounts", async () => {
    const clearIntervalSpy = vi.spyOn(window, "clearInterval").mockImplementation(() => {});
    vi.spyOn(window, "setInterval").mockReturnValue(123);
    vi.stubGlobal("fetch", vi.fn().mockImplementation((url) => {
      if (url.includes("/predict")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ job_id: "job-unmount", seriesuid: "series-unmount" }),
        });
      }
      return Promise.resolve({ ok: true, json: async () => [] });
    }));

    const { unmount } = render(<App />);
    const input = document.querySelector("input[type='file']");
    await userEvent.upload(input, new File(["data"], "scan.zip", { type: "application/zip" }));

    await waitFor(() => expect(screen.getByText("Queued…")).toBeInTheDocument());
    unmount();

    expect(clearIntervalSpy).toHaveBeenCalledWith(123);
  });
});
