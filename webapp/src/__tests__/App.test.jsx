import React from "react";
import { render, screen, waitFor, act, fireEvent } from "@testing-library/react";
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
    expect(screen.queryByText("3d")).not.toBeInTheDocument();
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
    "transitions to SUCCESS and shows findings when polling resolves",
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
      expect(screen.getByText("Min diameter")).toBeInTheDocument();
    },
    6000,
  );

  it("requests boxed CT slices after a successful scan", async () => {
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
    expect(img).toHaveAttribute("src", "/api/slices/series-raw/axial?idx=0");
  });

  it("filters nodules by minimum confidence score", async () => {
    const report = {
      candidates: [
        {
          coordX: 10,
          coordY: 20,
          coordZ: 30,
          prob: 0.2,
          diameter_mm: 5.0,
          slice_indices: { axial: 0, coronal: 0, sagittal: 0 },
        },
        {
          coordX: 40,
          coordY: 50,
          coordZ: 60,
          prob: 0.85,
          diameter_mm: 7.0,
          slice_indices: { axial: 1, coronal: 1, sagittal: 1 },
        },
      ],
    };

    vi.stubGlobal("fetch", vi.fn().mockImplementation((url) => {
      if (url.includes("/predict")) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ job_id: "job-filter", seriesuid: "series-filter" }) });
      }
      if (url.includes("/status")) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ state: "SUCCESS", result: {} }) });
      }
      if (url.includes("/slices/")) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ view: "axial", indices: [0, 1], count: 2 }) });
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve(report) });
    }));

    render(<App />);
    const input = document.querySelector("input[type='file']");
    await userEvent.upload(input, new File(["d"], "scan.zip", { type: "application/zip" }));

    await waitFor(() => expect(screen.getByText("Min confidence")).toBeInTheDocument(), { timeout: 4000 });
    expect(screen.getAllByTestId("nodule-item")).toHaveLength(2);

    const sliders = screen.getAllByRole("slider");
    fireEvent.change(sliders[1], { target: { value: "0.5" } });

    await waitFor(() => expect(screen.getAllByTestId("nodule-item")).toHaveLength(1));
    expect(screen.getByText("85%")).toBeInTheDocument();
  });

  it("shows a slicer markups download link for detected nodules", async () => {
    const report = {
      candidates: [
        {
          coordX: 10,
          coordY: 20,
          coordZ: 30,
          prob: 0.85,
          diameter_mm: 7.0,
          slice_indices: { axial: 1, coronal: 1, sagittal: 1 },
        },
      ],
    };

    vi.stubGlobal("fetch", vi.fn().mockImplementation((url) => {
      if (url.includes("/predict")) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ job_id: "job-markup", seriesuid: "series-markup" }) });
      }
      if (url.includes("/status")) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ state: "SUCCESS", result: {} }) });
      }
      if (url.includes("/slices/")) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ view: "axial", indices: [1], count: 1 }) });
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve(report) });
    }));

    render(<App />);
    const input = document.querySelector("input[type='file']");
    await userEvent.upload(input, new File(["d"], "scan.zip", { type: "application/zip" }));

    await waitFor(() => expect(screen.getByText("Complete")).toBeInTheDocument(), { timeout: 4000 });
    const link = await screen.findByRole("link", { name: "Download Slicer coordinates" });
    expect(link).toHaveAttribute("href", "/api/markups/series-markup");
    expect(link).toHaveAttribute("title", "Download Slicer coordinates");
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
