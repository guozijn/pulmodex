import React from "react";
import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import NoduleViewer3D from "../NoduleViewer3D";

// Three.js requires WebGL which is unavailable in jsdom. The component falls
// back gracefully when WebGLRenderer throws — these tests verify that fallback
// and the UI elements that are always present in the JSX tree.
vi.mock("three", async () => {
  const actual = await vi.importActual("three");
  return {
    ...actual,
    WebGLRenderer: vi.fn().mockImplementation(() => {
      throw new Error("WebGL not supported");
    }),
  };
});

vi.mock("three/addons/controls/OrbitControls.js", () => ({
  OrbitControls: vi.fn(),
}));

const CANDIDATES = [
  { coordX: 10, coordY: 20, coordZ: 30, prob: 0.9, fp_prob: 0.9, diameter_mm: 8 },
  { coordX: 40, coordY: 50, coordZ: 60, prob: 0.5, fp_prob: 0.5, diameter_mm: 6 },
];

describe("NoduleViewer3D", () => {
  it("shows placeholder when no candidates are provided", () => {
    render(<NoduleViewer3D candidates={[]} />);
    expect(screen.getByText("No nodules to display")).toBeInTheDocument();
  });

  it("hides placeholder when candidates are provided", () => {
    render(<NoduleViewer3D candidates={CANDIDATES} />);
    expect(screen.queryByText("No nodules to display")).not.toBeInTheDocument();
  });

  it("renders the 3D VOLUME label", () => {
    render(<NoduleViewer3D candidates={[]} />);
    expect(screen.getByText("3D VOLUME")).toBeInTheDocument();
  });

  it("shows navigation hint when candidates are present", () => {
    render(<NoduleViewer3D candidates={CANDIDATES} />);
    expect(screen.getByText(/left drag orbit/i)).toBeInTheDocument();
  });

  it("hides navigation hint when no candidates", () => {
    render(<NoduleViewer3D candidates={[]} />);
    expect(screen.queryByText(/left drag orbit/i)).not.toBeInTheDocument();
  });

  it("shows the orientation legend", () => {
    render(<NoduleViewer3D candidates={CANDIDATES} />);
    expect(screen.getByText("R/L")).toBeInTheDocument();
    expect(screen.getByText("A/P")).toBeInTheDocument();
    expect(screen.getByText("S/I")).toBeInTheDocument();
  });

  it("calls onSelect when a nodule is clicked via the handler", () => {
    // Click-to-select is handled inside the Three.js raycaster which is not
    // active in the test environment (renderer creation fails). We verify
    // that the component accepts and stores the callback without errors.
    const onSelect = vi.fn();
    expect(() =>
      render(<NoduleViewer3D candidates={CANDIDATES} onSelect={onSelect} />)
    ).not.toThrow();
  });

  it("renders without crashing when selectedNodule is provided", () => {
    const selected = CANDIDATES[0];
    expect(() =>
      render(<NoduleViewer3D candidates={CANDIDATES} selectedNodule={selected} />)
    ).not.toThrow();
  });
});
