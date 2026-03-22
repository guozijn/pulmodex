import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import Viewer from "../Viewer";

describe("Viewer", () => {
  it("shows upload prompt when no sliceUrl is provided", () => {
    render(<Viewer sliceUrl={null} sliceIdx={0} onSliceChange={vi.fn()} saliencyAlpha={0.4} />);
    expect(screen.getByText("Upload a scan to begin")).toBeInTheDocument();
  });

  it("renders an img element when sliceUrl is set", () => {
    render(<Viewer sliceUrl="/api/slices/uid/axial?idx=5" sliceIdx={5} onSliceChange={vi.fn()} saliencyAlpha={0.4} />);
    const img = screen.getByRole("img");
    expect(img).toHaveAttribute("src", "/api/slices/uid/axial?idx=5");
    expect(img).toHaveAttribute("alt", "slice 5");
  });

  it("shows 'Slice not available' after image load error", () => {
    render(<Viewer sliceUrl="/api/slices/uid/axial?idx=0" sliceIdx={0} onSliceChange={vi.fn()} saliencyAlpha={0.4} />);
    const img = screen.getByRole("img");
    fireEvent.error(img);
    expect(screen.getByText("Slice not available")).toBeInTheDocument();
  });

  it("displays the slice index overlay when an image is loaded", () => {
    render(<Viewer sliceUrl="/api/slices/uid/axial?idx=42" sliceIdx={42} onSliceChange={vi.fn()} saliencyAlpha={0.4} />);
    expect(screen.getByText(/Slice 42/)).toBeInTheDocument();
  });

  it("increments sliceIdx on scroll down", () => {
    const onSliceChange = vi.fn();
    render(<Viewer sliceUrl="/api/slices/uid/axial?idx=10" sliceIdx={10} onSliceChange={onSliceChange} saliencyAlpha={0.4} />);
    const container = screen.getByRole("img").parentElement;
    fireEvent.wheel(container, { deltaY: 50 });
    expect(onSliceChange).toHaveBeenCalled();
    // Verify the updater function clamps correctly
    const updater = onSliceChange.mock.calls[0][0];
    expect(updater(10)).toBe(11);
  });

  it("does not decrement sliceIdx below 0 on scroll up", () => {
    const onSliceChange = vi.fn();
    render(<Viewer sliceUrl="/api/slices/uid/axial?idx=0" sliceIdx={0} onSliceChange={onSliceChange} saliencyAlpha={0.4} />);
    const container = screen.getByRole("img").parentElement;
    fireEvent.wheel(container, { deltaY: -50 });
    const updater = onSliceChange.mock.calls[0][0];
    expect(updater(0)).toBe(0);
  });
});
