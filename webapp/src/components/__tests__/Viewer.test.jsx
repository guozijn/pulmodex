import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import Viewer from "../Viewer";

describe("Viewer", () => {
  it("shows upload prompt when no sliceUrl is provided", () => {
    render(<Viewer baseSliceUrl={null} overlaySliceUrl={null} sliceIdx={0} onSliceChange={vi.fn()} />);
    expect(screen.getByText("Upload a scan to begin")).toBeInTheDocument();
  });

  it("renders base and overlay images when URLs are set", () => {
    const { container } = render(
      <Viewer
        baseSliceUrl="/api/slices/uid/axial?idx=5&layer=base"
        overlaySliceUrl="/api/slices/uid/axial?idx=5&layer=overlay"
        sliceIdx={5}
        onSliceChange={vi.fn()}
      />
    );
    const imgs = container.querySelectorAll("img");
    expect(imgs[0]).toHaveAttribute("src", "/api/slices/uid/axial?idx=5&layer=base");
    expect(imgs[0]).toHaveAttribute("alt", "slice 5");
    expect(imgs[1]).toHaveAttribute("src", "/api/slices/uid/axial?idx=5&layer=overlay");
  });

  it("shows 'Slice not available' after image load error", () => {
    render(<Viewer baseSliceUrl="/api/slices/uid/axial?idx=0&layer=base" overlaySliceUrl={null} sliceIdx={0} onSliceChange={vi.fn()} />);
    const img = screen.getByRole("img");
    fireEvent.error(img);
    expect(screen.getByText("Slice not available")).toBeInTheDocument();
  });

  it("displays the slice index overlay when an image is loaded", () => {
    render(<Viewer baseSliceUrl="/api/slices/uid/axial?idx=42&layer=base" overlaySliceUrl={null} sliceIdx={42} onSliceChange={vi.fn()} />);
    expect(screen.getByText(/Slice 42/)).toBeInTheDocument();
  });

  it("increments sliceIdx on scroll down", () => {
    const onSliceChange = vi.fn();
    render(<Viewer baseSliceUrl="/api/slices/uid/axial?idx=10&layer=base" overlaySliceUrl={null} sliceIdx={10} onSliceChange={onSliceChange} />);
    const container = screen.getByRole("img").parentElement;
    fireEvent.wheel(container, { deltaY: 50 });
    expect(onSliceChange).toHaveBeenCalled();
    // Verify the updater function clamps correctly
    const updater = onSliceChange.mock.calls[0][0];
    expect(updater(10)).toBe(11);
  });

  it("does not decrement sliceIdx below 0 on scroll up", () => {
    const onSliceChange = vi.fn();
    render(<Viewer baseSliceUrl="/api/slices/uid/axial?idx=0&layer=base" overlaySliceUrl={null} sliceIdx={0} onSliceChange={onSliceChange} />);
    const container = screen.getByRole("img").parentElement;
    fireEvent.wheel(container, { deltaY: -50 });
    const updater = onSliceChange.mock.calls[0][0];
    expect(updater(0)).toBe(0);
  });

  it("does not increment beyond the known max slice index", () => {
    const onSliceChange = vi.fn();
    render(
      <Viewer
        baseSliceUrl="/api/slices/uid/axial?idx=12&layer=base"
        overlaySliceUrl={null}
        sliceIdx={12}
        onSliceChange={onSliceChange}
        maxSliceIdx={12}
      />
    );
    const container = screen.getByRole("img").parentElement;
    fireEvent.wheel(container, { deltaY: 50 });
    const updater = onSliceChange.mock.calls[0][0];
    expect(updater(12)).toBe(12);
  });

  it("hides overlay image when showOverlay is false", () => {
    const { container } = render(
      <Viewer
        baseSliceUrl="/api/slices/uid/axial?idx=5&layer=base"
        overlaySliceUrl="/api/slices/uid/axial?idx=5&layer=overlay"
        sliceIdx={5}
        onSliceChange={vi.fn()}
        showOverlay={false}
      />
    );
    const imgs = container.querySelectorAll("img");
    expect(imgs).toHaveLength(1);
  });
});
