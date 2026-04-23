import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import Viewer from "../Viewer";

describe("Viewer", () => {
  it("shows upload prompt when no sliceUrl is provided", () => {
    render(<Viewer baseSliceUrl={null} sliceIdx={0} />);
    expect(screen.getByText("Upload a scan to begin")).toBeInTheDocument();
  });

  it("renders the boxed image when URL is set", () => {
    const { container } = render(
      <Viewer
        baseSliceUrl="/api/slices/uid/axial?idx=5"
        sliceIdx={5}
      />
    );
    const imgs = container.querySelectorAll("img");
    expect(imgs).toHaveLength(1);
    expect(imgs[0]).toHaveAttribute("src", "/api/slices/uid/axial?idx=5");
    expect(imgs[0]).toHaveAttribute("alt", "slice 5");
  });

  it("shows 'Slice not available' after image load error", () => {
    render(<Viewer baseSliceUrl="/api/slices/uid/axial?idx=0" sliceIdx={0} />);
    const img = screen.getByRole("img");
    fireEvent.error(img);
    expect(screen.getByText("Slice not available")).toBeInTheDocument();
  });

  it("displays the slice index overlay when an image is loaded", () => {
    render(<Viewer baseSliceUrl="/api/slices/uid/axial?idx=42" sliceIdx={42} />);
    expect(screen.getByText(/Slice 42/)).toBeInTheDocument();
  });

  it("does not react to wheel events", () => {
    render(<Viewer baseSliceUrl="/api/slices/uid/axial?idx=10" sliceIdx={10} />);
    const container = screen.getByRole("img").closest("div");
    fireEvent.wheel(container, { deltaY: 50 });
    expect(screen.getByText("Slice 10")).toBeInTheDocument();
  });

  it("renders only one image element", () => {
    const { container } = render(
      <Viewer
        baseSliceUrl="/api/slices/uid/axial?idx=5"
        sliceIdx={5}
      />
    );
    const imgs = container.querySelectorAll("img");
    expect(imgs).toHaveLength(1);
  });
});
