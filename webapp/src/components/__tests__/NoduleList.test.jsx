import React from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi } from "vitest";
import NoduleList from "../NoduleList";

const CANDIDATES = [
  { fp_prob: 0.92, diameter_mm: 8.4, coordX: 123.1, coordY: 45.6, coordZ: 78.0 },
  { fp_prob: 0.31, diameter_mm: 4.2, coordX: 55.0, coordY: 60.0, coordZ: 32.5 },
];

describe("NoduleList", () => {
  it("renders nothing when candidates is empty", () => {
    const { container } = render(<NoduleList candidates={[]} selected={null} onSelect={vi.fn()} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders nothing when candidates is undefined", () => {
    const { container } = render(<NoduleList candidates={undefined} selected={null} onSelect={vi.fn()} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders one row per candidate with correct label", () => {
    render(<NoduleList candidates={CANDIDATES} selected={null} onSelect={vi.fn()} />);
    expect(screen.getByText("Nodule 1")).toBeInTheDocument();
    expect(screen.getByText("Nodule 2")).toBeInTheDocument();
  });

  it("displays confidence percentage rounded correctly", () => {
    render(<NoduleList candidates={CANDIDATES} selected={null} onSelect={vi.fn()} />);
    expect(screen.getByText("92%")).toBeInTheDocument();
    expect(screen.getByText("31%")).toBeInTheDocument();
  });

  it("calls onSelect with the candidate and index when clicked", async () => {
    const onSelect = vi.fn();
    render(<NoduleList candidates={CANDIDATES} selected={null} onSelect={onSelect} />);
    await userEvent.click(screen.getByText("Nodule 2").closest("div[style]"));
    expect(onSelect).toHaveBeenCalledWith(CANDIDATES[1], 1);
  });

  it("falls back to prob field when fp_prob is absent", () => {
    const c = [{ prob: 0.75, diameter_mm: 6.0, coordX: 1, coordY: 2, coordZ: 3 }];
    render(<NoduleList candidates={c} selected={null} onSelect={vi.fn()} />);
    expect(screen.getByText("75%")).toBeInTheDocument();
  });
});
