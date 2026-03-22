import React from "react";
import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import StatusBanner from "../StatusBanner";

describe("StatusBanner", () => {
  it("renders nothing when status is null", () => {
    const { container } = render(<StatusBanner status={null} />);
    expect(container.firstChild).toBeNull();
  });

  it("shows queued label for PENDING", () => {
    render(<StatusBanner status="PENDING" />);
    expect(screen.getByText("Queued…")).toBeInTheDocument();
  });

  it("shows running label for PROGRESS", () => {
    render(<StatusBanner status="PROGRESS" />);
    expect(screen.getByText("Running inference…")).toBeInTheDocument();
  });

  it("shows complete label for SUCCESS", () => {
    render(<StatusBanner status="SUCCESS" />);
    expect(screen.getByText("Complete")).toBeInTheDocument();
  });

  it("shows failed label for FAILURE", () => {
    render(<StatusBanner status="FAILURE" />);
    expect(screen.getByText("Failed")).toBeInTheDocument();
  });

  it("falls back to raw status string for unknown states", () => {
    render(<StatusBanner status="UNKNOWN_STATE" />);
    expect(screen.getByText("UNKNOWN_STATE")).toBeInTheDocument();
  });
});
