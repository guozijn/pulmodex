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

  it("shows uploading label for UPLOADING", () => {
    render(<StatusBanner status="UPLOADING" />);
    expect(screen.getByText("Uploading scan…")).toBeInTheDocument();
  });

  it("shows running label for PROGRESS", () => {
    render(<StatusBanner status="PROGRESS" />);
    expect(screen.getByText("Running inference…")).toBeInTheDocument();
  });

  it("shows the backend progress step when provided", () => {
    render(<StatusBanner status="PROGRESS" step="detection" />);
    expect(screen.getByText("Running detection…")).toBeInTheDocument();
  });

  it("shows complete label for SUCCESS", () => {
    render(<StatusBanner status="SUCCESS" />);
    expect(screen.getByText("Complete")).toBeInTheDocument();
  });

  it("shows failed label for FAILURE", () => {
    render(<StatusBanner status="FAILURE" />);
    expect(screen.getByText("Failed")).toBeInTheDocument();
  });

  it("shows backend error details for FAILURE", () => {
    render(<StatusBanner status="FAILURE" error="No DICOM files found in uploaded zip" />);
    expect(screen.getByText("No DICOM files found in uploaded zip")).toBeInTheDocument();
  });

  it("shows elapsed runtime when provided", () => {
    render(<StatusBanner status="PROGRESS" elapsedSeconds={125} />);
    expect(screen.getByText("ELAPSED 02:05")).toBeInTheDocument();
  });

  it("falls back to raw status string for unknown states", () => {
    render(<StatusBanner status="UNKNOWN_STATE" />);
    expect(screen.getByText("UNKNOWN_STATE")).toBeInTheDocument();
  });
});
