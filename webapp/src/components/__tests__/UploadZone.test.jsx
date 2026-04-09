import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi } from "vitest";
import UploadZone from "../UploadZone";

describe("UploadZone", () => {
  it("shows upload prompt when not disabled", () => {
    render(<UploadZone onUpload={vi.fn()} disabled={false} />);
    expect(screen.getByText(/Drop a .zip DICOM series here or click to upload/)).toBeInTheDocument();
  });

  it("shows processing message when disabled", () => {
    render(<UploadZone onUpload={vi.fn()} disabled={true} />);
    expect(screen.getByText("Processing…")).toBeInTheDocument();
  });

  it("shows uploading message when uploading", () => {
    render(<UploadZone onUpload={vi.fn()} disabled={true} uploading />);
    expect(screen.getByText("Uploading scan…")).toBeInTheDocument();
  });

  it("calls onUpload with file when a file is selected via input", async () => {
    const onUpload = vi.fn();
    render(<UploadZone onUpload={onUpload} disabled={false} />);
    const input = document.querySelector("input[type='file']");
    const file = new File(["data"], "scan.zip", { type: "application/zip" });
    await userEvent.upload(input, file);
    expect(onUpload).toHaveBeenCalledWith(file);
    expect(input.value).toBe("");
  });

  it("calls onUpload with dropped file when not disabled", () => {
    const onUpload = vi.fn();
    render(<UploadZone onUpload={onUpload} disabled={false} />);
    const dropZone = screen.getByText(/Drop a .zip DICOM series here/).closest("div");
    const file = new File(["data"], "scan.zip", { type: "application/zip" });
    fireEvent.drop(dropZone, { dataTransfer: { files: [file] } });
    expect(onUpload).toHaveBeenCalledWith(file);
  });

  it("does not call onUpload when disabled and a file is dropped", () => {
    const onUpload = vi.fn();
    render(<UploadZone onUpload={onUpload} disabled={true} />);
    const dropZone = screen.getByText("Processing…").closest("div");
    const file = new File(["data"], "scan.zip", { type: "application/zip" });
    fireEvent.drop(dropZone, { dataTransfer: { files: [file] } });
    expect(onUpload).not.toHaveBeenCalled();
  });
});
