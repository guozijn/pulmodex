import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi } from "vitest";
import UploadZone from "../UploadZone";

describe("UploadZone", () => {
  it("shows upload prompt when not disabled", () => {
    render(<UploadZone onUpload={vi.fn()} disabled={false} />);
    expect(screen.getByText(/Drop .mhd file here or click to upload/)).toBeInTheDocument();
  });

  it("shows processing message when disabled", () => {
    render(<UploadZone onUpload={vi.fn()} disabled={true} />);
    expect(screen.getByText("Processing…")).toBeInTheDocument();
  });

  it("calls onUpload with file when a file is selected via input", async () => {
    const onUpload = vi.fn();
    render(<UploadZone onUpload={onUpload} disabled={false} />);
    const input = document.querySelector("input[type='file']");
    const file = new File(["data"], "scan.mhd", { type: "" });
    await userEvent.upload(input, file);
    expect(onUpload).toHaveBeenCalledWith(file);
  });

  it("calls onUpload with dropped file when not disabled", () => {
    const onUpload = vi.fn();
    render(<UploadZone onUpload={onUpload} disabled={false} />);
    const dropZone = screen.getByText(/Drop .mhd file here/).closest("div");
    const file = new File(["data"], "scan.mhd", { type: "" });
    fireEvent.drop(dropZone, { dataTransfer: { files: [file] } });
    expect(onUpload).toHaveBeenCalledWith(file);
  });

  it("does not call onUpload when disabled and a file is dropped", () => {
    const onUpload = vi.fn();
    render(<UploadZone onUpload={onUpload} disabled={true} />);
    const dropZone = screen.getByText("Processing…").closest("div");
    const file = new File(["data"], "scan.mhd", { type: "" });
    fireEvent.drop(dropZone, { dataTransfer: { files: [file] } });
    expect(onUpload).not.toHaveBeenCalled();
  });
});
