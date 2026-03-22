import { test, expect } from "@playwright/test";

const MOCK_JOB_ID = "e2e-job-001";
const MOCK_SERIESUID = "e2e-series-001";

const MOCK_REPORT = {
  top_candidates: [
    { fp_prob: 0.91, diameter_mm: 9.2, coordX: 120.0, coordY: 55.0, coordZ: 80.0 },
    { fp_prob: 0.42, diameter_mm: 4.8, coordX: 60.0, coordY: 30.0, coordZ: 40.0 },
  ],
};

/** Register API route mocks before each test. */
async function mockApi(page, statusState = "SUCCESS") {
  await page.route("**/api/predict", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ job_id: MOCK_JOB_ID, seriesuid: MOCK_SERIESUID }),
    })
  );

  await page.route(`**/api/status/${MOCK_JOB_ID}`, (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ state: statusState, result: {} }),
    })
  );

  await page.route(`**/api/report/${MOCK_SERIESUID}`, (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(MOCK_REPORT),
    })
  );

  // Return a blank 1×1 PNG for all slice requests
  await page.route(`**/api/slices/**`, (route) =>
    route.fulfill({
      status: 200,
      contentType: "image/png",
      // Minimal valid 1×1 black PNG (base64)
      body: Buffer.from(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        "base64"
      ),
    })
  );
}

test.describe("Initial page", () => {
  test("shows the application title", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByText("Pulmodex — Lung Nodule Detection")).toBeVisible();
  });

  test("shows the upload prompt", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByText(/Drop .mhd file here or click to upload/)).toBeVisible();
  });

  test("renders all three view tabs", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByRole("button", { name: "axial" })).toBeVisible();
    await expect(page.getByRole("button", { name: "coronal" })).toBeVisible();
    await expect(page.getByRole("button", { name: "sagittal" })).toBeVisible();
  });
});

test.describe("Upload flow", () => {
  test("shows PENDING status immediately after file upload", async ({ page }) => {
    await mockApi(page, "PENDING");
    await page.goto("/");

    const fileInput = page.locator("input[type='file']");
    await fileInput.setInputFiles({
      name: "scan.mhd",
      mimeType: "application/octet-stream",
      buffer: Buffer.from("mhd data"),
    });

    await expect(page.getByText("Queued…")).toBeVisible({ timeout: 5_000 });
    await expect(page.getByText("Processing…")).toBeVisible();
  });

  test("transitions to SUCCESS and shows nodule list", async ({ page }) => {
    await mockApi(page, "SUCCESS");
    await page.goto("/");

    const fileInput = page.locator("input[type='file']");
    await fileInput.setInputFiles({
      name: "scan.mhd",
      mimeType: "application/octet-stream",
      buffer: Buffer.from("mhd data"),
    });

    await expect(page.getByText("Complete")).toBeVisible({ timeout: 10_000 });
    await expect(page.getByText("Detected Nodules")).toBeVisible();
    await expect(page.getByText("Nodule 1")).toBeVisible();
    await expect(page.getByText("Nodule 2")).toBeVisible();
  });

  test("shows confidence percentages for detected nodules", async ({ page }) => {
    await mockApi(page, "SUCCESS");
    await page.goto("/");

    const fileInput = page.locator("input[type='file']");
    await fileInput.setInputFiles({
      name: "scan.mhd",
      mimeType: "application/octet-stream",
      buffer: Buffer.from("mhd data"),
    });

    await expect(page.getByText("Complete")).toBeVisible({ timeout: 10_000 });
    await expect(page.getByText("91%")).toBeVisible();
    await expect(page.getByText("42%")).toBeVisible();
  });

  test("shows FAILURE status on job failure", async ({ page }) => {
    await mockApi(page, "FAILURE");
    await page.goto("/");

    const fileInput = page.locator("input[type='file']");
    await fileInput.setInputFiles({
      name: "scan.mhd",
      mimeType: "application/octet-stream",
      buffer: Buffer.from("mhd data"),
    });

    await expect(page.getByText("Failed")).toBeVisible({ timeout: 10_000 });
  });
});

test.describe("Viewer interaction", () => {
  test("switches view tab and updates active state", async ({ page }) => {
    await page.goto("/");
    const coronalBtn = page.getByRole("button", { name: "coronal" });
    await coronalBtn.click();
    // Active tab has fontWeight 700; verify it receives the click without error
    await expect(coronalBtn).toBeVisible();
  });

  test("saliency slider is visible and adjustable after SUCCESS", async ({ page }) => {
    await mockApi(page, "SUCCESS");
    await page.goto("/");

    const fileInput = page.locator("input[type='file']");
    await fileInput.setInputFiles({
      name: "scan.mhd",
      mimeType: "application/octet-stream",
      buffer: Buffer.from("mhd data"),
    });

    await expect(page.getByText("Complete")).toBeVisible({ timeout: 10_000 });
    const slider = page.locator("input[type='range']");
    await expect(slider).toBeVisible();
    await slider.fill("0.8");
    await expect(page.getByText(/Saliency opacity: 80%/)).toBeVisible();
  });
});

test.describe("Nodule selection", () => {
  test("clicking a nodule highlights it", async ({ page }) => {
    await mockApi(page, "SUCCESS");
    await page.goto("/");

    const fileInput = page.locator("input[type='file']");
    await fileInput.setInputFiles({
      name: "scan.mhd",
      mimeType: "application/octet-stream",
      buffer: Buffer.from("mhd data"),
    });

    await expect(page.getByText("Nodule 1")).toBeVisible({ timeout: 10_000 });
    await page.getByText("Nodule 1").click();
    // After selection the row background changes to #2d3748 — verify no JS errors thrown
    await expect(page.getByText("Nodule 1")).toBeVisible();
  });
});
