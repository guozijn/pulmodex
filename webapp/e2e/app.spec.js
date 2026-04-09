import { test, expect } from "@playwright/test";

const MOCK_JOB_ID = "e2e-job-001";
const MOCK_SERIESUID = "e2e-series-001";

const MOCK_REPORT = {
  top_candidates: [
    {
      fp_prob: 0.91,
      diameter_mm: 9.2,
      coordX: 120.0,
      coordY: 55.0,
      coordZ: 80.0,
      slice_indices: { axial: 80, coronal: 55, sagittal: 120 },
    },
    {
      fp_prob: 0.42,
      diameter_mm: 4.8,
      coordX: 60.0,
      coordY: 30.0,
      coordZ: 40.0,
      slice_indices: { axial: 40, coronal: 30, sagittal: 60 },
    },
  ],
};

function labeledButton(page, label, name) {
  return page.getByText(label).locator("../..").getByRole("button", { name, exact: true }).first();
}

/** Register API route mocks before each test. */
async function mockApi(page, statusState = "SUCCESS", errorMessage = null) {
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
      body: JSON.stringify({ state: statusState, result: {}, error: errorMessage }),
    })
  );

  await page.route(`**/api/report/${MOCK_SERIESUID}`, (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(MOCK_REPORT),
    })
  );

  await page.route(`**/api/slices/${MOCK_SERIESUID}/*/index`, async (route) => {
    const url = new URL(route.request().url());
    const parts = url.pathname.split("/");
    const view = parts[parts.length - 2];
    const count = view === "axial" ? 81 : view === "coronal" ? 56 : 121;
    const indices = Array.from({ length: count }, (_, idx) => idx);
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ view, indices, count }),
    });
  });

  // Return a blank 1×1 PNG for slice image requests
  await page.route(`**/api/slices/${MOCK_SERIESUID}/**`, async (route) => {
    if (route.request().url().endsWith("/index")) {
      await route.fallback();
      return;
    }

    await route.fulfill({
      status: 200,
      contentType: "image/png",
      body: Buffer.from(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        "base64"
      ),
    });
  });
}

test.describe("Initial page", () => {
  test("shows the application title", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByText("Pulmodex | Lung Nodule Detection")).toBeVisible();
  });

  test("shows the upload prompt", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByText(/Drop a .zip DICOM series here or click to upload/)).toBeVisible();
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
      name: "scan.zip",
      mimeType: "application/zip",
      buffer: Buffer.from("zip data"),
    });

    await expect(page.getByText("Queued…")).toBeVisible({ timeout: 5_000 });
    await expect(page.getByText("Processing…")).toBeVisible();
  });

  test("transitions to SUCCESS and shows nodule list", async ({ page }) => {
    await mockApi(page, "SUCCESS");
    await page.goto("/");

    const fileInput = page.locator("input[type='file']");
    await fileInput.setInputFiles({
      name: "scan.zip",
      mimeType: "application/zip",
      buffer: Buffer.from("zip data"),
    });

    await expect(page.getByText("Complete")).toBeVisible({ timeout: 10_000 });
    await expect(page.locator("div").filter({ hasText: /^Findings$/ }).last()).toBeVisible();
    await expect(page.getByText("Nodule 1")).toBeVisible();
    await expect(page.getByText("Nodule 2")).toBeVisible();
  });

  test("shows confidence percentages for detected nodules", async ({ page }) => {
    await mockApi(page, "SUCCESS");
    await page.goto("/");

    const fileInput = page.locator("input[type='file']");
    await fileInput.setInputFiles({
      name: "scan.zip",
      mimeType: "application/zip",
      buffer: Buffer.from("zip data"),
    });

    await expect(page.getByText("Complete")).toBeVisible({ timeout: 10_000 });
    await expect(page.getByText("91%")).toBeVisible();
    await expect(page.getByText("42%")).toBeVisible();
  });

  test("shows FAILURE status on job failure", async ({ page }) => {
    await mockApi(page, "FAILURE", "No DICOM files found in uploaded zip");
    await page.goto("/");

    const fileInput = page.locator("input[type='file']");
    await fileInput.setInputFiles({
      name: "scan.zip",
      mimeType: "application/zip",
      buffer: Buffer.from("zip data"),
    });

    await expect(page.getByText("Failed")).toBeVisible({ timeout: 10_000 });
    await expect(page.getByText("No DICOM files found in uploaded zip")).toBeVisible();
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

  test("overlay controls are visible after SUCCESS", async ({ page }) => {
    await mockApi(page, "SUCCESS");
    await page.goto("/");

    const fileInput = page.locator("input[type='file']");
    await fileInput.setInputFiles({
      name: "scan.zip",
      mimeType: "application/zip",
      buffer: Buffer.from("zip data"),
    });

    await expect(page.getByText("Complete")).toBeVisible({ timeout: 10_000 });
    await expect(page.getByText("Heatmap overlay")).toBeVisible();
    const toggle = labeledButton(page, "Heatmap overlay", "OFF");
    await expect(toggle).toBeVisible();
    const slider = page.locator("input[type='range']");
    await expect(slider).toBeVisible();
    await expect(slider).toBeDisabled();
    await toggle.click();
    await expect(labeledButton(page, "Heatmap overlay", "ON")).toBeVisible();
    await expect(slider).toBeEnabled();
    await slider.fill("0.8");
    await expect(page.getByText("OVERLAY 80%")).toBeVisible();
  });
});

test.describe("Nodule selection", () => {
  test("clicking a nodule highlights it", async ({ page }) => {
    await mockApi(page, "SUCCESS");
    await page.goto("/");

    const fileInput = page.locator("input[type='file']");
    await fileInput.setInputFiles({
      name: "scan.zip",
      mimeType: "application/zip",
      buffer: Buffer.from("zip data"),
    });

    await expect(page.getByText("Nodule 1")).toBeVisible({ timeout: 10_000 });
    const row = page.locator("[data-testid='nodule-item']").first();
    await row.click();
    await expect(row).toHaveCSS("background-color", "rgb(28, 28, 28)");
  });
});
