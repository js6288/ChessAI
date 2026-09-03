import { expect, test } from "@playwright/test";

test("human can move, undo, flip, and export tools remain available", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "人机对弈" })).toBeVisible();
  await expect(page.getByTestId("piece-h2")).toBeVisible();

  await page.getByRole("button", { name: /试锋/ }).click();
  await page.getByRole("button", { name: /另开一局/ }).click();
  await page.getByTestId("piece-h2").click();
  await page.getByTestId("square-e2").click();
  await expect(page.getByText("炮二平五")).toBeVisible();

  await page.getByRole("button", { name: /翻转|红在下/ }).click();
  await page.getByRole("button", { name: "悔棋" }).click();
  await expect(page.getByText("棋局未启，着法簿尚空。")).toBeVisible();
});

test("restart cancels an in-flight AI turn without a ghost move", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: /试锋/ }).click();
  await page.getByRole("button", { name: /另开一局/ }).click();
  await page.getByTestId("piece-h2").click();
  await page.getByTestId("square-e2").click();
  await page.getByRole("button", { name: "重开" }).click();
  await expect(page.getByText("PLY 000")).toBeVisible();
  await page.waitForTimeout(1_000);
  await expect(page.getByText("PLY 000")).toBeVisible();
});
