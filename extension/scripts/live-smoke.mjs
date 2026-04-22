import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

import { chromium } from "playwright";

const targetUrl = process.argv[2] ?? "https://polymarket.com/event/gpt-5pt5-released-by";
const outputDir = process.argv[3]
  ? path.resolve(process.argv[3])
  : path.resolve(process.cwd(), "../output/playwright/live-smoke");
const extensionPath = path.resolve(process.cwd(), "dist");
const profileDir = process.env.INSIDER_EXTENSION_PROFILE ?? "/tmp/insider-extension-profile";

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const getExtensionState = async (serviceWorker) => {
  return serviceWorker.evaluate(async () => {
    const items = await new Promise((resolve) => chrome.storage.local.get(null, resolve));
    const activeTabs = await chrome.tabs.query({ active: true, currentWindow: true });
    return {
      items,
      activeTabs: activeTabs.map((tab) => ({
        id: tab.id ?? null,
        url: tab.url ?? null,
        title: tab.title ?? null,
      })),
    };
  });
};

const findMarketState = (items) => {
  const marketStateKey = Object.keys(items).find((key) => key.startsWith("insider:market-state:"));
  return marketStateKey ? { key: marketStateKey, value: items[marketStateKey] } : null;
};

await mkdir(outputDir, { recursive: true });

const context = await chromium.launchPersistentContext(profileDir, {
  headless: false,
  viewport: { width: 1440, height: 1100 },
  args: [
    `--disable-extensions-except=${extensionPath}`,
    `--load-extension=${extensionPath}`,
    "--no-first-run",
    "--no-default-browser-check",
  ],
});

try {
  let serviceWorker = context.serviceWorkers()[0] ?? null;
  if (!serviceWorker) {
    serviceWorker = await context.waitForEvent("serviceworker", { timeout: 15000 });
  }
  if (!serviceWorker) {
    throw new Error("Extension service worker did not start.");
  }

  const extensionId = new URL(serviceWorker.url()).host;
  const page = context.pages()[0] ?? (await context.newPage());
  const consoleMessages = [];
  const pageErrors = [];

  page.on("console", (message) => {
    consoleMessages.push({
      type: message.type(),
      text: message.text(),
    });
  });
  page.on("pageerror", (error) => {
    pageErrors.push(error.message);
  });

  await page.goto(targetUrl, { waitUntil: "domcontentloaded", timeout: 60000 });
  await page.waitForLoadState("networkidle", { timeout: 60000 }).catch(() => undefined);

  let extensionState = null;
  let marketState = null;
  for (let attempt = 0; attempt < 12; attempt += 1) {
    extensionState = await getExtensionState(serviceWorker);
    marketState = findMarketState(extensionState.items);
    if (marketState) {
      break;
    }
    await delay(5000);
  }

  await page.screenshot({
    path: path.join(outputDir, "market-page.png"),
    fullPage: true,
  });

  const summary = {
    testedAt: new Date().toISOString(),
    targetUrl,
    extensionId,
    activeTabs: extensionState?.activeTabs ?? [],
    storageKeys: extensionState ? Object.keys(extensionState.items).sort() : [],
    marketStateKey: marketState?.key ?? null,
    snapshotMode: marketState?.value?.snapshot?.mode ?? null,
    marketQuestion: marketState?.value?.snapshot?.market?.question ?? null,
    suspiciousWallets:
      marketState?.value?.snapshot?.suspiciousWallets?.slice(0, 5).map((wallet) => ({
        walletAddress: wallet.walletAddress,
        combinedScore: wallet.combinedScore,
        reasons: wallet.reasons,
      })) ?? [],
    linkedPairs:
      marketState?.value?.snapshot?.linkedPairs?.slice(0, 5).map((pair) => ({
        pairKey: pair.pairKey,
        score: pair.score,
        evidence: pair.evidence,
      })) ?? [],
    consoleMessages,
    pageErrors,
  };

  await writeFile(path.join(outputDir, "summary.json"), JSON.stringify(summary, null, 2));

  console.log(JSON.stringify(summary, null, 2));

  if (!marketState) {
    process.exitCode = 1;
    console.error("No insider market state was written to chrome.storage.local.");
  }
} finally {
  await context.close();
}
