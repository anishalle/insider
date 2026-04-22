import { HORIZON_OPTIONS } from "../shared/constants";
import type { HorizonId, HorizonResult, MarketSnapshot, WalletMetricScore, WalletScore } from "../shared/types";
import { walletLabel } from "../shared/utils";

const subtitle = document.querySelector<HTMLParagraphElement>("#market-subtitle");
const modeValue = document.querySelector<HTMLElement>("#mode-value");
const walletCountValue = document.querySelector<HTMLElement>("#wallet-count-value");
const tradeCountValue = document.querySelector<HTMLElement>("#trade-count-value");
const spreadValue = document.querySelector<HTMLElement>("#spread-value");
const walletList = document.querySelector<HTMLDivElement>("#wallet-list");
const pairList = document.querySelector<HTMLDivElement>("#pair-list");
const horizonButtons = document.querySelector<HTMLDivElement>("#horizon-buttons");
const horizonOutput = document.querySelector<HTMLDivElement>("#horizon-output");
const refreshButton = document.querySelector<HTMLButtonElement>("#refresh-button");

let currentSnapshot: MarketSnapshot | null = null;
let activeHorizon: HorizonId | null = null;

const formatScore = (value: number | null): string => (value === null ? "-" : value.toFixed(1));

const renderMetric = (metric: WalletMetricScore): string => `
  <div class="metric-row">
    <div class="metric-bar-row">
      <span>${metric.label}</span>
      <span>${(metric.normalized * 100).toFixed(0)}%</span>
    </div>
    <div class="metric-bar" aria-hidden="true">
      <div class="metric-bar-fill" style="width: ${Math.round(metric.normalized * 100)}%"></div>
    </div>
  </div>
`;

const renderWalletCard = (wallet: WalletScore): string => `
  <article class="wallet-card">
    <div class="wallet-head">
      <div>
        <div class="wallet-score">${wallet.combinedScore.toFixed(1)}</div>
        <div class="wallet-meta">
          ${walletLabel(wallet.walletAddress)} • ${wallet.tradeCount} trades • ${wallet.dominantOutcome} • $${wallet.totalUsdc.toLocaleString()}
        </div>
      </div>
      <div class="wallet-meta">
        Live ${wallet.liveScore.toFixed(1)} / Post ${wallet.postEventScore.toFixed(1)}<br />
        Realized markout ${wallet.realizedMarkoutBps === null ? "-" : `${wallet.realizedMarkoutBps.toFixed(0)} bps`}
      </div>
    </div>
    <div class="reasons">
      ${wallet.reasons.length ? wallet.reasons.map((reason) => `<span class="chip">${reason}</span>`).join("") : `<span class="chip">No strong rule hits yet.</span>`}
    </div>
    <div class="metric-grid">
      ${wallet.metrics
        .filter((metric) => metric.weight > 0)
        .sort((left, right) => right.contribution - left.contribution)
        .slice(0, 5)
        .map(renderMetric)
        .join("")}
    </div>
  </article>
`;

const renderPairCard = (pair: MarketSnapshot["linkedPairs"][number]): string => `
  <article class="pair-card">
    <div class="pair-head">
      <div>
        <div class="pair-score">${pair.score.toFixed(1)}</div>
        <div class="pair-meta">${walletLabel(pair.leftWallet)} ↔ ${walletLabel(pair.rightWallet)}</div>
      </div>
      <div class="pair-meta">
        Entry ${(pair.synchronizedEntryScore * 100).toFixed(0)}% • Age ${(pair.accountAgeClosenessScore * 100).toFixed(0)}% • Side ${(pair.sameSideScore * 100).toFixed(0)}%
      </div>
    </div>
    <div class="evidence-list">
      ${pair.evidence.length ? pair.evidence.map((reason) => `<span class="chip">${reason}</span>`).join("") : `<span class="chip">Weak evidence.</span>`}
    </div>
  </article>
`;

const renderHorizonResult = (result: HorizonResult | undefined): void => {
  if (!horizonOutput) {
    return;
  }
  if (!result) {
    horizonOutput.innerHTML = `<div class="empty-state">No markout result is available for this horizon yet.</div>`;
    return;
  }
  horizonOutput.innerHTML = `
    <table class="horizon-table">
      <thead>
        <tr>
          <th>Wallet</th>
          <th>Weighted Markout</th>
          <th>Matched Trades</th>
        </tr>
      </thead>
      <tbody>
        ${result.wallets
          .slice(0, 10)
          .map((wallet) => {
            const markout = wallet.weightedMarkoutBps;
            const scoreClass = markout !== null && markout >= 0 ? "score-positive" : "score-negative";
            return `
              <tr>
                <td>${walletLabel(wallet.walletAddress)}</td>
                <td class="${scoreClass}">${markout === null ? "-" : `${markout.toFixed(0)} bps`}</td>
                <td>${wallet.matchedTradeCount}/${wallet.tradeCount}</td>
              </tr>
            `;
          })
          .join("")}
      </tbody>
    </table>
  `;
};

const renderSnapshot = (snapshot: MarketSnapshot | null): void => {
  currentSnapshot = snapshot;
  if (!snapshot) {
    if (subtitle) subtitle.textContent = "Waiting for a Polymarket market page.";
    if (modeValue) modeValue.textContent = "Idle";
    if (walletCountValue) walletCountValue.textContent = "0";
    if (tradeCountValue) tradeCountValue.textContent = "0";
    if (spreadValue) spreadValue.textContent = "-";
    if (walletList) walletList.innerHTML = `<div class="empty-state">Open a Polymarket event page to begin live scoring.</div>`;
    if (pairList) pairList.innerHTML = `<div class="empty-state">Linked-account signals appear once market activity is loaded.</div>`;
    renderHorizonResult(undefined);
    return;
  }

  if (subtitle) {
    subtitle.textContent = `${snapshot.market.question} (${snapshot.market.marketSlug})`;
  }
  if (modeValue) {
    modeValue.textContent = snapshot.mode === "live" ? "Live" : "Post-Event";
  }
  if (walletCountValue) {
    walletCountValue.textContent = snapshot.uniqueWalletCount.toLocaleString();
  }
  if (tradeCountValue) {
    tradeCountValue.textContent = snapshot.tradeCount.toLocaleString();
  }
  if (spreadValue) {
    spreadValue.textContent =
      snapshot.marketSignals.averageSpreadBps === null
        ? "-"
        : `${snapshot.marketSignals.averageSpreadBps.toFixed(0)} bps`;
  }
  if (walletList) {
    walletList.innerHTML = snapshot.suspiciousWallets.length
      ? snapshot.suspiciousWallets.map(renderWalletCard).join("")
      : `<div class="empty-state">No suspicious wallets surfaced yet for this market.</div>`;
  }
  if (pairList) {
    pairList.innerHTML = snapshot.linkedPairs.length
      ? snapshot.linkedPairs.map(renderPairCard).join("")
      : `<div class="empty-state">No strong linked-wallet pairs yet.</div>`;
  }
  renderHorizonResult(activeHorizon ? snapshot.horizonResults[activeHorizon] : undefined);
};

const requestSnapshot = async (message: { type: string; horizonId?: HorizonId }): Promise<MarketSnapshot | null> => {
  const response = await chrome.runtime.sendMessage(message).catch(() => ({ snapshot: null }));
  return (response?.snapshot as MarketSnapshot | null | undefined) ?? null;
};

const buildHorizonButtons = (): void => {
  if (!horizonButtons) {
    return;
  }
  horizonButtons.innerHTML = HORIZON_OPTIONS.map(
    (option) =>
      `<button class="button" data-horizon="${option.id}" type="button">${option.label}</button>`,
  ).join("");
  for (const button of Array.from(horizonButtons.querySelectorAll<HTMLButtonElement>("button[data-horizon]"))) {
    button.addEventListener("click", async () => {
      const horizonId = button.dataset.horizon as HorizonId;
      activeHorizon = horizonId;
      for (const peerButton of Array.from(horizonButtons.querySelectorAll<HTMLButtonElement>("button"))) {
        peerButton.classList.toggle("is-active", peerButton === button);
      }
      renderHorizonResult(undefined);
      const snapshot = await requestSnapshot({
        type: "insider:compute-horizon-markout",
        horizonId,
      });
      renderSnapshot(snapshot);
    });
  }
};

refreshButton?.addEventListener("click", async () => {
  const snapshot = await requestSnapshot({ type: "insider:refresh-active-market" });
  renderSnapshot(snapshot);
});

chrome.runtime.onMessage.addListener((message: unknown) => {
  const payload = message as { type?: string; snapshot?: MarketSnapshot };
  if (payload.type === "insider:snapshot-updated") {
    renderSnapshot(payload.snapshot ?? null);
  }
});

buildHorizonButtons();

void requestSnapshot({ type: "insider:get-active-snapshot" }).then((snapshot) => {
  renderSnapshot(snapshot);
});
