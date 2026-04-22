import { ACTIVE_TAB_QUERY, MARKET_REFRESH_ALARM, STORAGE_KEYS } from "../shared/constants";
import { computeHorizonMarkout } from "../shared/markout";
import { parsePolymarketContextFromUrl } from "../shared/page-context";
import { fetchBook, fetchHolders, fetchMarketTrades, fetchOpenInterest, fetchPriceHistory, fetchUserProfile, openMarketSocket, resolveMarketContext } from "../shared/polymarket";
import { analyzeMarket } from "../shared/scoring";
import type {
  HorizonId,
  MarketPageContext,
  MarketSnapshot,
  StoredMarketState,
  UserProfile,
} from "../shared/types";

interface MarketWatcher {
  conditionId: string;
  socket: WebSocket | null;
}

const setStorageItem = async (key: string, value: unknown): Promise<void> => {
  await chrome.storage.local.set({ [key]: value });
};

const getStorageItem = async <T>(key: string): Promise<T | null> => {
  const result = await chrome.storage.local.get(key);
  return (result[key] as T | undefined) ?? null;
};

export class MarketRuntime {
  private watchers = new Map<string, MarketWatcher>();
  private lastRealtimeRefreshAt = new Map<string, number>();

  private logFailure(scope: string, error: unknown, context?: Record<string, unknown>): void {
    console.error(`[insider] ${scope} failed`, {
      error,
      ...context,
    });
  }

  private async resolveContextFromTabs(tabs: chrome.tabs.Tab[]): Promise<MarketPageContext | null> {
    const orderedTabs = [...tabs].sort((left, right) => (right.lastAccessed ?? 0) - (left.lastAccessed ?? 0));
    for (const tab of orderedTabs) {
      const context = await this.resolveContextForTab(tab);
      if (context) {
        return context;
      }
    }
    return null;
  }

  private async resolveContextForTab(tab: chrome.tabs.Tab): Promise<MarketPageContext | null> {
    if (!tab.id) {
      return null;
    }
    const storedContext = await getStorageItem<MarketPageContext>(STORAGE_KEYS.tabContext(tab.id));
    if (storedContext?.eventSlug || storedContext?.marketSlug) {
      return this.hydrateTabContext(storedContext);
    }
    if (!tab.url) {
      return null;
    }
    const parsedContext = parsePolymarketContextFromUrl(tab.url);
    if (!parsedContext) {
      return null;
    }
    const nextContext = await this.hydrateTabContext({ ...parsedContext, tabId: tab.id });
    await setStorageItem(STORAGE_KEYS.tabContext(tab.id), nextContext);
    return nextContext;
  }

  private async readSelectedLabelFromTab(tabId: number): Promise<string | undefined> {
    try {
      const [result] = await chrome.scripting.executeScript({
        target: { tabId },
        func: () => {
          const normalizeText = (value: string): string => value.replace(/\s+/g, " ").trim();
          const normalizeComparableText = (value: string): string =>
            normalizeText(value).toLowerCase().replace(/[^a-z0-9]+/g, "");
          const looksLikeOutcome = (value: string): boolean =>
            /\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)\b/i.test(
              value,
            ) || /no release/i.test(value);

          const ignored = new Set([
            "buy",
            "sell",
            "market",
            "amount",
            "trade",
            "yes",
            "no",
            "rewards",
            "max",
            "by trading you agree to the terms of use",
          ]);

          const tradeButton = Array.from(document.querySelectorAll<HTMLButtonElement>("button"))
            .filter((button) => {
              const comparableText =
                normalizeComparableText(button.innerText) ||
                normalizeComparableText(button.getAttribute("aria-label") ?? "");
              if (comparableText !== "trade") {
                return false;
              }
              const rect = button.getBoundingClientRect();
              return rect.width > 0 && rect.height > 0;
            })
            .sort((left, right) => right.getBoundingClientRect().right - left.getBoundingClientRect().right)[0];

          if (!tradeButton) {
            return undefined;
          }

          const tradeRect = tradeButton.getBoundingClientRect();
          const tradeCenterX = tradeRect.left + tradeRect.width / 2;
          const candidates = Array.from(document.querySelectorAll<HTMLElement>("h1, h2, h3, h4, h5, p, span, div, button"))
            .map((element) => ({
              text: normalizeText(element.innerText),
              rect: element.getBoundingClientRect(),
            }))
            .filter(({ text, rect }) => Boolean(text) && rect.width > 0 && rect.height > 0)
            .filter(({ text }) => text.length <= 48)
            .filter(({ text }) => !ignored.has(normalizeComparableText(text)))
            .filter(({ text }) => !/\b(?:buy|sell|market)\b/i.test(text))
            .filter(({ text }) => !/[$¢%]/.test(text))
            .filter(({ text }) => !/^\d+(\.\d+)?$/.test(text))
            .filter(({ text }) => looksLikeOutcome(text))
            .filter(({ rect }) => rect.bottom <= tradeRect.top - 8)
            .filter(({ rect }) => rect.bottom >= tradeRect.top - 360)
            .filter(({ rect }) => rect.left >= tradeRect.left - 24)
            .filter(({ rect }) => rect.right <= tradeRect.right + 24)
            .sort((left, right) => {
              if (left.text.length !== right.text.length) {
                return left.text.length - right.text.length;
              }
              if (left.rect.bottom !== right.rect.bottom) {
                return right.rect.bottom - left.rect.bottom;
              }
              return (
                Math.abs(left.rect.left + left.rect.width / 2 - tradeCenterX) -
                Math.abs(right.rect.left + right.rect.width / 2 - tradeCenterX)
              );
            });

          return candidates[0]?.text;
        },
      });
      return typeof result?.result === "string" && result.result ? result.result : undefined;
    } catch {
      return undefined;
    }
  }

  private async hydrateTabContext(context: MarketPageContext): Promise<MarketPageContext> {
    if (!context.tabId) {
      return context;
    }
    const selectedLabel = await this.readSelectedLabelFromTab(context.tabId);
    if (!selectedLabel || selectedLabel === context.selectedLabel) {
      return context;
    }
    const nextContext = { ...context, selectedLabel };
    await setStorageItem(STORAGE_KEYS.tabContext(context.tabId), nextContext);
    return nextContext;
  }

  private async contextFromActiveTab(): Promise<MarketPageContext | null> {
    const activeContext = await this.resolveContextFromTabs(await chrome.tabs.query(ACTIVE_TAB_QUERY));
    if (activeContext) {
      return activeContext;
    }
    const marketTabs = await chrome.tabs.query({
      url: ["https://polymarket.com/event/*"],
    });
    if (!marketTabs.length) {
      return null;
    }
    return this.resolveContextFromTabs(marketTabs);
  }

  async setPageContext(tabId: number, context: MarketPageContext): Promise<void> {
    const nextContext = await this.hydrateTabContext({ ...context, tabId });
    await setStorageItem(STORAGE_KEYS.tabContext(tabId), nextContext);
    const snapshot = await this.refreshContext(nextContext).catch((error) => {
      this.logFailure("setPageContext.refreshContext", error, { tabId, context: nextContext });
      return null;
    });
    if (snapshot?.market.conditionId) {
      await setStorageItem(STORAGE_KEYS.tabCondition(tabId), snapshot.market.conditionId);
    }
  }

  async getActiveSnapshot(): Promise<MarketSnapshot | null> {
    const context = await this.contextFromActiveTab();
    if (!context?.tabId) {
      return null;
    }
    const conditionId = await getStorageItem<string>(STORAGE_KEYS.tabCondition(context.tabId));
    if (conditionId) {
      const storedState = await getStorageItem<StoredMarketState>(STORAGE_KEYS.marketState(conditionId));
      if (storedState?.snapshot) {
        return storedState.snapshot;
      }
    }
    return this.refreshContext(context).catch((error) => {
      this.logFailure("getActiveSnapshot.refreshContext", error, { context });
      return null;
    });
  }

  async refreshActiveMarket(): Promise<MarketSnapshot | null> {
    const context = await this.contextFromActiveTab();
    if (!context) {
      return null;
    }
    return this.refreshContext(context).catch((error) => {
      this.logFailure("refreshActiveMarket.refreshContext", error, { context });
      return null;
    });
  }

  async computeHorizonMarkoutForActiveMarket(horizonId: HorizonId): Promise<MarketSnapshot | null> {
    const snapshot = await this.getActiveSnapshot();
    if (!snapshot) {
      return null;
    }
    const storedState = await getStorageItem<StoredMarketState>(STORAGE_KEYS.marketState(snapshot.market.conditionId));
    if (!storedState) {
      return null;
    }
    const earliestTrade = Math.min(...storedState.input.trades.map((trade) => trade.timestamp));
    const latestTrade = Math.max(...storedState.input.trades.map((trade) => trade.timestamp));
    const endTs = horizonId === "to_resolution" ? latestTrade : latestTrade + 4 * 60 * 60 * 1000;
    const histories = await Promise.all(
      storedState.input.market.tokenIds.map(async (assetId) => [
        assetId,
        await fetchPriceHistory(assetId, earliestTrade, endTs),
      ] as const),
    );
    const priceHistories = Object.fromEntries(histories);
    const horizonResult = computeHorizonMarkout(
      storedState.input.market,
      storedState.input.trades,
      horizonId,
      priceHistories,
    );
    const nextSnapshot = {
      ...storedState.snapshot,
      horizonResults: {
        ...storedState.snapshot.horizonResults,
        [horizonId]: horizonResult,
      },
    };
    await this.storeState({
      ...storedState,
      snapshot: nextSnapshot,
      lastRefreshedAt: Date.now(),
    });
    return nextSnapshot;
  }

  async handleAlarm(alarmName: string): Promise<void> {
    if (alarmName !== MARKET_REFRESH_ALARM) {
      return;
    }
    const context = await this.contextFromActiveTab();
    if (!context?.tabId) {
      return;
    }
    await this.refreshContext(context);
  }

  private async refreshContext(context: MarketPageContext): Promise<MarketSnapshot | null> {
    const hydratedContext = await this.hydrateTabContext(context);
    if (!hydratedContext.eventSlug && !hydratedContext.marketSlug) {
      return null;
    }
    const market = await resolveMarketContext(hydratedContext);
    const [trades, holders, openInterest, bookResults] = await Promise.all([
      fetchMarketTrades(market.conditionId),
      fetchHolders(market.conditionId),
      fetchOpenInterest(market.conditionId),
      Promise.allSettled(market.tokenIds.map(async (assetId) => [assetId, await fetchBook(assetId)] as const)),
    ]);
    market.openInterest = openInterest ?? undefined;
    const books = bookResults.flatMap((result) => (result.status === "fulfilled" ? [result.value] : []));

    const interestingWallets = Array.from(
      new Set(
        [
          ...trades
            .sort((left, right) => right.usdcSize - left.usdcSize)
            .slice(0, 20)
            .map((trade) => trade.walletAddress),
          ...holders.slice(0, 10).map((holder) => holder.walletAddress),
        ].filter(Boolean),
      ),
    ).slice(0, 25);

    const userProfiles = await this.fetchUserProfiles(interestingWallets);
    const input = {
      market,
      trades,
      holders,
      userProfiles,
      books: Object.fromEntries(books),
    };
    const snapshot = analyzeMarket(input);
    const state: StoredMarketState = {
      input,
      snapshot,
      lastRefreshedAt: Date.now(),
    };
    await this.storeState(state);
    this.ensureWatcher(state);
    await chrome.alarms.create(MARKET_REFRESH_ALARM, { periodInMinutes: 1 });
    return snapshot;
  }

  private async fetchUserProfiles(walletAddresses: string[]): Promise<Record<string, UserProfile>> {
    const entries = await Promise.all(
      walletAddresses.map(async (walletAddress) => {
        try {
          return [walletAddress, await fetchUserProfile(walletAddress)] as const;
        } catch {
          return [
            walletAddress,
            {
              walletAddress,
              totalMarketsTraded: null,
              firstSeenTimestamp: null,
            },
          ] as const;
        }
      }),
    );
    return Object.fromEntries(entries);
  }

  private ensureWatcher(state: StoredMarketState): void {
    const currentWatcher = this.watchers.get(state.input.market.conditionId);
    if (currentWatcher?.socket) {
      return;
    }
    const socket = openMarketSocket(state.input.market.tokenIds, async (message) => {
      const eventType = String(message.event_type ?? message.type ?? "");
      if (!eventType) {
        return;
      }
      if (
        (eventType === "last_trade_price" || eventType === "price_change" || eventType === "market_resolved") &&
        this.canRefreshFromRealtime(state.input.market.conditionId)
      ) {
        await this.refreshContext({
          eventSlug: state.input.market.eventSlug,
          marketSlug: state.input.market.marketSlug,
          url: `https://polymarket.com/event/${state.input.market.eventSlug}`,
        });
      }
    });
    socket.addEventListener("close", () => {
      this.watchers.delete(state.input.market.conditionId);
    });
    this.watchers.set(state.input.market.conditionId, {
      conditionId: state.input.market.conditionId,
      socket,
    });
  }

  private canRefreshFromRealtime(conditionId: string): boolean {
    const currentTime = Date.now();
    const previousRefresh = this.lastRealtimeRefreshAt.get(conditionId) ?? 0;
    if (currentTime - previousRefresh < 10_000) {
      return false;
    }
    this.lastRealtimeRefreshAt.set(conditionId, currentTime);
    return true;
  }

  private async storeState(state: StoredMarketState): Promise<void> {
    await setStorageItem(STORAGE_KEYS.marketState(state.input.market.conditionId), state);
    await chrome.runtime.sendMessage({
      type: "insider:snapshot-updated",
      snapshot: state.snapshot,
    }).catch(() => undefined);
  }
}
