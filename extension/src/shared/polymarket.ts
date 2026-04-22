import { CLOB_BASE_URL, DATA_BASE_URL, GAMMA_BASE_URL, MARKET_SOCKET_URL } from "./constants";
import type {
  BookSummary,
  HolderBalance,
  MarketInfo,
  MarketPageContext,
  MarketTrade,
  PricePoint,
  UserProfile,
} from "./types";
import { normalizeTimestamp, parseJsonArray, toNumber } from "./utils";

const fetchJson = async <T>(url: string): Promise<T> => {
  const response = await fetch(url, { credentials: "omit" });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} for ${url}`);
  }
  return (await response.json()) as T;
};

const queryString = (params: Record<string, string | number | boolean | undefined>): string => {
  const searchParams = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined) {
      continue;
    }
    searchParams.set(key, String(value));
  }
  const serialized = searchParams.toString();
  return serialized ? `?${serialized}` : "";
};

const candidateTimestamp = (payload: Record<string, unknown>): number => {
  const raw =
    (typeof payload.closeTime === "string" && payload.closeTime) ||
    (typeof payload.endDate === "string" && payload.endDate) ||
    (typeof payload.resolutionDate === "string" && payload.resolutionDate) ||
    "";
  const parsed = Date.parse(raw);
  return Number.isNaN(parsed) ? Number.POSITIVE_INFINITY : parsed;
};

const candidateLiquidityScore = (payload: Record<string, unknown>): number =>
  toNumber(payload.volume, 0) + toNumber(payload.liquidity, 0);

const normalizeLookupValue = (value: unknown): string =>
  String(value ?? "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();

const sortMarketCandidates = (markets: Record<string, unknown>[]): Record<string, unknown>[] =>
  [...markets].sort((left, right) => {
    const leftTime = candidateTimestamp(left);
    const rightTime = candidateTimestamp(right);
    if (leftTime !== rightTime) {
      return leftTime - rightTime;
    }
    return candidateLiquidityScore(right) - candidateLiquidityScore(left);
  });

const selectEventMarket = (
  markets: Record<string, unknown>[],
  context: MarketPageContext,
): Record<string, unknown> | undefined => {
  if (!markets.length) {
    return undefined;
  }

  if (context.marketSlug) {
    const exactMatch = markets.find((market) => String(market.slug ?? "") === context.marketSlug);
    if (exactMatch) {
      return exactMatch;
    }
  }

  const selectedLabel = normalizeLookupValue(context.selectedLabel);
  if (selectedLabel) {
    const labelMatches = markets.filter((market) =>
      [market.slug, market.question, market.title].some((value) => normalizeLookupValue(value).includes(selectedLabel)),
    );
    if (labelMatches.length) {
      return selectEventMarket(labelMatches, { ...context, selectedLabel: undefined });
    }
  }

  const groups = [
    markets.filter((market) => Boolean(market.active) && !Boolean(market.closed) && !Boolean(market.resolved)),
    markets.filter((market) => Boolean(market.active) && !Boolean(market.closed)),
    markets.filter((market) => !Boolean(market.resolved)),
    markets,
  ];

  for (const group of groups) {
    const candidates = sortMarketCandidates(
      group.filter((market) => parseJsonArray(market.clobTokenIds ?? market.clob_token_ids).length > 0),
    );
    if (candidates.length) {
      return candidates[0];
    }
  }

  return sortMarketCandidates(markets)[0];
};

const normalizeMarketInfo = (payload: Record<string, unknown>, eventPayload?: Record<string, unknown>): MarketInfo => {
  const outcomes = parseJsonArray(payload.outcomes ?? payload.outcomeNames);
  const outcomePrices = parseJsonArray(payload.outcomePrices).map((value) => toNumber(value));
  const tokenIds = parseJsonArray(payload.clobTokenIds ?? payload.clob_token_ids);
  return {
    marketId: String(payload.id ?? payload.marketId ?? payload.conditionId ?? ""),
    marketSlug: String(payload.slug ?? eventPayload?.slug ?? ""),
    conditionId: String(payload.conditionId ?? payload.condition_id ?? ""),
    eventId: String(payload.eventId ?? payload.event_id ?? eventPayload?.id ?? ""),
    eventSlug: String(payload.eventSlug ?? payload.event_slug ?? eventPayload?.slug ?? ""),
    eventTitle: String(eventPayload?.title ?? payload.title ?? payload.question ?? ""),
    question: String(payload.question ?? payload.title ?? eventPayload?.title ?? ""),
    ticker: typeof payload.ticker === "string" ? payload.ticker : undefined,
    active: Boolean(payload.active),
    closed: Boolean(payload.closed),
    resolved: Boolean(payload.resolved),
    resolution:
      typeof payload.resolution === "string"
        ? payload.resolution
        : typeof payload.outcome === "string"
          ? payload.outcome
          : undefined,
    endDate: typeof payload.endDate === "string" ? payload.endDate : undefined,
    closeTime: typeof payload.closeTime === "string" ? payload.closeTime : undefined,
    resolutionDate: typeof payload.resolutionDate === "string" ? payload.resolutionDate : undefined,
    outcomes,
    outcomePrices,
    tokenIds,
    volume: toNumber(payload.volume, NaN),
    liquidity: toNumber(payload.liquidity, NaN),
  };
};

const normalizeTrade = (payload: Record<string, unknown>): MarketTrade | null => {
  const walletAddress = String(payload.proxyWallet ?? payload.user ?? payload.walletAddress ?? "").toLowerCase();
  const side = String(payload.side ?? "").toUpperCase();
  if (!walletAddress || (side !== "BUY" && side !== "SELL")) {
    return null;
  }
  return {
    walletAddress,
    side: side as "BUY" | "SELL",
    asset: String(payload.asset ?? payload.asset_id ?? ""),
    conditionId: String(payload.conditionId ?? payload.condition_id ?? ""),
    marketSlug: payload.slug ? String(payload.slug) : undefined,
    eventSlug: payload.eventSlug ? String(payload.eventSlug) : undefined,
    outcome: String(payload.outcome ?? ""),
    outcomeIndex: toNumber(payload.outcomeIndex ?? payload.outcome_index),
    price: toNumber(payload.price),
    size: toNumber(payload.size),
    usdcSize: toNumber(payload.usdcSize ?? payload.usdc_size ?? payload.usd_amount ?? toNumber(payload.size) * toNumber(payload.price)),
    timestamp: normalizeTimestamp(payload.timestamp ?? payload.occurredAt ?? payload.occurred_at) ?? Date.now(),
    transactionHash: String(payload.transactionHash ?? payload.transaction_hash ?? ""),
  };
};

const normalizeHolder = (payload: Record<string, unknown>): HolderBalance => ({
  walletAddress: String(payload.holder ?? payload.user ?? payload.proxyWallet ?? "").toLowerCase(),
  asset: String(payload.asset ?? payload.asset_id ?? ""),
  amount: toNumber(payload.balance ?? payload.amount ?? payload.size),
  outcomeIndex: toNumber(payload.outcomeIndex ?? payload.outcome_index),
});

const normalizeBook = (assetId: string, payload: Record<string, unknown>): BookSummary => {
  const bids = Array.isArray(payload.bids) ? payload.bids : [];
  const asks = Array.isArray(payload.asks) ? payload.asks : [];
  const bestBid = bids.length ? toNumber((bids[0] as Record<string, unknown>).price) : null;
  const bestAsk = asks.length ? toNumber((asks[0] as Record<string, unknown>).price) : null;
  const midpoint = bestBid !== null && bestAsk !== null ? (bestBid + bestAsk) / 2 : null;
  const bidDepthUsd = bids.reduce((total, level) => {
    const entry = level as Record<string, unknown>;
    return total + toNumber(entry.price) * toNumber(entry.size);
  }, 0);
  const askDepthUsd = asks.reduce((total, level) => {
    const entry = level as Record<string, unknown>;
    return total + toNumber(entry.price) * toNumber(entry.size);
  }, 0);
  return {
    assetId,
    bestBid,
    bestAsk,
    midpoint,
    spread: bestBid !== null && bestAsk !== null ? bestAsk - bestBid : null,
    lastTradePrice: midpoint,
    bidDepthUsd,
    askDepthUsd,
  };
};

export const resolveMarketContext = async (context: MarketPageContext): Promise<MarketInfo> => {
  if (!context.eventSlug && !context.marketSlug) {
    throw new Error("No Polymarket slug detected on the page.");
  }

  let eventPayload: Record<string, unknown> | undefined;
  let marketPayload: Record<string, unknown> | undefined;

  if (context.eventSlug) {
    const eventCandidates = await fetchJson<Record<string, unknown>[]>(
      `${GAMMA_BASE_URL}/events${queryString({ slug: context.eventSlug, limit: 1 })}`,
    );
    eventPayload = eventCandidates[0];
    if (eventPayload && Array.isArray(eventPayload.markets)) {
      const markets = eventPayload.markets as Record<string, unknown>[];
      marketPayload = selectEventMarket(markets, context);
    }
  }

  if (!marketPayload && context.marketSlug) {
    const marketCandidates = await fetchJson<Record<string, unknown>[]>(
      `${GAMMA_BASE_URL}/markets${queryString({ slug: context.marketSlug, limit: 1 })}`,
    );
    marketPayload = marketCandidates[0];
  }

  if (!marketPayload) {
    throw new Error("Unable to resolve the active Polymarket market.");
  }

  return normalizeMarketInfo(marketPayload, eventPayload);
};

export const fetchMarketTrades = async (conditionId: string, limit = 500): Promise<MarketTrade[]> => {
  const payload = await fetchJson<Record<string, unknown>[]>(
    `${DATA_BASE_URL}/trades${queryString({ market: conditionId, limit, takerOnly: true })}`,
  );
  return payload.map(normalizeTrade).filter((trade): trade is MarketTrade => trade !== null);
};

export const fetchHolders = async (conditionId: string, limit = 20): Promise<HolderBalance[]> => {
  const payload = await fetchJson<Record<string, unknown>[]>(
    `${DATA_BASE_URL}/holders${queryString({ market: conditionId, limit, minBalance: 1 })}`,
  );
  return payload.map(normalizeHolder).filter((holder) => Boolean(holder.walletAddress));
};

export const fetchOpenInterest = async (conditionId: string): Promise<number | null> => {
  const payload = await fetchJson<unknown>(`${DATA_BASE_URL}/oi${queryString({ market: conditionId })}`);
  if (typeof payload === "number") {
    return payload;
  }
  if (payload && typeof payload === "object") {
    const objectPayload = payload as Record<string, unknown>;
    return toNumber(objectPayload.openInterest ?? objectPayload.oi ?? objectPayload.value, NaN);
  }
  return null;
};

export const fetchBook = async (assetId: string): Promise<BookSummary> => {
  const payload = await fetchJson<Record<string, unknown>>(
    `${CLOB_BASE_URL}/book${queryString({ token_id: assetId })}`,
  );
  return normalizeBook(assetId, payload);
};

export const fetchUserProfile = async (walletAddress: string): Promise<UserProfile> => {
  const [tradedPayload, firstActivityPayload] = await Promise.all([
    fetchJson<unknown>(`${DATA_BASE_URL}/traded${queryString({ user: walletAddress })}`),
    fetchJson<Record<string, unknown>[]>(
      `${DATA_BASE_URL}/activity${queryString({
        user: walletAddress,
        limit: 1,
        sortDirection: "ASC",
      })}`,
    ),
  ]);
  const totalMarketsTraded =
    typeof tradedPayload === "number"
      ? tradedPayload
      : tradedPayload && typeof tradedPayload === "object"
        ? toNumber((tradedPayload as Record<string, unknown>).count ?? (tradedPayload as Record<string, unknown>).total)
        : null;
  const firstActivity = firstActivityPayload[0];
  return {
    walletAddress,
    totalMarketsTraded,
    firstSeenTimestamp: normalizeTimestamp(firstActivity?.timestamp ?? firstActivity?.occurredAt ?? firstActivity?.occurred_at),
  };
};

export const fetchPriceHistory = async (
  assetId: string,
  startTs: number,
  endTs: number,
): Promise<PricePoint[]> => {
  const payload = await fetchJson<unknown>(
    `${CLOB_BASE_URL}/prices-history${queryString({
      market: assetId,
      startTs: Math.floor(startTs / 1000),
      endTs: Math.floor(endTs / 1000),
      fidelity: 1,
    })}`,
  );
  const history =
    payload && typeof payload === "object" && Array.isArray((payload as Record<string, unknown>).history)
      ? ((payload as Record<string, unknown>).history as Record<string, unknown>[])
      : [];
  return history
    .map((point) => ({
      timestamp: normalizeTimestamp(point.t ?? point.timestamp ?? point.time) ?? 0,
      price: toNumber(point.p ?? point.price),
    }))
    .filter((point) => point.timestamp > 0 && Number.isFinite(point.price));
};

export const openMarketSocket = (
  assetIds: string[],
  onMessage: (message: Record<string, unknown>) => void,
): WebSocket => {
  const socket = new WebSocket(MARKET_SOCKET_URL);
  socket.addEventListener("open", () => {
    socket.send(
      JSON.stringify({
        assets_ids: assetIds,
        type: "market",
      }),
    );
  });
  socket.addEventListener("message", (event) => {
    try {
      const payload = JSON.parse(String(event.data)) as Record<string, unknown> | Record<string, unknown>[];
      if (Array.isArray(payload)) {
        for (const message of payload) {
          onMessage(message);
        }
      } else {
        onMessage(payload);
      }
    } catch {
      // Ignore malformed frames.
    }
  });
  return socket;
};
