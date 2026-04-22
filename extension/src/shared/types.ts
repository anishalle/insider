export type MarketMode = "live" | "post_event";

export type MetricId =
  | "timingAnomaly"
  | "marketSizeAnomaly"
  | "walletSizeAnomaly"
  | "oddsDepthOiPressure"
  | "orderFlowPressure"
  | "directionalConcentration"
  | "spreadVolatilityRegime"
  | "newWalletShock"
  | "linkedWalletCoordination"
  | "realizedMarkout";

export type HorizonId = "5m" | "30m" | "1h" | "4h" | "to_resolution";

export interface MarketPageContext {
  url: string;
  eventSlug?: string;
  marketSlug?: string;
  selectedLabel?: string;
  tabId?: number;
}

export interface MarketInfo {
  marketId: string;
  marketSlug: string;
  conditionId: string;
  eventId: string;
  eventSlug: string;
  eventTitle: string;
  question: string;
  ticker?: string;
  active: boolean;
  closed: boolean;
  resolved: boolean;
  resolution?: string;
  endDate?: string;
  closeTime?: string;
  resolutionDate?: string;
  outcomes: string[];
  outcomePrices: number[];
  tokenIds: string[];
  volume?: number;
  liquidity?: number;
  openInterest?: number;
}

export interface MarketTrade {
  walletAddress: string;
  side: "BUY" | "SELL";
  asset: string;
  conditionId: string;
  marketSlug?: string;
  eventSlug?: string;
  outcome: string;
  outcomeIndex: number;
  price: number;
  size: number;
  usdcSize: number;
  timestamp: number;
  transactionHash: string;
}

export interface HolderBalance {
  walletAddress: string;
  asset: string;
  amount: number;
  outcomeIndex: number;
}

export interface UserProfile {
  walletAddress: string;
  totalMarketsTraded: number | null;
  firstSeenTimestamp: number | null;
}

export interface BookSummary {
  assetId: string;
  bestBid: number | null;
  bestAsk: number | null;
  midpoint: number | null;
  spread: number | null;
  lastTradePrice: number | null;
  bidDepthUsd: number;
  askDepthUsd: number;
}

export interface PricePoint {
  timestamp: number;
  price: number;
}

export interface WalletMetricScore {
  id: MetricId;
  label: string;
  weight: number;
  raw: number;
  normalized: number;
  contribution: number;
  evidence: Record<string, unknown>;
}

export interface WalletScore {
  walletAddress: string;
  totalUsdc: number;
  tradeCount: number;
  dominantOutcome: string;
  liveScore: number;
  postEventScore: number;
  combinedScore: number;
  realizedMarkoutBps: number | null;
  linkedWalletCount: number;
  metrics: WalletMetricScore[];
  reasons: string[];
}

export interface PairInsight {
  pairKey: string;
  leftWallet: string;
  rightWallet: string;
  score: number;
  synchronizedEntryScore: number;
  sameSideScore: number;
  sizeSimilarityScore: number;
  accountAgeClosenessScore: number;
  synchronizedExitScore: number;
  evidence: string[];
}

export interface HorizonWalletSummary {
  walletAddress: string;
  weightedMarkoutBps: number | null;
  tradeCount: number;
  matchedTradeCount: number;
}

export interface HorizonResult {
  horizonId: HorizonId;
  wallets: HorizonWalletSummary[];
  computedAt: number;
}

export interface MarketAnalysisInput {
  market: MarketInfo;
  trades: MarketTrade[];
  holders: HolderBalance[];
  userProfiles: Record<string, UserProfile>;
  books: Record<string, BookSummary>;
}

export interface MarketSnapshot {
  generatedAt: number;
  mode: MarketMode;
  market: MarketInfo;
  tradeCount: number;
  uniqueWalletCount: number;
  suspiciousWallets: WalletScore[];
  linkedPairs: PairInsight[];
  recentTrades: MarketTrade[];
  marketSignals: {
    averageSpreadBps: number | null;
    realizedVolatility: number;
    totalVolumeUsdc: number;
    openInterest: number | null;
  };
  horizonResults: Partial<Record<HorizonId, HorizonResult>>;
}

export interface StoredMarketState {
  input: MarketAnalysisInput;
  snapshot: MarketSnapshot;
  lastRefreshedAt: number;
}

export interface RuntimeMessageMap {
  "insider:page-context": { context: MarketPageContext };
  "insider:get-active-snapshot": Record<string, never>;
  "insider:refresh-active-market": Record<string, never>;
  "insider:compute-horizon-markout": { horizonId: HorizonId };
}
