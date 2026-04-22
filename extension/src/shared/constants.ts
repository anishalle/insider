export const GAMMA_BASE_URL = "https://gamma-api.polymarket.com";
export const DATA_BASE_URL = "https://data-api.polymarket.com";
export const CLOB_BASE_URL = "https://clob.polymarket.com";
export const MARKET_SOCKET_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market";

export const MARKET_REFRESH_ALARM = "insider-market-refresh";
export const ACTIVE_TAB_QUERY: chrome.tabs.QueryInfo = { active: true };

export const STORAGE_KEYS = {
  tabContext: (tabId: number) => `insider:tab-context:${tabId}`,
  marketState: (conditionId: string) => `insider:market-state:${conditionId}`,
  tabCondition: (tabId: number) => `insider:tab-condition:${tabId}`,
};

export const SCORE_LABELS = {
  timingAnomaly: "Pre-event Timing",
  marketSizeAnomaly: "Market-Relative Size",
  walletSizeAnomaly: "Wallet-Relative Size",
  oddsDepthOiPressure: "Odds / Depth / OI Pressure",
  orderFlowPressure: "Order-Flow Pressure",
  directionalConcentration: "Directional Concentration",
  spreadVolatilityRegime: "Spread / Volatility Regime",
  newWalletShock: "New Wallet / First-Trade Shock",
  linkedWalletCoordination: "Linked Wallet Coordination",
  realizedMarkout: "Realized / Resolved Markout",
} as const;

export const LIVE_WEIGHTS = {
  timingAnomaly: 16,
  marketSizeAnomaly: 12,
  walletSizeAnomaly: 8,
  oddsDepthOiPressure: 12,
  orderFlowPressure: 10,
  directionalConcentration: 8,
  spreadVolatilityRegime: 6,
  newWalletShock: 8,
  linkedWalletCoordination: 10,
  realizedMarkout: 0,
} as const;

export const POST_EVENT_WEIGHTS = {
  timingAnomaly: 14,
  marketSizeAnomaly: 10,
  walletSizeAnomaly: 8,
  oddsDepthOiPressure: 10,
  orderFlowPressure: 9,
  directionalConcentration: 8,
  spreadVolatilityRegime: 5,
  newWalletShock: 8,
  linkedWalletCoordination: 10,
  realizedMarkout: 18,
} as const;

export const HORIZON_OPTIONS = [
  { id: "5m", label: "5m", minutes: 5 },
  { id: "30m", label: "30m", minutes: 30 },
  { id: "1h", label: "1h", minutes: 60 },
  { id: "4h", label: "4h", minutes: 240 },
  { id: "to_resolution", label: "To Resolution", minutes: null },
] as const;
