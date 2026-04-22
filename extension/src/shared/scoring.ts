import { LIVE_WEIGHTS, POST_EVENT_WEIGHTS, SCORE_LABELS } from "./constants";
import { computeRealizedMarkoutByWallet } from "./markout";
import type {
  BookSummary,
  MarketAnalysisInput,
  MarketMode,
  MarketSnapshot,
  MarketTrade,
  MetricId,
  PairInsight,
  UserProfile,
  WalletMetricScore,
  WalletScore,
} from "./types";
import { average, clamp, groupBy, percentile, sortDescending, sum } from "./utils";

interface WalletContext {
  walletAddress: string;
  trades: MarketTrade[];
  profile?: UserProfile;
  totalUsdc: number;
  maxTradeUsdc: number;
  firstTradeTimestamp: number;
  lastTradeTimestamp: number;
  dominantOutcome: string;
  dominantAsset: string;
  tradeCount: number;
  buyCount: number;
  sellCount: number;
}

const DEADLINE_LOOKBACK_MS = 72 * 60 * 60 * 1000;
const SYNCHRONIZED_ENTRY_WINDOW_MS = 15 * 60 * 1000;
const SYNCHRONIZED_EXIT_WINDOW_MS = 30 * 60 * 1000;
const ACCOUNT_AGE_WINDOW_MS = 7 * 24 * 60 * 60 * 1000;
const FLOW_BUCKET_MS = 5 * 60 * 1000;

const modeForMarket = (input: MarketAnalysisInput): MarketMode =>
  input.market.active && !input.market.closed && !input.market.resolved ? "live" : "post_event";

const deadlineTimestamp = (input: MarketAnalysisInput): number | null => {
  const candidate = input.market.closeTime ?? input.market.endDate ?? input.market.resolutionDate;
  if (!candidate) {
    return null;
  }
  const parsed = Date.parse(candidate);
  return Number.isNaN(parsed) ? null : parsed;
};

const computeRealizedVolatility = (trades: MarketTrade[]): number => {
  const byAsset = groupBy([...trades].sort((left, right) => left.timestamp - right.timestamp), (trade) => trade.asset);
  const volatilities = [...byAsset.values()].map((assetTrades) => {
    const returns: number[] = [];
    for (let index = 1; index < assetTrades.length; index += 1) {
      const current = assetTrades[index];
      const previous = assetTrades[index - 1];
      if (!current || !previous || previous.price <= 0) {
        continue;
      }
      returns.push(Math.abs(current.price / previous.price - 1));
    }
    return average(returns);
  });
  return average(volatilities);
};

const buildWalletContexts = (input: MarketAnalysisInput): WalletContext[] => {
  const grouped = groupBy(
    [...input.trades].sort((left, right) => left.timestamp - right.timestamp),
    (trade) => trade.walletAddress,
  );

  return [...grouped.entries()].map(([walletAddress, trades]) => {
    const outcomeTotals = new Map<string, number>();
    const assetTotals = new Map<string, number>();
    for (const trade of trades) {
      outcomeTotals.set(trade.outcome, (outcomeTotals.get(trade.outcome) ?? 0) + trade.usdcSize);
      assetTotals.set(trade.asset, (assetTotals.get(trade.asset) ?? 0) + trade.usdcSize);
    }
    const dominantOutcome =
      [...outcomeTotals.entries()].sort((left, right) => right[1] - left[1])[0]?.[0] ??
      trades[0]?.outcome ??
      "Unknown";
    const dominantAsset =
      [...assetTotals.entries()].sort((left, right) => right[1] - left[1])[0]?.[0] ??
      trades[0]?.asset ??
      "";
    return {
      walletAddress,
      trades,
      profile: input.userProfiles[walletAddress],
      totalUsdc: sum(trades.map((trade) => trade.usdcSize)),
      maxTradeUsdc: Math.max(...trades.map((trade) => trade.usdcSize), 0),
      firstTradeTimestamp: trades[0]?.timestamp ?? 0,
      lastTradeTimestamp: trades.at(-1)?.timestamp ?? 0,
      dominantOutcome,
      dominantAsset,
      tradeCount: trades.length,
      buyCount: trades.filter((trade) => trade.side === "BUY").length,
      sellCount: trades.filter((trade) => trade.side === "SELL").length,
    };
  });
};

const buildBucketFlowShare = (trades: MarketTrade[]): Map<string, number> => {
  const marketBuckets = new Map<number, number>();
  const walletBuckets = new Map<string, number>();

  for (const trade of trades) {
    const bucket = Math.floor(trade.timestamp / FLOW_BUCKET_MS) * FLOW_BUCKET_MS;
    const signed = trade.side === "BUY" ? trade.usdcSize : -trade.usdcSize;
    marketBuckets.set(bucket, (marketBuckets.get(bucket) ?? 0) + signed);
    const walletBucketKey = `${trade.walletAddress}:${bucket}`;
    walletBuckets.set(walletBucketKey, (walletBuckets.get(walletBucketKey) ?? 0) + signed);
  }

  const shares = new Map<string, number>();
  for (const [walletBucketKey, walletSignedFlow] of walletBuckets) {
    const [, bucketString] = walletBucketKey.split(":");
    const bucket = Number(bucketString);
    const marketSignedFlow = Math.abs(marketBuckets.get(bucket) ?? 0);
    const share = marketSignedFlow > 0 ? clamp(Math.abs(walletSignedFlow) / marketSignedFlow, 0, 1) : 0;
    const walletAddress = walletBucketKey.split(":")[0] ?? walletBucketKey;
    shares.set(walletAddress, Math.max(shares.get(walletAddress) ?? 0, share));
  }
  return shares;
};

const buildPairInsights = (wallets: WalletContext[]): PairInsight[] => {
  const pairs: PairInsight[] = [];
  for (let leftIndex = 0; leftIndex < wallets.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < wallets.length; rightIndex += 1) {
      const left = wallets[leftIndex];
      const right = wallets[rightIndex];
      if (!left || !right) {
        continue;
      }
      const synchronizedEntryScore =
        1 - clamp(Math.abs(left.firstTradeTimestamp - right.firstTradeTimestamp) / SYNCHRONIZED_ENTRY_WINDOW_MS, 0, 1);
      const sameSideScore = left.dominantOutcome === right.dominantOutcome ? 1 : 0;
      const sizeSimilarityScore =
        1 - Math.abs(left.totalUsdc - right.totalUsdc) / Math.max(left.totalUsdc, right.totalUsdc, 1);
      const accountAgeClosenessScore =
        left.profile?.firstSeenTimestamp && right.profile?.firstSeenTimestamp
          ? 1 -
            clamp(
              Math.abs(left.profile.firstSeenTimestamp - right.profile.firstSeenTimestamp) / ACCOUNT_AGE_WINDOW_MS,
              0,
              1,
            )
          : 0;
      const leftSellTime = left.trades.find((trade) => trade.side === "SELL")?.timestamp ?? null;
      const rightSellTime = right.trades.find((trade) => trade.side === "SELL")?.timestamp ?? null;
      const synchronizedExitScore =
        leftSellTime && rightSellTime
          ? 1 - clamp(Math.abs(leftSellTime - rightSellTime) / SYNCHRONIZED_EXIT_WINDOW_MS, 0, 1)
          : 0;
      const score =
        100 *
        (0.3 * synchronizedEntryScore +
          0.2 * sameSideScore +
          0.18 * sizeSimilarityScore +
          0.17 * accountAgeClosenessScore +
          0.15 * synchronizedExitScore);
      if (score < 35) {
        continue;
      }
      const evidence: string[] = [];
      if (synchronizedEntryScore >= 0.7) {
        evidence.push("Entered the market in a tight time window.");
      }
      if (sameSideScore === 1) {
        evidence.push("Built exposure on the same outcome.");
      }
      if (accountAgeClosenessScore >= 0.7) {
        evidence.push("Observed account ages are unusually close.");
      }
      if (synchronizedExitScore >= 0.7) {
        evidence.push("Exits were synchronized.");
      }
      pairs.push({
        pairKey: [left.walletAddress, right.walletAddress].sort().join("|"),
        leftWallet: left.walletAddress,
        rightWallet: right.walletAddress,
        score: Number(score.toFixed(2)),
        synchronizedEntryScore: Number(synchronizedEntryScore.toFixed(3)),
        sameSideScore: Number(sameSideScore.toFixed(3)),
        sizeSimilarityScore: Number(sizeSimilarityScore.toFixed(3)),
        accountAgeClosenessScore: Number(accountAgeClosenessScore.toFixed(3)),
        synchronizedExitScore: Number(synchronizedExitScore.toFixed(3)),
        evidence,
      });
    }
  }
  return sortDescending(pairs, (pair) => pair.score).slice(0, 20);
};

const pairScoreByWallet = (pairs: PairInsight[]): Map<string, { maxScore: number; linkedWalletCount: number }> => {
  const map = new Map<string, { maxScore: number; linkedWalletCount: number }>();
  for (const pair of pairs) {
    for (const walletAddress of [pair.leftWallet, pair.rightWallet]) {
      const current = map.get(walletAddress) ?? { maxScore: 0, linkedWalletCount: 0 };
      current.maxScore = Math.max(current.maxScore, pair.score);
      if (pair.score >= 60) {
        current.linkedWalletCount += 1;
      }
      map.set(walletAddress, current);
    }
  }
  return map;
};

const metricScore = (
  id: MetricId,
  weight: number,
  raw: number,
  normalized: number,
  evidence: Record<string, unknown>,
): WalletMetricScore => ({
  id,
  label: SCORE_LABELS[id],
  weight,
  raw: Number(raw.toFixed(4)),
  normalized: Number(normalized.toFixed(4)),
  contribution: Number((normalized * weight).toFixed(4)),
  evidence,
});

const walletReasons = (metrics: WalletMetricScore[], mode: MarketMode): string[] => {
  return metrics
    .filter((metric) => metric.normalized >= 0.65 && metric.weight > 0)
    .sort((left, right) => right.contribution - left.contribution)
    .slice(0, 3)
    .map((metric) => metric.label + (metric.id === "realizedMarkout" && mode === "post_event" ? " lit up after the market closed." : "."));
};

export const analyzeMarket = (input: MarketAnalysisInput): MarketSnapshot => {
  const mode = modeForMarket(input);
  const deadline = deadlineTimestamp(input);
  const weights = mode === "live" ? LIVE_WEIGHTS : POST_EVENT_WEIGHTS;
  const marketTrades = [...input.trades].sort((left, right) => left.timestamp - right.timestamp);
  const wallets = buildWalletContexts(input);
  const marketTradeSizes = marketTrades.map((trade) => trade.usdcSize);
  const marketP95Trade = percentile(marketTradeSizes, 0.95);
  const marketMedianTrade = percentile(marketTradeSizes, 0.5);
  const marketTotalVolume = sum(marketTradeSizes);
  const realizedVolatility = computeRealizedVolatility(marketTrades);
  const bucketShares = buildBucketFlowShare(marketTrades);
  const pairInsights = buildPairInsights(wallets);
  const pairRollup = pairScoreByWallet(pairInsights);
  const realizedMarkout = computeRealizedMarkoutByWallet(marketTrades);
  const spreads = Object.values(input.books)
    .map((book) => (book.spread !== null ? book.spread * 10_000 : null))
    .filter((value): value is number => value !== null);
  const averageSpreadBps = spreads.length ? average(spreads) : null;

  const walletScores: WalletScore[] = wallets.map((wallet) => {
    const dominantBook: BookSummary | undefined = input.books[wallet.dominantAsset];
    const secondsBeforeDeadline =
      deadline !== null ? Math.max((deadline - wallet.firstTradeTimestamp) / 1000, 0) : null;
    const timingNormalized =
      secondsBeforeDeadline === null
        ? 0
        : 1 - clamp(((deadline ?? wallet.firstTradeTimestamp) - wallet.firstTradeTimestamp) / DEADLINE_LOOKBACK_MS, 0, 1);

    const marketSizeNormalized = clamp(wallet.maxTradeUsdc / Math.max(marketP95Trade * 1.25, 1), 0, 1);
    const walletSizeNormalized = clamp(wallet.maxTradeUsdc / Math.max(wallet.totalUsdc / wallet.tradeCount, marketMedianTrade, 1), 0, 1);

    const oddsExtremity = average(wallet.trades.map((trade) => Math.abs(trade.price - 0.5) * 2));
    const depthPressure = dominantBook
      ? clamp(wallet.maxTradeUsdc / Math.max(dominantBook.askDepthUsd, dominantBook.bidDepthUsd, 1), 0, 1)
      : 0;
    const oiPressure = input.market.openInterest
      ? clamp(wallet.maxTradeUsdc / Math.max(input.market.openInterest * 0.05, 1), 0, 1)
      : 0;
    const oddsDepthOiNormalized = 0.4 * depthPressure + 0.35 * oiPressure + 0.25 * oddsExtremity;

    const flowShare = bucketShares.get(wallet.walletAddress) ?? 0;
    const orderFlowNormalized = clamp(flowShare, 0, 1);

    const dominantOutcomeUsdc = sum(
      wallet.trades.filter((trade) => trade.outcome === wallet.dominantOutcome).map((trade) => trade.usdcSize),
    );
    const directionalNormalized = clamp(dominantOutcomeUsdc / Math.max(wallet.totalUsdc, 1), 0, 1);

    const spreadTightness = dominantBook?.spread !== null ? 1 - clamp((dominantBook.spread ?? 0) / 0.12, 0, 1) : 0.5;
    const volatilityNormalized = clamp(realizedVolatility / 0.06, 0, 1);
    const spreadVolatilityNormalized = 0.6 * spreadTightness + 0.4 * volatilityNormalized;

    const profile = wallet.profile;
    const firstSeenGapHours =
      profile?.firstSeenTimestamp !== null && profile?.firstSeenTimestamp !== undefined
        ? Math.max((wallet.firstTradeTimestamp - profile.firstSeenTimestamp) / 3_600_000, 0)
        : null;
    const noveltyScore =
      profile?.totalMarketsTraded !== null && profile?.totalMarketsTraded !== undefined
        ? 1 - clamp((profile.totalMarketsTraded - 1) / 20, 0, 1)
        : 0.2;
    const freshnessScore =
      firstSeenGapHours === null ? 0.2 : 1 - clamp(firstSeenGapHours / 72, 0, 1);
    const firstTradeShock = clamp(wallet.maxTradeUsdc / Math.max(marketP95Trade * 1.5, 1), 0, 1);
    const newWalletNormalized = 0.45 * noveltyScore + 0.3 * freshnessScore + 0.25 * firstTradeShock;

    const coordination = pairRollup.get(wallet.walletAddress);
    const linkedWalletNormalized = clamp((coordination?.maxScore ?? 0) / 100, 0, 1);

    const realizedBps = realizedMarkout.get(wallet.walletAddress) ?? null;
    const resolutionPayoutScore =
      input.market.resolved && input.market.resolution
        ? average(
            wallet.trades.map((trade) => {
              const winningOutcome = input.market.resolution?.trim().toLowerCase() === trade.outcome.trim().toLowerCase();
              const terminalPrice = winningOutcome ? 1 : 0;
              return clamp((trade.side === "BUY" ? (terminalPrice - trade.price) / Math.max(trade.price, 0.01) : (trade.price - terminalPrice) / Math.max(trade.price, 0.01)) / 0.5, 0, 1);
            }),
          )
        : 0;
    const realizedMarkoutNormalized =
      realizedBps !== null
        ? clamp(Math.max(realizedBps, 0) / 5_000, 0, 1)
        : resolutionPayoutScore;

    const metrics = [
      metricScore("timingAnomaly", weights.timingAnomaly, timingNormalized, timingNormalized, {
        secondsBeforeDeadline,
      }),
      metricScore("marketSizeAnomaly", weights.marketSizeAnomaly, wallet.maxTradeUsdc, marketSizeNormalized, {
        maxTradeUsdc: wallet.maxTradeUsdc,
        marketP95Trade,
      }),
      metricScore("walletSizeAnomaly", weights.walletSizeAnomaly, wallet.maxTradeUsdc, walletSizeNormalized, {
        maxTradeUsdc: wallet.maxTradeUsdc,
        walletAverageTradeUsdc: wallet.totalUsdc / Math.max(wallet.tradeCount, 1),
      }),
      metricScore("oddsDepthOiPressure", weights.oddsDepthOiPressure, oddsDepthOiNormalized, oddsDepthOiNormalized, {
        oddsExtremity,
        depthPressure,
        oiPressure,
      }),
      metricScore("orderFlowPressure", weights.orderFlowPressure, flowShare, orderFlowNormalized, {
        maxBucketFlowShare: flowShare,
      }),
      metricScore("directionalConcentration", weights.directionalConcentration, dominantOutcomeUsdc, directionalNormalized, {
        dominantOutcome: wallet.dominantOutcome,
        dominantOutcomeUsdc,
      }),
      metricScore("spreadVolatilityRegime", weights.spreadVolatilityRegime, spreadVolatilityNormalized, spreadVolatilityNormalized, {
        spread: dominantBook?.spread ?? null,
        realizedVolatility,
      }),
      metricScore("newWalletShock", weights.newWalletShock, newWalletNormalized, newWalletNormalized, {
        totalMarketsTraded: profile?.totalMarketsTraded ?? null,
        firstSeenGapHours,
      }),
      metricScore("linkedWalletCoordination", weights.linkedWalletCoordination, linkedWalletNormalized, linkedWalletNormalized, {
        linkedWalletCount: coordination?.linkedWalletCount ?? 0,
        strongestPairScore: coordination?.maxScore ?? 0,
      }),
      metricScore("realizedMarkout", weights.realizedMarkout, realizedBps ?? resolutionPayoutScore * 5_000, realizedMarkoutNormalized, {
        realizedMarkoutBps: realizedBps,
        marketResolved: input.market.resolved,
      }),
    ];

    const activeWeightTotal = metrics.reduce((total, metric) => total + metric.weight, 0);
    const weightedScore =
      activeWeightTotal > 0
        ? (metrics.reduce((total, metric) => total + metric.contribution, 0) / activeWeightTotal) * 100
        : 0;
    const liveScore = mode === "live" ? weightedScore : weightedScore * 0.92;
    const postEventScore = mode === "post_event" ? weightedScore : weightedScore * 0.88;
    return {
      walletAddress: wallet.walletAddress,
      totalUsdc: Number(wallet.totalUsdc.toFixed(2)),
      tradeCount: wallet.tradeCount,
      dominantOutcome: wallet.dominantOutcome,
      liveScore: Number(liveScore.toFixed(2)),
      postEventScore: Number(postEventScore.toFixed(2)),
      combinedScore: Number(weightedScore.toFixed(2)),
      realizedMarkoutBps: realizedBps !== null ? Number(realizedBps.toFixed(2)) : null,
      linkedWalletCount: coordination?.linkedWalletCount ?? 0,
      metrics,
      reasons: walletReasons(metrics, mode),
    };
  });

  return {
    generatedAt: Date.now(),
    mode,
    market: input.market,
    tradeCount: marketTrades.length,
    uniqueWalletCount: wallets.length,
    suspiciousWallets: sortDescending(walletScores, (wallet) => wallet.combinedScore).slice(0, 15),
    linkedPairs: pairInsights,
    recentTrades: marketTrades.slice(-20).reverse(),
    marketSignals: {
      averageSpreadBps: averageSpreadBps !== null ? Number(averageSpreadBps.toFixed(2)) : null,
      realizedVolatility: Number(realizedVolatility.toFixed(4)),
      totalVolumeUsdc: Number(marketTotalVolume.toFixed(2)),
      openInterest: input.market.openInterest ?? null,
    },
    horizonResults: {},
  };
};
