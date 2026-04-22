import type { HorizonId, HorizonResult, MarketInfo, MarketTrade, PricePoint } from "./types";
import { bpsFromPrices, groupBy } from "./utils";

interface InventoryLot {
  remainingSize: number;
  price: number;
}

export const computeRealizedMarkoutByWallet = (trades: MarketTrade[]): Map<string, number | null> => {
  const walletGroups = groupBy(
    [...trades].sort((left, right) => left.timestamp - right.timestamp),
    (trade) => `${trade.walletAddress}:${trade.asset}`,
  );
  const totals = new Map<string, { notional: number; bpsWeighted: number }>();

  for (const [walletAssetKey, walletTrades] of walletGroups) {
    const lots: InventoryLot[] = [];
    for (const trade of walletTrades) {
      if (trade.side === "BUY") {
        lots.push({ remainingSize: trade.size, price: trade.price });
        continue;
      }

      let remainingSellSize = trade.size;
      while (remainingSellSize > 0 && lots.length) {
        const lot = lots[0];
        const matchedSize = Math.min(remainingSellSize, lot.remainingSize);
        lot.remainingSize -= matchedSize;
        remainingSellSize -= matchedSize;
        const matchedNotional = matchedSize * lot.price;
        const matchedBps = bpsFromPrices(lot.price, trade.price, "BUY");
        const walletKey = walletAssetKey.split(":")[0] ?? walletAssetKey;
        const aggregate = totals.get(walletKey) ?? { notional: 0, bpsWeighted: 0 };
        aggregate.notional += matchedNotional;
        aggregate.bpsWeighted += matchedBps * matchedNotional;
        totals.set(walletKey, aggregate);
        if (lot.remainingSize <= 0) {
          lots.shift();
        }
      }
    }
  }

  const realized = new Map<string, number | null>();
  for (const [walletKey, aggregate] of totals) {
    realized.set(walletKey, aggregate.notional > 0 ? aggregate.bpsWeighted / aggregate.notional : null);
  }
  return realized;
};

const settlementPriceForTrade = (market: MarketInfo, trade: MarketTrade): number | null => {
  if (!market.resolved || !market.resolution) {
    return null;
  }
  const normalizedResolution = market.resolution.trim().toLowerCase();
  const outcome = trade.outcome.trim().toLowerCase();
  if (!normalizedResolution) {
    return null;
  }
  return normalizedResolution === outcome ? 1 : 0;
};

const findFuturePrice = (history: PricePoint[], targetTimestamp: number): number | null => {
  for (const point of history) {
    if (point.timestamp >= targetTimestamp) {
      return point.price;
    }
  }
  const lastPoint = history.at(-1);
  return lastPoint?.price ?? null;
};

export const computeHorizonMarkout = (
  market: MarketInfo,
  trades: MarketTrade[],
  horizonId: HorizonId,
  priceHistories: Record<string, PricePoint[]>,
): HorizonResult => {
  const walletTotals = new Map<string, { weightedBps: number; totalNotional: number; matchedTradeCount: number; tradeCount: number }>();
  const horizonMinutes =
    horizonId === "5m"
      ? 5
      : horizonId === "30m"
        ? 30
        : horizonId === "1h"
          ? 60
          : horizonId === "4h"
            ? 240
            : null;

  for (const trade of trades) {
    const walletAggregate = walletTotals.get(trade.walletAddress) ?? {
      weightedBps: 0,
      totalNotional: 0,
      matchedTradeCount: 0,
      tradeCount: 0,
    };
    walletAggregate.tradeCount += 1;
    const futurePrice =
      horizonMinutes === null
        ? settlementPriceForTrade(market, trade)
        : findFuturePrice(priceHistories[trade.asset] ?? [], trade.timestamp + horizonMinutes * 60_000);
    if (futurePrice !== null) {
      const notional = Math.max(trade.price * trade.size, trade.usdcSize, 1);
      walletAggregate.totalNotional += notional;
      walletAggregate.weightedBps += bpsFromPrices(trade.price, futurePrice, trade.side) * notional;
      walletAggregate.matchedTradeCount += 1;
    }
    walletTotals.set(trade.walletAddress, walletAggregate);
  }

  return {
    horizonId,
    computedAt: Date.now(),
    wallets: [...walletTotals.entries()]
      .map(([walletAddress, aggregate]) => ({
        walletAddress,
        weightedMarkoutBps:
          aggregate.totalNotional > 0 ? aggregate.weightedBps / aggregate.totalNotional : null,
        tradeCount: aggregate.tradeCount,
        matchedTradeCount: aggregate.matchedTradeCount,
      }))
      .sort((left, right) => (right.weightedMarkoutBps ?? -Infinity) - (left.weightedMarkoutBps ?? -Infinity)),
  };
};
