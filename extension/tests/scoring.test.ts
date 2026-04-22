import { describe, expect, it } from "vitest";

import { computeHorizonMarkout, computeRealizedMarkoutByWallet } from "../src/shared/markout";
import { analyzeMarket } from "../src/shared/scoring";
import { demoMarketFixture } from "./fixtures/demo-market";

describe("market scoring", () => {
  it("surfaces linked wallets near the top of the suspicious ranking", () => {
    const snapshot = analyzeMarket(demoMarketFixture);
    expect(snapshot.mode).toBe("post_event");
    expect(snapshot.suspiciousWallets[0]?.walletAddress).toBe("0xa100000000000000000000000000000000000001");
    expect(snapshot.linkedPairs[0]?.score).toBeGreaterThan(60);
  });

  it("computes realized wallet markout from matched buy/sell lots", () => {
    const realized = computeRealizedMarkoutByWallet(demoMarketFixture.trades);
    expect((realized.get("0xa100000000000000000000000000000000000001") ?? 0)).toBeGreaterThan(7000);
    expect((realized.get("0xa100000000000000000000000000000000000002") ?? 0)).toBeGreaterThan(6500);
  });

  it("computes to-resolution markout when requested", () => {
    const result = computeHorizonMarkout(demoMarketFixture.market, demoMarketFixture.trades, "to_resolution", {});
    expect(result.wallets[0]?.weightedMarkoutBps ?? 0).toBeGreaterThan(3000);
    expect(result.wallets.some((wallet) => wallet.walletAddress === "0xb100000000000000000000000000000000000001")).toBe(true);
  });
});
