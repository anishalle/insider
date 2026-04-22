import type { MarketPageContext } from "./types";

export const parsePolymarketContextFromUrl = (rawUrl: string): MarketPageContext | null => {
  try {
    const url = new URL(rawUrl);
    if (url.hostname !== "polymarket.com") {
      return null;
    }
    const segments = url.pathname.split("/").filter(Boolean);
    const eventIndex = segments.indexOf("event");
    if (eventIndex < 0) {
      return null;
    }
    const eventSlug = segments[eventIndex + 1];
    const marketSlug = segments[eventIndex + 2];
    if (!eventSlug) {
      return null;
    }
    return {
      url: url.toString(),
      eventSlug,
      marketSlug,
    };
  } catch {
    return null;
  }
};
