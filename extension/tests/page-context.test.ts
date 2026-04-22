import { describe, expect, it } from "vitest";

import { parsePolymarketContextFromUrl } from "../src/shared/page-context";

describe("parsePolymarketContextFromUrl", () => {
  it("extracts an event slug from a standard Polymarket event URL", () => {
    expect(parsePolymarketContextFromUrl("https://polymarket.com/event/gpt-5pt5-released-on")).toEqual({
      url: "https://polymarket.com/event/gpt-5pt5-released-on",
      eventSlug: "gpt-5pt5-released-on",
      marketSlug: undefined,
    });
  });

  it("extracts event and market slugs when both are present", () => {
    expect(
      parsePolymarketContextFromUrl(
        "https://polymarket.com/event/gpt-5pt5-released-by/will-gpt-5pt5-be-released-by-april-25-2026",
      ),
    ).toEqual({
      url: "https://polymarket.com/event/gpt-5pt5-released-by/will-gpt-5pt5-be-released-by-april-25-2026",
      eventSlug: "gpt-5pt5-released-by",
      marketSlug: "will-gpt-5pt5-be-released-by-april-25-2026",
    });
  });

  it("ignores non-event routes", () => {
    expect(parsePolymarketContextFromUrl("https://polymarket.com/markets")).toBeNull();
  });
});
