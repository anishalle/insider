import { parsePolymarketContextFromUrl } from "../shared/page-context";

let lastSerializedContext = "";

const LOOKS_LIKE_OUTCOME_LABEL =
  /\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)\b/i;

const normalizeText = (value: string): string => value.replace(/\s+/g, " ").trim();
const normalizeComparableText = (value: string): string => normalizeText(value).toLowerCase().replace(/[^a-z0-9]+/g, "");

const findTradeButton = (): HTMLButtonElement | undefined =>
  Array.from(document.querySelectorAll<HTMLButtonElement>("button"))
    .filter((button) => {
      const comparableText =
        normalizeComparableText(button.innerText) || normalizeComparableText(button.getAttribute("aria-label") ?? "");
      if (comparableText !== "trade") {
        return false;
      }
      const rect = button.getBoundingClientRect();
      return rect.width > 0 && rect.height > 0;
    })
    .sort((left, right) => right.getBoundingClientRect().right - left.getBoundingClientRect().right)[0];

const detectSelectedLabel = (): string | undefined => {
  const tradeButton = findTradeButton();
  if (!tradeButton) {
    return undefined;
  }

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
    "by trading, you agree to the terms of use.",
  ]);

  const tradeRect = tradeButton.getBoundingClientRect();
  const tradeCenterX = tradeRect.left + tradeRect.width / 2;

  const candidates = Array.from(document.querySelectorAll<HTMLElement>("h1, h2, h3, h4, h5, p, span, div, button"))
    .map((element) => ({
      element,
      text: normalizeText(element.innerText),
      rect: element.getBoundingClientRect(),
    }))
    .filter(({ text, rect }) => Boolean(text) && rect.width > 0 && rect.height > 0)
    .filter(({ text }) => text.length <= 48)
    .filter(({ text }) => !ignored.has(normalizeComparableText(text)))
    .filter(({ text }) => !/\b(?:buy|sell|market)\b/i.test(text))
    .filter(({ text }) => !/[$¢%]/.test(text))
    .filter(({ text }) => !/^\d+(\.\d+)?$/.test(text))
    .filter(({ text }) => LOOKS_LIKE_OUTCOME_LABEL.test(text) || /no release/i.test(text))
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
      return Math.abs(left.rect.left + left.rect.width / 2 - tradeCenterX) - Math.abs(right.rect.left + right.rect.width / 2 - tradeCenterX);
    });

  return candidates[0]?.text;
};

const publishContext = (): void => {
  const baseContext = parsePolymarketContextFromUrl(window.location.href) ?? {
    url: window.location.href,
  };
  const context = {
    ...baseContext,
    selectedLabel: detectSelectedLabel(),
  };
  const serialized = JSON.stringify(context);
  if (serialized === lastSerializedContext) {
    return;
  }
  lastSerializedContext = serialized;
  void chrome.runtime.sendMessage({
    type: "insider:page-context",
    context,
  }).catch(() => undefined);
};

publishContext();

setInterval(() => {
  publishContext();
}, 1000);

window.addEventListener("popstate", publishContext);
