export const clamp = (value: number, min = 0, max = 1): number => Math.min(max, Math.max(min, value));

export const toNumber = (value: unknown, fallback = 0): number => {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return fallback;
};

export const parseJsonArray = (value: unknown): string[] => {
  if (Array.isArray(value)) {
    return value.map((item) => String(item));
  }
  if (typeof value === "string" && value.trim().startsWith("[")) {
    try {
      const parsed = JSON.parse(value);
      return Array.isArray(parsed) ? parsed.map((item) => String(item)) : [];
    } catch {
      return [];
    }
  }
  return [];
};

export const average = (values: number[]): number =>
  values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;

export const sum = (values: number[]): number => values.reduce((total, value) => total + value, 0);

export const percentile = (values: number[], p: number): number => {
  if (!values.length) {
    return 0;
  }
  const sorted = [...values].sort((left, right) => left - right);
  const index = clamp(p, 0, 1) * (sorted.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) {
    return sorted[lower] ?? 0;
  }
  const fraction = index - lower;
  return (sorted[lower] ?? 0) * (1 - fraction) + (sorted[upper] ?? 0) * fraction;
};

export const normalizeTimestamp = (value: unknown): number | null => {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value > 1e12 ? Math.floor(value) : Math.floor(value * 1000);
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Date.parse(value);
    return Number.isNaN(parsed) ? null : parsed;
  }
  return null;
};

export const groupBy = <T>(items: T[], keyFn: (item: T) => string): Map<string, T[]> => {
  const grouped = new Map<string, T[]>();
  for (const item of items) {
    const key = keyFn(item);
    const bucket = grouped.get(key);
    if (bucket) {
      bucket.push(item);
    } else {
      grouped.set(key, [item]);
    }
  }
  return grouped;
};

export const unique = <T>(items: T[]): T[] => Array.from(new Set(items));

export const sortDescending = <T>(items: T[], by: (item: T) => number): T[] =>
  [...items].sort((left, right) => by(right) - by(left));

export const walletLabel = (walletAddress: string): string =>
  walletAddress.length > 12
    ? `${walletAddress.slice(0, 6)}...${walletAddress.slice(-4)}`
    : walletAddress;

export const capitalizeWords = (value: string): string =>
  value
    .split(/[\s_-]+/)
    .filter(Boolean)
    .map((token) => token.charAt(0).toUpperCase() + token.slice(1))
    .join(" ");

export const now = (): number => Date.now();

export const bpsFromPrices = (entryPrice: number, futurePrice: number, side: "BUY" | "SELL"): number => {
  if (entryPrice <= 0) {
    return 0;
  }
  const sideSign = side === "BUY" ? 1 : -1;
  return sideSign * ((futurePrice / entryPrice) - 1) * 10_000;
};
