import { MarketRuntime } from "./runtime";

const runtime = new MarketRuntime();

chrome.runtime.onInstalled.addListener(() => {
  void chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true });
});

chrome.runtime.onStartup.addListener(() => {
  void chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true });
});

chrome.runtime.onMessage.addListener((message: unknown, sender, sendResponse) => {
  const payload = message as { type?: string; [key: string]: unknown };
  const respond = async (): Promise<void> => {
    switch (payload.type) {
      case "insider:page-context": {
        if (typeof sender.tab?.id === "number" && payload.context && typeof payload.context === "object") {
          await runtime.setPageContext(sender.tab.id, payload.context as { url: string; eventSlug?: string; marketSlug?: string });
        }
        sendResponse({ ok: true });
        break;
      }
      case "insider:get-active-snapshot": {
        sendResponse({ snapshot: await runtime.getActiveSnapshot() });
        break;
      }
      case "insider:refresh-active-market": {
        sendResponse({ snapshot: await runtime.refreshActiveMarket() });
        break;
      }
      case "insider:compute-horizon-markout": {
        const horizonId = String(payload.horizonId ?? "") as "5m" | "30m" | "1h" | "4h" | "to_resolution";
        sendResponse({ snapshot: await runtime.computeHorizonMarkoutForActiveMarket(horizonId) });
        break;
      }
      default: {
        sendResponse({ ok: false });
      }
    }
  };

  void respond();
  return true;
});

chrome.alarms.onAlarm.addListener((alarm) => {
  void runtime.handleAlarm(alarm.name);
});
