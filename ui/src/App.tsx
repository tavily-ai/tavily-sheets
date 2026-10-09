import { useEffect, useState } from "react";
import { Header, Spreadsheet } from "./components";
import { SpreadsheetData } from "./types";
import { motion } from "framer-motion";
import Toast from "./components/Toast";
import ApiKeyDialog, { ApiKeyStatus } from "./components/ApiKeyDialog";
const API_URL = import.meta.env.VITE_API_URL;
const WS_URL = import.meta.env.VITE_WS_URL;

if (!API_URL || !WS_URL) {
  throw new Error(
    "Environment variables VITE_API_URL and VITE_WS_URL must be set"
  );
}

// Add this near the top of the file, after the imports
const writingAnimation = `
@keyframes writing {
  0% {
    stroke-dashoffset: 1000;
  }
  100% {
    stroke-dashoffset: 0;
  }
}

.animate-writing {
  animation: writing 1.5s linear infinite;
}
`;

// Add this right after the imports
const style = document.createElement("style");
style.textContent = writingAnimation;
document.head.appendChild(style);


export type ToastDetail = {
  message?: string;
  type?: "success" | "error" | "info";
  isShowing?: boolean;
};

const normalizeApiKey = (value?: string): string =>
  (value ?? "").replace(/\s+/g, "").trim();

const isLikelyTavilyApiKey = (value?: string): boolean =>
  /^tvly-[A-Za-z0-9_-]{20,}$/.test(normalizeApiKey(value));

function App() {
  const [toastDetail, setToastDetail] = useState<ToastDetail>({});
  const [data, setData] = useState<SpreadsheetData>({
    headers: Array(5).fill(""),
    rows: Array(5)
      .fill(0)
      .map(() => Array(5).fill({ value: "" })),
  });

  const [apiKey, setApiKey] = useState<string>("");
  const [isApiKeyDialogOpen, setIsApiKeyDialogOpen] = useState<boolean>(false);

  const checkApiKey = () => {
    return isLikelyTavilyApiKey(apiKey);
  };

  const hasApiKey = checkApiKey();

  const fetchKey = async () => {
    try {
      const response = await fetch(`${API_URL}/api/verify-jwt`, {
        method: "GET",
        credentials: "include",
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "An error occurred");
      }

      const result = await response.json();
      setApiKey(normalizeApiKey(result.data));
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    if (!apiKey) {
      fetchKey();
    }
  }, []);

  return (
    <div
      className="min-h-screen-dvh w-full relative"
      style={{ background: "var(--color-background)" }}
    >
      <div className="landscape-background" aria-hidden="true">
        <img src="/tavily-landscape.jpg" alt="" />
      </div>

      {toastDetail.isShowing && (
        <Toast
          message={toastDetail.message}
          type={toastDetail.type}
          onClose={() => setToastDetail({})}
        />
      )}

      <div className="relative z-10 mx-auto w-[calc(100%-2rem)] sm:w-[calc(100%-4rem)] max-w-7xl pb-16">
        <motion.div
          initial={{ opacity: 0, y: -12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Header />
        </motion.div>

        {/* API key entry point - always visible */}
        <motion.div
          className="flex justify-center mt-2 mb-8"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.1 }}
        >
          <ApiKeyStatus
            isValid={hasApiKey}
            apiKey={apiKey}
            onOpen={() => setIsApiKeyDialogOpen(true)}
          />
        </motion.div>

        <ApiKeyDialog
          apiKey={apiKey}
          setApiKey={(key) => setApiKey(normalizeApiKey(key))}
          isOpen={isApiKeyDialogOpen}
          setIsOpen={setIsApiKeyDialogOpen}
        />

        {/* Content wrapper - disabled when API key is missing */}
        <div className="relative">
          {/* Overlay message when API key is missing. Kept outside the dimmed
              wrapper so it stays fully legible. */}
          {!hasApiKey && (
            <motion.div
              className="absolute inset-0 z-40 flex items-start justify-center pt-24"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              style={{
                pointerEvents: "none",
                borderRadius: "var(--surface-radius)",
              }}
            >
              <div
                className="glass rounded-[var(--surface-radius)] px-7 py-6 text-center max-w-sm"
                style={{ pointerEvents: "auto", background: "#fffdf7f2" }}
              >
                <p className="text-[15px] font-medium">
                  Add a Tavily API key to enable the table
                </p>
                <p
                  className="text-[12px] mt-2 leading-[1.6]"
                  style={{ color: "var(--color-muted)" }}
                >
                  Enrichment runs against your own Tavily account, so nothing is
                  researched until a key is connected.
                </p>
                <button
                  type="button"
                  className="btn-pill mt-5"
                  onClick={() => setIsApiKeyDialogOpen(true)}
                >
                  Add API key
                </button>
              </div>
            </motion.div>
          )}

          {/* Spreadsheet Component */}
          <motion.div
            className="relative"
            initial={{ opacity: 0 }}
            animate={{
              opacity: hasApiKey ? 1 : 0.62,
              filter: hasApiKey ? "blur(0px)" : "blur(2px)",
            }}
            transition={{ duration: 0.5, delay: 0.2 }}
            style={{ pointerEvents: hasApiKey ? "auto" : "none" }}
          >
            <Spreadsheet
              data={data}
              setData={setData}
              setToast={setToastDetail}
              apiKey={apiKey}
              checkApiKey={checkApiKey}
            />
          </motion.div>
        </div>
      </div>

      <a
        className="ot-sdk-show-settings text-[11px] underline"
        href="#"
        style={{
          position: "fixed",
          bottom: "1rem",
          right: "1rem",
          zIndex: 50,
          color: "var(--color-muted)",
        }}
      >
        Cookie Settings
      </a>
    </div>
  );
}

export default App;
