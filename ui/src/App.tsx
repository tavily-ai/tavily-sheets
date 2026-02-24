import { useEffect, useState } from "react";
import { Header, Spreadsheet } from "./components";
import { GlassStyle, SpreadsheetData } from "./types";
import { motion } from "framer-motion";
import Toast from "./components/Toast";
import ApiKeyInput from "./components/ApiKeyInput";
import DemoBanner from "./components/DemoBanner";

const API_URL = import.meta.env.VITE_API_URL;
const WS_URL = import.meta.env.VITE_WS_URL;

const API_KEY_LENGTH = 32;

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

function App() {
  // const [isInfoPanelOpen, setIsInfoPanelOpen] = useState<boolean>(true);
  const [toastDetail, setToastDetail] = useState<ToastDetail>({});
  const [data, setData] = useState<SpreadsheetData>({
    headers: Array(5).fill(""),
    rows: Array(5)
      .fill(0)
      .map(() => Array(5).fill({ value: "" })),
  });

  const [apiKey, setApiKey] = useState<string>();
  const [isApiKeyDropdownOpen, setIsApiKeyDropdownOpen] =
    useState<boolean>(false);

  // Add these styles at the top of the component, before the return statement
  const glassStyle: GlassStyle = {
    base: "glass",
    card: "glass rounded-2xl p-6",
    input:
      "glass pl-10 w-full rounded-lg py-3 px-4 focus:border-[var(--color-primary-blue)]/50 focus:outline-none focus:ring-1 focus:ring-[var(--color-primary-blue)]/50 placeholder-dark",
  };

  const checkApiKey = () => {
    const splitKey = apiKey?.split("-");
    return (
      splitKey &&
      splitKey.length &&
      splitKey[splitKey.length - 1]?.length === API_KEY_LENGTH &&
      apiKey?.includes("tvly-")
    );
  };

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
      setApiKey(result.data);
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
      className="min-h-screen-dvh w-screen relative p-8"
      style={{ background: "var(--color-background)" }}
    >
      {/* Background Image Container - Fixed to viewport */}
      <div className="fixed inset-0 w-full h-full pointer-events-none">
        <img
          src="/tavily_landscapes_edited_11.webp"
          alt=""
          className="w-full h-full object-cover"
          style={{ opacity: 0.7 }}
        />
        {/* White gradient overlay at top */}
        <div
          className="absolute top-0 left-0 right-0 pointer-events-none"
          style={{
            height: "50%",
            background:
              "linear-gradient(to bottom, var(--color-background) 0%, var(--color-background) 10%, transparent 100%)",
          }}
        />
        {/* White gradient overlay at bottom for better readability */}
        <div
          className="absolute bottom-0 left-0 right-0 pointer-events-none"
          style={{
            height: "30%",
            background:
              "linear-gradient(to top, var(--color-background) 0%, transparent 100%)",
          }}
        />
      </div>

      {toastDetail.isShowing && (
        <Toast
          message={toastDetail.message}
          type={toastDetail.type}
          onClose={() => setToastDetail({})}
        />
      )}

      {/* Experimental Demo Banner - Fixed at top */}
      <DemoBanner />

      <div className="max-w-7xl mx-auto space-y-8 relative z-10" style={{ minHeight: "100vh", paddingBottom: "2rem" }}>
        {/* Header Component */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Header
            glassStyle={glassStyle.card}
          />
        </motion.div>

        {/* API Key Input - Always visible */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.1 }}
        >
          <ApiKeyInput
            apiKey={apiKey}
            setApiKey={setApiKey}
            isOpen={isApiKeyDropdownOpen}
            setIsOpen={setIsApiKeyDropdownOpen}
            checkApiKey={checkApiKey}
          />
        </motion.div>

        {/* Content wrapper - disabled when API key is missing */}
        <div
          className="relative"
          style={{
            opacity: checkApiKey() ? 1 : 0.7,
            pointerEvents: checkApiKey() ? "auto" : "none",
            transition: "opacity 0.3s ease-in-out"
          }}
        >
          {/* Overlay message when API key is missing */}
          {!checkApiKey() && (
            <motion.div
              className="absolute inset-0 z-50 flex items-center justify-center"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              style={{
                pointerEvents: "none",
                background: "rgba(255, 255, 255, 0.1)",
                backdropFilter: "blur(2px)",
                borderRadius: "1rem"
              }}
            >
              <div
                className="glass rounded-2xl p-6 text-center"
                style={{
                  pointerEvents: "auto",
                  backdropFilter: "none",
                  WebkitBackdropFilter: "none"
                }}
              >
                <p className="text-lg font-medium" style={{ color: "var(--color-black)", opacity: 1 }}>
                  Please enter your API key above to enable the table
                </p>
                <p className="text-sm mt-2" style={{ color: "var(--color-black-60)", opacity: 1 }}>
                  Get your API key at{" "}
                  <a
                    href="https://app.tavily.com"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="underline"
                    style={{ color: "var(--color-primary-blue)", opacity: 1 }}
                  >
                    app.tavily.com
                  </a>
                </p>
              </div>
            </motion.div>
          )}

          {/* Spreadsheet Component */}
          <motion.div
            className="relative"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <Spreadsheet
              data={data}
              setData={setData}
              setToast={setToastDetail}
              apiKey={apiKey || ""}
              checkApiKey={checkApiKey}
              isApiKeyDropdownOpen={isApiKeyDropdownOpen}
              setIsApiKeyDropdownOpen={setIsApiKeyDropdownOpen}
              setApiKey={setApiKey}
            />
          </motion.div>
        </div>
      </div>
      <a
        className="ot-sdk-show-settings px-4 py-2 text-sm text-gray-700 hover:text-gray-900 underline"
        href="#"
        style={{
          position: "fixed",
          bottom: "1rem",
          right: "1rem",
          zIndex: 50
        }}
      >
        Cookie Settings
      </a>
    </div>
  );
}

export default App;
