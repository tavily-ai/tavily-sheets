import {
  KeyRound,
  ChevronDown,
  ChevronUp,
  CheckCircle2,
  Eye,
  EyeOff,
} from "lucide-react";
import React, { useState } from "react";

interface ApiKeyInputProps {
  apiKey?: string;
  setApiKey: (key: string) => void;
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
  checkApiKey: () => boolean | 0 | undefined;
}

const ApiKeyInput: React.FC<ApiKeyInputProps> = ({
  apiKey,
  setApiKey,
  isOpen,
  setIsOpen,
  checkApiKey,
}) => {
  const [showKey, setShowKey] = useState(false);

  return (
    <div className="w-full max-w-[calc(100%-1rem)] sm:max-w-3xl mt-6">
      <div
        className="flex items-center gap-2 px-3 sm:px-4 py-3 cursor-pointer transition min-h-[48px]"
        onClick={() => setIsOpen(!isOpen)}
        style={{ color: "var(--color-black-60)" }}
      >
        <div className="flex items-center gap-2">
          <KeyRound className="h-4 w-4" />
          <span className="text-sm">Enter your Tavily API Key</span>
        </div>
        {checkApiKey() && (
          <CheckCircle2 className="h-4 w-4" style={{ color: "var(--color-primary-blue)" }} />
        )}
        {isOpen ? (
          <ChevronUp className="h-4 w-4 ml-auto" />
        ) : (
          <ChevronDown className="h-4 w-4 ml-auto" />
        )}
      </div>
      <div
        className={`px-3 sm:px-4 overflow-hidden transition-all duration-300 ${
          isOpen
            ? "max-h-[200px] opacity-100 pb-4"
            : "max-h-0 opacity-0 pb-0"
        }`}
      >
        <div className="relative mt-2">
          <input
            type={showKey ? "text" : "password"}
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            className="w-full p-3 sm:p-3 pr-12 glass rounded-[12px] border-none outline-none text-base sm:text-sm"
            style={{
              color: "var(--color-black)",
              minHeight: "48px",
            }}
            placeholder="tvly-xxxx..."
            autoComplete="off"
            autoCorrect="off"
            autoCapitalize="off"
            spellCheck="false"
          />
          <button
            type="button"
            onClick={() => setShowKey(!showKey)}
            className="absolute right-3 top-1/2 -translate-y-1/2 transition p-2"
            style={{ color: "var(--color-black-40)" }}
          >
            {showKey ? (
              <EyeOff className="w-5 h-5" />
            ) : (
              <Eye className="w-5 h-5" />
            )}
          </button>
        </div>
        <p
          className="text-xs mt-2"
          style={{ color: "var(--color-black-40)" }}
        >
          Get your API key at{" "}
          <a
            href="https://app.tavily.com"
            target="_blank"
            rel="noopener noreferrer"
            style={{ color: "var(--color-primary-blue)" }}
            className="hover:underline"
          >
            app.tavily.com
          </a>
        </p>
      </div>
    </div>
  );
};

export default ApiKeyInput;
