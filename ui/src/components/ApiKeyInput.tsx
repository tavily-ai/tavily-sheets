import {
  KeyRound,
  ChevronDown,
  ChevronUp,
  CheckCircle2,
  Eye,
  EyeOff,
} from "lucide-react";
import React, { useState } from "react";
import { motion } from "framer-motion";

interface ApiKeyInputProps {
  glassStyle?: string;
  apiKey: string;
  setApiKey?: (key: string) => void;
  onApiKeyChange?: (key: string) => void;
  isOpen?: boolean;
  setIsOpen?: (open: boolean) => void;
  checkApiKey?: () => boolean;
}

const ApiKeyInput: React.FC<ApiKeyInputProps> = ({
  glassStyle = "",
  apiKey,
  setApiKey,
  onApiKeyChange,
  isOpen = true,
  setIsOpen,
  checkApiKey,
}) => {
  // Toggle function to open or close the form
  const toggleAccordion = () => setIsOpen?.(!(isOpen));
  const [showKey, setShowKey] = useState(false);

  // Handle API key changes with fallback
  const handleApiKeyChange = (value: string) => {
    setApiKey?.(value);
    onApiKeyChange?.(value);
  };

  // Check if API key is valid (fallback to checking if it's not empty)
  const isApiKeyValid = () => {
    if (checkApiKey) {
      return checkApiKey();
    }
    return apiKey && apiKey.trim().length > 0;
  };

  return (
    <motion.div
      className={`${glassStyle} mb-4 rounded-lg shadow-sm px-4 py-2`}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div className="relative">
        {/* Main Card for API Keys */}

        {/* Header with a button to toggle the form visibility */}
        <div
          className="flex justify-between items-center p-4 cursor-pointer"
          onClick={toggleAccordion} // Toggle on header click
        >
          <div className="flex items-center gap-3">
            <span className="text-xl font-medium text-gray-700">
              Tavily API Key
            </span>
            <span className="text-xs font-medium text-gray-500 px-2 py-1 bg-gray-100 rounded-md mt-0.5">
              required
            </span>
            {isApiKeyValid() && (
              <CheckCircle2 className="h-5 w-5 mt-0.5 text-[#22C55E]" />
            )}
          </div>
          <span className="text-xl text-gray-500">
            {isOpen ? <ChevronUp /> : <ChevronDown />}
            {/* Show up/down arrow based on state */}
          </span>
        </div>

        {/* Form Fields (conditionally rendered based on `isOpen` state) */}
        {isOpen && (
          <div className="space-y-6 pb-3 px-2 transition-all duration-500 ease-in-out">
            <div className="relative group">
              <div className="relative">
                <KeyRound
                  className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 stroke-[#468BFF] transition-all duration-200 group-hover:stroke-[#8FBCFA] z-10"
                  strokeWidth={1.5}
                />
                <input
                  required
                  id="tavilyApiKey"
                  type={showKey ? "text" : "password"}
                  value={apiKey}
                  spellCheck="false"
                  onChange={(e) => handleApiKeyChange(e.target.value)}
                  className={`backdrop-filter backdrop-blur-lg bg-white/80 border border-gray-200 shadow-xl pl-10 w-full rounded-lg py-3 px-4 text-gray-900 focus:border-[#468BFF]/50 focus:outline-none focus:ring-1 focus:ring-[#468BFF]/50 placeholder-gray-400 bg-white/80 shadow-none transition-all duration-300 focus:border-[#468BFF]/50 focus:ring-1 focus:ring-[#468BFF]/50 group-hover:border-[#468BFF]/30 bg-white/80 backdrop-blur-sm text-lg py-4 pl-12 pr-12 font-['DM_Sans']`}
                  placeholder="Enter Tavily API Key"
                />
                <button
                  type="button"
                  onClick={() => setShowKey(!showKey)}
                  className="absolute right-4 top-1/2 -translate-y-1/2 text-[#468BFF] hover:text-[#8FBCFA] focus:outline-none z-10"
                >
                  {showKey ? (
                    <EyeOff className="h-5 w-5" />
                  ) : (
                    <Eye className="h-5 w-5" />
                  )}
                </button>
              </div>
            </div>
            <div className="flex align-center">
              <p className="text-sm text-gray-600">
                Each enriched cell will use 1 API credit.
              </p>

              <a
                href="https://app.tavily.com" // <-- replace with your real URL
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-[#468BFF] hover:text-blue-300 px-1"
              >
                Don't have an API key?
              </a>
            </div>
          </div>
        )}
      </div>
    </motion.div>
  );
};

export default ApiKeyInput;
