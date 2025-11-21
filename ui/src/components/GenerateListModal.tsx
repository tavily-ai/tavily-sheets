import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Loader2, Sparkles, Info, AlertCircle, Zap, CheckCircle } from "lucide-react";
import { SpreadsheetData } from "../types";

interface GenerateListModalProps {
  isOpen: boolean;
  onClose: () => void;
  onGenerate: (data: SpreadsheetData) => void;
  apiKey: string;
  checkApiKey: () => boolean | 0 | undefined;
  setToast: (toast: { message?: string; type?: "success" | "error" | "info"; isShowing?: boolean }) => void;
}

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const MAX_PROMPT_LENGTH = 500;

const GenerateListModal: React.FC<GenerateListModalProps> = ({
  isOpen,
  onClose,
  onGenerate,
  apiKey,
  checkApiKey,
  setToast,
}) => {
  const [prompt, setPrompt] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedExample, setSelectedExample] = useState<number | null>(null);

  // Reset state when modal opens/closes
  useEffect(() => {
    if (!isOpen) {
      setPrompt("");
      setError(null);
      setSelectedExample(null);
    }
  }, [isOpen]);

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      setError("Please enter a prompt");
      return;
    }

    if (!checkApiKey()) {
      setError("Please set a valid API Key");
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_URL}/api/generate-list`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: apiKey,
        },
        body: JSON.stringify({ prompt: prompt.trim() }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      
      if (!result.data) {
        throw new Error("Invalid response format from server");
      }

      // Handle new backend response structure
      if (result.data && result.data.headers && result.data.rows) {
        onGenerate({
          headers: result.data.headers,
          rows: result.data.rows,
        });
      } else if (Array.isArray(result.data)) {
        // fallback for old response
        const newData: SpreadsheetData = {
          headers: [
            { name: "Company/Entity" },
            { name: "Website" },
            { name: "Description" },
            { name: "Category" },
          ],
          rows: result.data.map((item: string) => [
            { value: item },
            { value: "" },
            { value: "" },
            { value: "" },
          ]),
        };
        onGenerate(newData);
      } else {
        throw new Error("Unexpected response format from server");
      }

      const itemCount = result.data.rows?.length || result.data.length || 0;
      setToast({
        message: `✨ Successfully generated ${itemCount} items!`,
        type: "success",
        isShowing: true,
      });
      onClose();
    } catch (error) {
      console.error("Error generating list:", error);
      setError(error instanceof Error ? error.message : "Failed to generate list. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      handleGenerate();
    }
  };

  const handlePromptChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const value = e.target.value;
    if (value.length <= MAX_PROMPT_LENGTH) {
      setPrompt(value);
      setError(null);
    }
  };

  const handleExampleClick = (example: string, index: number) => {
    setPrompt(example);
    setSelectedExample(index);
    setError(null);
    // Reset selection after animation
    setTimeout(() => setSelectedExample(null), 1000);
  };

  const isPromptValid = prompt.trim().length > 0 && prompt.trim().length <= MAX_PROMPT_LENGTH;

  const examples = [
    "Top 10 SaaS companies in project management",
    "Leading AI startups in 2024",
    "Main competitors to Figma in design tools",
    "Best productivity apps for remote teams",
    "Top e-commerce platforms for small businesses"
  ];

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
        >
          <motion.div
            className="bg-white rounded-2xl shadow-2xl w-full max-w-2xl mx-4 overflow-hidden border border-gray-100"
            initial={{ scale: 0.9, opacity: 0, y: 20 }}
            animate={{ scale: 1, opacity: 1, y: 0 }}
            exit={{ scale: 0.9, opacity: 0, y: 20 }}
            transition={{ type: "spring", damping: 25, stiffness: 300 }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between p-6 border-b border-gray-200 bg-gradient-to-r from-blue-50 via-indigo-50 to-purple-50">
              <div className="flex items-center gap-3">
                <motion.div 
                  className="p-2 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-xl text-white"
                  animate={{ rotate: isLoading ? 360 : 0 }}
                  transition={{ duration: 2, repeat: isLoading ? Infinity : 0, ease: "linear" }}
                >
                  <Sparkles className="w-5 h-5" />
                </motion.div>
                <div>
                  <h2 className="text-xl font-bold text-gray-900">
                    Generate List with AI
                  </h2>
                  <p className="text-sm text-gray-600 mt-1">
                    Describe what kind of list you want to generate
                  </p>
                </div>
              </div>
              <button
                onClick={onClose}
                className="text-gray-400 hover:text-gray-600 transition-colors p-2 rounded-full hover:bg-gray-100"
                disabled={isLoading}
              >
                <X size={20} />
              </button>
            </div>

            {/* Content */}
            <div className="p-6">
              <div className="mb-6">
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  Prompt
                </label>
                <div className="relative">
                  <textarea
                    value={prompt}
                    onChange={handlePromptChange}
                    onKeyDown={handleKeyDown}
                    placeholder="e.g., Who are the main competitors to Figma? List the top 10 SaaS companies in the project management space. What are the leading AI startups in 2024?"
                    className={`w-full h-32 p-4 border rounded-xl resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all ${
                      error ? 'border-red-300' : 'border-gray-300'
                    }`}
                    disabled={isLoading}
                  />
                  <div className="flex justify-between items-center mt-2">
                    <div className="flex items-center gap-2">
                      <Info className="w-4 h-4 text-gray-400" />
                      <span className="text-xs text-gray-500">
                        Press Cmd/Ctrl + Enter to generate
                      </span>
                    </div>
                    <span className={`text-xs font-medium ${
                      prompt.length > MAX_PROMPT_LENGTH * 0.9 
                        ? 'text-orange-500' 
                        : 'text-gray-400'
                    }`}>
                      {prompt.length}/{MAX_PROMPT_LENGTH}
                    </span>
                  </div>
                </div>
                
                {error && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex items-center gap-2 mt-2 p-3 bg-red-50 border border-red-200 rounded-lg"
                  >
                    <AlertCircle className="w-4 h-4 text-red-500 flex-shrink-0" />
                    <span className="text-sm text-red-700">{error}</span>
                  </motion.div>
                )}
              </div>

              <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-xl p-4 mb-6">
                <h3 className="text-sm font-semibold text-blue-900 mb-3 flex items-center gap-2">
                  <Zap className="w-4 h-4" />
                  What happens next?
                </h3>
                <ul className="text-sm text-blue-800 space-y-2">
                  <li className="flex items-start gap-2">
                    <span className="w-1.5 h-1.5 bg-blue-400 rounded-full mt-2 flex-shrink-0"></span>
                    <span>AI will analyze your prompt and generate a relevant list</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="w-1.5 h-1.5 bg-blue-400 rounded-full mt-2 flex-shrink-0"></span>
                    <span>The list will be populated in your spreadsheet with multiple columns</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="w-1.5 h-1.5 bg-blue-400 rounded-full mt-2 flex-shrink-0"></span>
                    <span>You can then use "Enrich" features to add more detailed data</span>
                  </li>
                </ul>
              </div>

              {/* Example Prompts */}
              <div className="bg-gray-50 rounded-xl p-4">
                <h4 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-gray-500" />
                  Example Prompts:
                </h4>
                <div className="grid grid-cols-1 gap-2">
                  {examples.map((example, index) => (
                    <motion.button
                      key={index}
                      onClick={() => handleExampleClick(example, index)}
                      disabled={isLoading}
                      className={`text-left p-3 text-sm text-gray-600 hover:bg-gray-100 rounded-lg transition-all duration-200 disabled:opacity-50 ${
                        selectedExample === index ? 'bg-green-100 border border-green-300' : ''
                      }`}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      <div className="flex items-center justify-between">
                        <span>"{example}"</span>
                        {selectedExample === index && (
                          <motion.div
                            initial={{ scale: 0 }}
                            animate={{ scale: 1 }}
                            className="text-green-600"
                          >
                            <CheckCircle className="w-4 h-4" />
                          </motion.div>
                        )}
                      </div>
                    </motion.button>
                  ))}
                </div>
              </div>
            </div>

            {/* Footer */}
            <div className="flex items-center justify-end gap-3 p-6 border-t border-gray-200 bg-gradient-to-r from-gray-50 to-gray-100">
              <button
                onClick={onClose}
                className="px-4 py-2.5 text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors font-medium"
                disabled={isLoading}
              >
                Cancel
              </button>
              <motion.button
                onClick={handleGenerate}
                disabled={isLoading || !isPromptValid}
                className="px-6 py-2.5 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-lg hover:from-blue-700 hover:to-indigo-700 disabled:from-gray-400 disabled:to-gray-400 disabled:cursor-not-allowed transition-all font-medium flex items-center gap-2 shadow-sm"
                whileHover={{ scale: isPromptValid ? 1.02 : 1 }}
                whileTap={{ scale: isPromptValid ? 0.98 : 1 }}
              >
                {isLoading ? (
                  <>
                    <Loader2 size={16} className="animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Sparkles size={16} />
                    Generate List
                  </>
                )}
              </motion.button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default GenerateListModal; 