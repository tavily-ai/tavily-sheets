import { X, Globe } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { Source } from "./Tooltip";

interface UnifiedSourcesPanelProps {
  sources: Source[];
  isOpen: boolean;
  onClose: () => void;
}

export default function UnifiedSourcesPanel({
  sources,
  isOpen,
  onClose,
}: UnifiedSourcesPanelProps) {
  // Deduplicate sources by URL
  const uniqueSources = Array.from(
    new Map(sources.map((source) => [source.url, source])).values()
  );

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            className="fixed inset-0 z-50"
            style={{ background: "rgba(60, 58, 57, 0.5)" }}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />
          {/* Panel */}
          <motion.div
            className="fixed right-0 top-0 h-full w-full max-w-2xl shadow-2xl z-50 overflow-hidden flex flex-col glass-subtle"
            style={{ background: "var(--color-background)" }}
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", damping: 30, stiffness: 300 }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between p-6 border-b" style={{
              borderColor: "var(--color-black-10)",
              background: "var(--color-light-gray)"
            }}>
              <div>
                <h2 className="text-2xl font-semibold" style={{ color: "var(--color-black)" }}>
                  All Sources
                </h2>
                <p className="text-sm mt-1" style={{ color: "var(--color-black-60)" }}>
                  {uniqueSources.length} Unique Source{uniqueSources.length !== 1 ? "s" : ""}
                </p>
              </div>
              <button
                onClick={onClose}
                className="p-2 rounded-full transition-colors"
                style={{ color: "var(--color-black-60)" }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = "var(--color-black-10)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = "transparent";
                }}
                aria-label="Close"
              >
                <X className="w-6 h-6" />
              </button>
            </div>

            {/* Sources List */}
            <div className="flex-1 overflow-y-auto p-6">
              {uniqueSources.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full" style={{ color: "var(--color-black-40)" }}>
                  <Globe className="w-12 h-12 mb-4" />
                  <p>No sources available</p>
                </div>
              ) : (
                <>
                  <div className="space-y-2 max-h-[calc(100vh-300px)] overflow-y-auto">
                    {uniqueSources.map((source, index) => {
                      const { title, url, favicon } = source;

                      return (
                        <motion.a
                          key={url}
                          href={url}
                          target="_blank"
                          rel="noopener noreferrer"
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: index * 0.02 }}
                          className="block p-2.5 sm:p-3 rounded-[10px] sm:rounded-[12px] transition active:scale-[0.98]"
                          style={{
                            background: "var(--color-white)",
                            border: "1px solid var(--color-black-10)",
                          }}
                          onMouseEnter={(e) => {
                            e.currentTarget.style.borderColor = "var(--color-primary-blue)";
                            e.currentTarget.style.boxShadow = "0 2px 4px rgba(38, 119, 255, 0.1)";
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.borderColor = "var(--color-black-10)";
                            e.currentTarget.style.boxShadow = "none";
                          }}
                        >
                          <div className="flex items-center gap-3">
                            {favicon ? (
                              <img
                                src={favicon}
                                alt=""
                                className="w-5 h-5 flex-shrink-0 rounded"
                                onError={(e) => {
                                  e.currentTarget.style.display = "none";
                                }}
                              />
                            ) : (
                              <Globe className="w-5 h-5 flex-shrink-0" style={{ color: "var(--color-black-40)" }} />
                            )}
                            <div
                              className="text-sm font-medium line-clamp-1 flex-1"
                              style={{ color: "var(--color-primary-blue)" }}
                            >
                              {title || url}
                            </div>
                          </div>
                          <div
                            className="text-xs mt-1"
                            style={{ color: "var(--color-black-40)" }}
                          >
                            {new URL(url).hostname}
                          </div>
                        </motion.a>
                      );
                    })}
                  </div>
                </>
              )}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}


