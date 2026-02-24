import { X, Search } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface UnifiedQueriesPanelProps {
  queries: string[];
  isOpen: boolean;
  onClose: () => void;
}

export default function UnifiedQueriesPanel({
  queries,
  isOpen,
  onClose,
}: UnifiedQueriesPanelProps) {
  // Deduplicate queries
  const uniqueQueries = Array.from(new Set(queries));

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
            className="fixed right-0 top-0 h-full w-full max-w-4xl shadow-2xl z-50 overflow-hidden flex flex-col"
            style={{ background: "var(--color-black)" }}
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", damping: 30, stiffness: 300 }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between p-6 border-b" style={{
              borderColor: "var(--color-black-20)",
            }}>
              <div>
                <h2 className="text-2xl font-semibold" style={{ color: "var(--color-white)" }}>
                  Search Queries
                </h2>
                <p className="text-sm mt-1" style={{ color: "var(--color-white-50)" }}>
                  {uniqueQueries.length} unique quer{uniqueQueries.length !== 1 ? "ies" : "y"} executed during enrichment
                </p>
              </div>
              <button
                onClick={onClose}
                className="p-2 rounded-full transition-colors"
                style={{ color: "var(--color-white-50)" }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = "var(--color-black-20)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = "transparent";
                }}
                aria-label="Close"
              >
                <X className="w-6 h-6" />
              </button>
            </div>

            {/* Queries List */}
            <div className="flex-1 overflow-y-auto p-6">
              {uniqueQueries.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full" style={{ color: "var(--color-white-50)" }}>
                  <Search className="w-12 h-12 mb-4" />
                  <p>No queries available</p>
                </div>
              ) : (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {uniqueQueries.map((query, index) => (
                    <motion.div
                      key={`${query}-${index}`}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.02 }}
                      className="p-4 rounded-lg transition"
                      style={{
                        background: "var(--color-black-80)",
                        border: "1px solid var(--color-black-20)",
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.borderColor = "var(--color-primary-blue)";
                        e.currentTarget.style.background = "var(--color-black-60)";
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.borderColor = "var(--color-black-20)";
                        e.currentTarget.style.background = "var(--color-black-80)";
                      }}
                    >
                      <div className="flex items-start gap-3">
                        <Search className="w-4 h-4 flex-shrink-0 mt-0.5" style={{ color: "var(--color-primary-blue)" }} />
                        <div
                          className="text-sm leading-relaxed"
                          style={{ color: "var(--color-white)" }}
                        >
                          {query}
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              )}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}



