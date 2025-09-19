import React, { useState } from "react";
import {
  Info,
  Sparkles,
  X,
  ChevronLeft,
  ChevronRight,
  Pencil,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface InfoPanelProps {
  glassStyle: string;
  onDismiss: () => void;
}

const InfoPanel: React.FC<InfoPanelProps> = ({ glassStyle, onDismiss }) => {
  const [currentPage, setCurrentPage] = useState(0);

  const pages = [
    {
      title: "How To Use",
      content: (
        <ol className="text-gray-600 text-sm list-decimal pl-0 space-y-1">
          <li className="flex items-center gap-2">
            <span className="w-6 text-center">1.</span> Add column headers for
            the information you want to enrich{" "}
            <Pencil className="text-blue-500 w-3 h-3 inline" />
          </li>
          <li className="flex items-center gap-2">
            <span className="w-6 text-center">2.</span> Fill in the first column
            with the key entities or items you want to enrich
          </li>
          <li className="flex items-center gap-2">
            <span className="w-6 text-center">3.</span> Click the{" "}
            <Sparkles className="text-blue-500 w-3 h-3 inline" /> in the
            top-left cell to enrich the entire table OR click the{" "}
            <Sparkles className="text-blue-500 w-3 h-3 inline" /> icon in a
            specific column header to enrich only that column
          </li>
        </ol>
      ),
    },
    {
      title: "About This Demo",
      content: (
        <p className="text-gray-600 text-sm leading-relaxed">
          This demo showcases how Sylke's AI can automatically enrich your
          spreadsheet data with real-time surgeon information.
        </p>
      ),
    },
    {
      title: "Usage",
      content: (
        <p className="text-gray-600 text-sm leading-relaxed">
          Each enriched cell will use 1 Tavily API credit.
        </p>
      ),
    },
  ];

  const nextPage = () => {
    setCurrentPage((prev) => (prev + 1) % pages.length);
  };

  const prevPage = () => {
    setCurrentPage((prev) => (prev - 1 + pages.length) % pages.length);
  };

  return (
    <motion.div
      className={`${glassStyle} mb-4 rounded-lg overflow-hidden shadow-sm `}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div className="px-2 relative">
        <div className="flex justify-between items-center mb-4">
          <div className="flex items-center">
            <div className="p-1.5 bg-blue-100 rounded-md flex-shrink-0 mr-2">
              <Info className="text-blue-600 w-3.5 h-3.5" />
            </div>
            <h3 className="text-md font-medium text-gray-800">
              {pages[currentPage].title}
            </h3>
          </div>
          <div className="flex items-center">
            <button
              onClick={prevPage}
              className="p-1 text-gray-400 hover:text-gray-600"
              aria-label="Previous page"
            >
              <ChevronLeft size={16} />
            </button>
            <span className="text-xs text-gray-500 mx-1">
              {currentPage + 1}/{pages.length}
            </span>
            <button
              onClick={nextPage}
              className="p-1 text-gray-400 hover:text-gray-600"
              aria-label="Next page"
            >
              <ChevronRight size={16} />
            </button>
            <button
              onClick={onDismiss}
              className="ml-2 p-1 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-full"
              aria-label="Dismiss"
            >
              <X size={14} />
            </button>
          </div>
        </div>

        <AnimatePresence mode="wait">
          <motion.div
            key={currentPage}
            initial={{ opacity: 0, x: 10 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -10 }}
            transition={{ duration: 0.2 }}
          >
            {pages[currentPage].content}
          </motion.div>
        </AnimatePresence>
      </div>
    </motion.div>
  );
};

export default InfoPanel;
