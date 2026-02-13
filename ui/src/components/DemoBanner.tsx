import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FlaskConical } from "lucide-react";

const DemoBanner = () => {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="fixed top-0 left-0 right-0 z-50 flex justify-center"
    >
      <div
        className="relative flex items-center gap-1.5 py-1 px-3 text-xs font-medium cursor-default"
        style={{
          background: "rgba(253, 194, 17, 0.9)",
          color: "#78350F",
          borderBottomLeftRadius: "6px",
          borderBottomRightRadius: "6px",
        }}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      >
        <FlaskConical size={12} />
        <span>Experimental Demo</span>

        <AnimatePresence>
          {isHovered && (
            <motion.div
              initial={{ opacity: 0, y: -5 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -5 }}
              transition={{ duration: 0.2 }}
              className="absolute top-full mt-2 left-1/2 -translate-x-1/2 w-72 p-3 text-xs rounded-lg shadow-lg"
              style={{
                background: "rgba(255, 255, 255, 0.95)",
                backdropFilter: "blur(10px)",
                border: "1px solid rgba(253, 194, 17, 0.4)",
                color: "var(--color-black)",
              }}
            >
              <p className="font-medium mb-1">This is a beta version</p>
              <p style={{ color: "var(--color-black-60)" }}>
                This feature is experimental and may produce unexpected results.
                It is not intended for production use.
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
};

export default DemoBanner;
