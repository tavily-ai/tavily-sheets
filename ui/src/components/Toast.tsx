import { useEffect } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { X } from "lucide-react";

export type ToastProps = {
  message?: string;
  type?: "success" | "error" | "info";
  duration?: number;
  onClose: () => void;
};

const Toast: React.FC<ToastProps> = ({
  message,
  type = "success",
  duration = 3000,
  onClose,
}) => {
  useEffect(() => {
    const timer = setTimeout(onClose, duration);
    return () => clearTimeout(timer);
  }, [onClose, duration]);

  const typeStyles: Record<string, { bg: string; text: string; border: string }> = {
    success: {
      bg: "rgba(38, 119, 255, 0.1)",
      text: "var(--color-primary-blue)",
      border: "rgba(38, 119, 255, 0.2)"
    },
    error: {
      bg: "rgba(255, 39, 45, 0.1)",
      text: "#ff272d",
      border: "rgba(255, 39, 45, 0.2)"
    },
    info: {
      bg: "rgba(38, 119, 255, 0.1)",
      text: "var(--color-primary-blue)",
      border: "rgba(38, 119, 255, 0.2)"
    },
  };

  const styles = typeStyles[type];

  return (
    <AnimatePresence>
      <motion.div
        initial={{ x: 50, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        exit={{ x: 100, opacity: 0 }}
        transition={{ type: "spring", stiffness: 300, damping: 25 }}
        className={`fixed px-4 py-2 rounded-lg shadow-md flex items-center gap-3 z-50 glass ${
          type === "error" ? "top-5 right-5" : "bottom-5 right-5"
        }`}
        style={{
          background: styles.bg,
          color: styles.text,
          border: `1px solid ${styles.border}`
        }}
      >
        <span>{message}</span>
        <button onClick={onClose} style={{ color: styles.text }}>
          <X size={18} />
        </button>
      </motion.div>
    </AnimatePresence>
  );
};

export default Toast;
