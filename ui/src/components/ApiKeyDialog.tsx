import React, { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import {
  ArrowUpRight,
  Eye,
  EyeOff,
  KeyRound,
  Lock,
  Trash2,
  X,
} from "lucide-react";

const KEY_PATTERN = /^tvly-[A-Za-z0-9_-]{20,}$/;

const normalize = (value: string): string => value.replace(/\s+/g, "").trim();

const maskKey = (value: string): string =>
  value.length <= 12 ? value : `${value.slice(0, 9)}...${value.slice(-4)}`;

interface ApiKeyStatusProps {
  isValid: boolean;
  apiKey: string;
  onOpen: () => void;
}

/** Compact, always-visible entry point into the key dialog. */
export const ApiKeyStatus: React.FC<ApiKeyStatusProps> = ({
  isValid,
  apiKey,
  onOpen,
}) => (
  <button type="button" className="key-status" onClick={onOpen}>
    <span className={`key-dot${isValid ? " connected" : ""}`} />
    <span>
      {isValid ? (
        <>
          API key connected
          <span style={{ color: "var(--color-muted)", marginLeft: 8 }}>
            {maskKey(apiKey)}
          </span>
        </>
      ) : (
        "Add your Tavily API key"
      )}
    </span>
  </button>
);

interface ApiKeyDialogProps {
  apiKey: string;
  setApiKey: (key: string) => void;
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
}

const ApiKeyDialog: React.FC<ApiKeyDialogProps> = ({
  apiKey,
  setApiKey,
  isOpen,
  setIsOpen,
}) => {
  const [draft, setDraft] = useState(apiKey);
  const [showKey, setShowKey] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const openerRef = useRef<Element | null>(null);
  const reduceMotion = useReducedMotion();

  const close = useCallback(() => {
    setIsOpen(false);
  }, [setIsOpen]);

  // Reset the draft from the saved key each time the dialog opens so a
  // cancelled edit never changes what the app is using.
  useEffect(() => {
    if (!isOpen) return;
    openerRef.current = document.activeElement;
    setDraft(apiKey);
    setShowKey(false);
    setError(null);
    const frame = requestAnimationFrame(() => inputRef.current?.focus());
    return () => cancelAnimationFrame(frame);
  }, [isOpen, apiKey]);

  useEffect(() => {
    if (!isOpen) {
      const opener = openerRef.current;
      if (opener instanceof HTMLElement) opener.focus();
      return;
    }

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.stopPropagation();
        close();
      }
    };

    document.addEventListener("keydown", onKeyDown);
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.body.style.overflow = previousOverflow;
    };
  }, [isOpen, close]);

  const save = () => {
    const value = normalize(draft);

    if (!value) {
      setError("Enter your Tavily API key to continue.");
      inputRef.current?.focus();
      return;
    }

    if (!KEY_PATTERN.test(value)) {
      setError(
        "That does not look like a Tavily key. Keys start with tvly- followed by at least 20 characters."
      );
      inputRef.current?.focus();
      return;
    }

    setApiKey(value);
    close();
  };

  const remove = () => {
    setApiKey("");
    setDraft("");
    setError(null);
    inputRef.current?.focus();
  };

  const motionProps = reduceMotion
    ? {
        initial: { opacity: 0 },
        animate: { opacity: 1 },
        exit: { opacity: 0 },
        transition: { duration: 0.15 },
      }
    : {
        initial: { opacity: 0, y: 12, scale: 0.98 },
        animate: { opacity: 1, y: 0, scale: 1 },
        exit: { opacity: 0, y: 8, scale: 0.98 },
        transition: { duration: 0.2, ease: [0.22, 1, 0.36, 1] as const },
      };

  return createPortal(
    <AnimatePresence>
      {isOpen && (
        <motion.div
          className="key-backdrop"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.18 }}
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) close();
          }}
        >
          <motion.div
            className="key-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="api-key-dialog-title"
            aria-describedby="api-key-dialog-description"
            {...motionProps}
          >
            <div className="flex items-start gap-4">
              <div className="key-dialog-icon">
                <KeyRound size={17} />
              </div>
              <div className="flex-1 min-w-0">
                <h2 id="api-key-dialog-title">Connect your Tavily API key</h2>
                <p
                  id="api-key-dialog-description"
                  className="mt-2 text-[12px] leading-[1.6]"
                  style={{ color: "var(--color-muted)" }}
                >
                  The key authorizes the research calls behind each enriched
                  cell.
                </p>
              </div>
              <button
                type="button"
                className="key-ghost-button"
                onClick={close}
                aria-label="Close dialog"
              >
                <X size={16} />
              </button>
            </div>

            <form
              className="mt-6"
              onSubmit={(event) => {
                event.preventDefault();
                save();
              }}
            >
              <label
                htmlFor="api-key-input"
                className="block text-[12px] font-medium mb-2.5"
              >
                API key
              </label>
              <div className={`key-field${error ? " invalid" : ""}`}>
                <input
                  id="api-key-input"
                  ref={inputRef}
                  type={showKey ? "text" : "password"}
                  value={draft}
                  onChange={(event) => {
                    setDraft(event.target.value);
                    if (error) setError(null);
                  }}
                  placeholder="tvly-..."
                  autoComplete="off"
                  autoCorrect="off"
                  autoCapitalize="off"
                  spellCheck="false"
                  aria-invalid={Boolean(error)}
                  aria-describedby={error ? "api-key-error" : undefined}
                />
                <button
                  type="button"
                  className="key-ghost-button"
                  onClick={() => setShowKey((shown) => !shown)}
                  aria-label={showKey ? "Hide API key" : "Show API key"}
                >
                  {showKey ? <EyeOff size={15} /> : <Eye size={15} />}
                </button>
              </div>

              {error && (
                <p id="api-key-error" className="key-error mt-3" role="alert">
                  {error}
                </p>
              )}

              <p className="mt-3 text-[11px]" style={{ color: "var(--color-muted)" }}>
                Do not have one yet?{" "}
                <a
                  href="https://app.tavily.com"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-0.5 hover:underline"
                  style={{ color: "var(--color-primary-blue)" }}
                >
                  Create a free key at app.tavily.com
                  <ArrowUpRight size={12} />
                </a>
              </p>

              <div className="key-note mt-5">
                <Lock size={13} style={{ flexShrink: 0, marginTop: 1 }} />
                <span>
                  Your key is held in this browser tab for the session only. It
                  is sent with your research requests and never stored on our
                  servers.
                </span>
              </div>

              <div className="mt-6 flex items-center justify-between gap-3">
                {apiKey ? (
                  <button
                    type="button"
                    className="key-text-button inline-flex items-center gap-2"
                    onClick={remove}
                  >
                    <Trash2 size={13} />
                    Remove key
                  </button>
                ) : (
                  <span />
                )}

                <div className="flex items-center gap-3">
                  <button
                    type="button"
                    className="key-text-button"
                    onClick={close}
                  >
                    Cancel
                  </button>
                  <button type="submit" className="btn-pill">
                    Save key
                  </button>
                </div>
              </div>
            </form>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>,
    document.body
  );
};

export default ApiKeyDialog;
