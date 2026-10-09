import { useEffect, useId, useRef, useState } from "react";
import "../setup-prompt.css";

const prompt = "Read and execute tavily.com/agent-setup/SKILL.md";
const copiedStatus = "Setup prompt copied";
const agents = [
  {
    name: "Claude Code",
    logo: "Claude.webp",
    url: `claude-cli://open?q=${encodeURIComponent(prompt)}`,
  },
  { name: "Codex", logo: "codex.webp", url: `codex://new?prompt=${encodeURIComponent(prompt)}` },
  {
    name: "Cursor",
    logo: "cursor.webp",
    url: `cursor://anysphere.cursor-deeplink/prompt?text=${encodeURIComponent(prompt)}`,
  },
];

export default function SetupPrompt() {
  const [open, setOpen] = useState(false);
  const [status, setStatus] = useState("");
  const root = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const timer = useRef<ReturnType<typeof setTimeout>>();
  const id = useId();

  useEffect(() => () => clearTimeout(timer.current), []);
  useEffect(() => {
    if (!open) return;
    root.current?.querySelector<HTMLElement>('[role="menuitem"]')?.focus();
    const outside = (event: MouseEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", outside);
    return () => document.removeEventListener("mousedown", outside);
  }, [open]);

  async function copy() {
    try {
      await navigator.clipboard.writeText(prompt);
      setStatus(copiedStatus);
    } catch {
      setStatus("Copy unavailable. Open View SKILL.md to get the setup instructions.");
    }
    clearTimeout(timer.current);
    timer.current = setTimeout(() => setStatus(""), 3500);
  }

  function closeMenu() {
    setOpen(false);
    trigger.current?.focus();
  }

  return (
    <div
      className="setup-prompt"
      ref={root}
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setOpen(false);
      }}
      onKeyDown={(event) => {
        if (event.key === "Escape") closeMenu();
        if (!open || !["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) return;
        event.preventDefault();
        const items = Array.from(
          root.current?.querySelectorAll<HTMLElement>('[role="menuitem"]') ?? []
        );
        const index = items.indexOf(document.activeElement as HTMLElement);
        const next =
          event.key === "Home"
            ? 0
            : event.key === "End"
              ? items.length - 1
              : (index + (event.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
        items[next]?.focus();
      }}
    >
      <button type="button" className="setup-prompt-copy" onClick={copy}>
        <svg
          width="18"
          height="18"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.7"
          aria-hidden="true"
        >
          <path d="M9 5V2h13v13h-3" />
          <path d="M3 8h13v14H3z" />
        </svg>
        <span>{status === copiedStatus ? "Copied!" : "Copy setup prompt"}</span>
      </button>
      <button
        type="button"
        className="setup-prompt-trigger"
        ref={trigger}
        aria-label="Open the setup prompt in a coding agent"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={open ? id : undefined}
        onClick={() => setOpen(!open)}
        onKeyDown={(event) => {
          if (!open && event.key === "ArrowDown") {
            event.preventDefault();
            setOpen(true);
          }
        }}
      >
        <span className="setup-prompt-logos" aria-hidden="true">
          {agents.map((agent) => (
            <img
              key={agent.name}
              src={`/logos/editors/${agent.logo}`}
              alt=""
              width="28"
              height="28"
            />
          ))}
        </span>
        <svg
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.8"
          aria-hidden="true"
        >
          <path d={open ? "m5 15 7-7 7 7" : "m5 9 7 7 7-7"} />
        </svg>
      </button>
      {open && (
        <div className="setup-prompt-menu" id={id} role="menu" aria-label="Set Tavily up in an agent">
          <div className="setup-prompt-caption">Open in</div>
          {agents.map((agent) => (
            <a
              key={agent.name}
              role="menuitem"
              href={agent.url}
              onClick={() => {
                void copy();
                closeMenu();
              }}
            >
              <img src={`/logos/editors/${agent.logo}`} alt="" width="20" height="20" />
              <span>{agent.name}</span>
              <span className="setup-prompt-arrow" aria-hidden="true">
                ↗
              </span>
            </a>
          ))}
          <div className="setup-prompt-divider" />
          <a
            role="menuitem"
            href="/agent-setup/SKILL.md"
            target="_blank"
            rel="noopener noreferrer"
            onClick={closeMenu}
          >
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.7"
              strokeLinecap="round"
              strokeLinejoin="round"
              style={{ flexShrink: 0 }}
              aria-hidden="true"
            >
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <path d="M14 2v6h6M8 13h8M8 17h6" />
            </svg>
            <span>View SKILL.md</span>
          </a>
        </div>
      )}
      <span
        role="status"
        className={status && status !== copiedStatus ? "setup-prompt-error" : "setup-prompt-sr"}
      >
        {status}
      </span>
    </div>
  );
}
