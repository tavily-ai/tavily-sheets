import { useRef, useState } from "react";
import { Globe, Info } from "lucide-react";

export type Source = {
  title: string;
  url: string;
  favicon?: string | null; // Optional - may be null or undefined
};

type SourcesTooltipProps = {
  sources: Source[];
  open: boolean;
  setOpen: (value: boolean) => void;
};

export default function SourcesTooltip({ sources }: SourcesTooltipProps) {
  const tooltipRef = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);

  return (
    <div
      className="relative inline-block flex align-center"
      ref={tooltipRef}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      <button
        onClick={(e) => {
          e.stopPropagation();
          setOpen(!open);
        }}
        className="focus:outline-none"
      >
        <Info className="w-5 h-5 cursor-pointer" style={{ color: "var(--color-primary-blue)" }} />
      </button>
      {open && (
        <div
          className="absolute left-0 mt-5 w-64 shadow-lg rounded-lg border p-3 z-50 glass"
          style={{
            background: "var(--color-background)",
            borderColor: "var(--color-black-10)"
          }}
          onClick={(e) => e.stopPropagation()}
        >
          <h4 className="text-sm font-semibold mb-2" style={{ color: "var(--color-black)" }}>Sources</h4>
          <ul className="max-h-60 overflow-y-auto">
            {sources.map((source, index) => {
              const { title, url, favicon } = source;

              return (
                <li key={index} className="flex items-center space-x-2 py-1">
                  {favicon ? (
                    <img
                      src={favicon}
                      alt="Favicon"
                      className="w-4 h-4"
                    />
                  ) : (
                    <Globe className="w-4 h-4" style={{ color: "var(--color-black-40)" }} />
                  )}
                  <a
                    href={url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm truncate w-full hover:underline"
                    style={{ 
                      display: "inline-block", 
                      maxWidth: "85%",
                      color: "var(--color-primary-blue)"
                    }}
                  >
                    {title}
                  </a>
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </div>
  );
}
