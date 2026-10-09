import React from "react";
import SetupPrompt from "./SetupPrompt";

const Header: React.FC = () => {
  return (
    <header className="w-full">
      <nav
        className="setup-header min-h-[84px] sm:min-h-[104px]"
        aria-label="Main navigation"
      >
        <a
          href="https://tavily.com"
          target="_blank"
          rel="noopener noreferrer"
          aria-label="Tavily home"
          className="brand-link"
        >
          <img
            src="/tavily-by-nebius.svg"
            alt="Tavily, by Nebius"
            className="brand-logo"
          />
        </a>

        <SetupPrompt />

        <div className="flex items-center gap-1 sm:gap-2">
          <a
            href="https://github.com/tavily-ai/tavily-sheets"
            target="_blank"
            rel="noopener noreferrer"
            aria-label="View this demo on GitHub"
            className="nav-link"
          >
            <img src="/github-icon.png" alt="" className="github-logo" />
          </a>
        </div>
      </nav>

      <div className="text-center pt-2 pb-2">
        <h1 className="mx-auto max-w-[760px] text-balance">
          Data Enrichment Agent
        </h1>
        <p
          className="mx-auto mt-3 max-w-[760px] text-[15px] font-medium text-balance"
          style={{ color: "var(--color-black)" }}
        >
          Enrich tabular data using Tavily /research
        </p>
        <p
          className="mx-auto mt-3 max-w-[640px] text-[13px] leading-[1.6]"
          style={{ color: "var(--color-muted)" }}
        >
          Best suited for complex and broad questions requiring comprehensive
          analysis.
          <br />
          Each row uses the output_schema to guide search and format results.
        </p>
      </div>
    </header>
  );
};

export default Header;
