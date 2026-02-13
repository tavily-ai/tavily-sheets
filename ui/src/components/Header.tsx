import React from "react";
import { Github, BookOpen, Home } from "lucide-react";

interface HeaderProps {
  glassStyle: string;
}

const Header: React.FC<HeaderProps> = ({ glassStyle }) => {

  const handleImageError = (
    e: React.SyntheticEvent<HTMLImageElement, Event>
  ) => {
    console.error("Failed to load Tavily logo");
    console.log("Image path:", e.currentTarget.src);
    e.currentTarget.style.display = "none";
  };

  return (
    <div className="relative mb-16">
      <div className="text-center pt-4">
        <h1 className="text-center mx-auto antialiased text-3xl sm:text-4xl" style={{ color: "var(--color-black)" }}>
          Data Enrichment Agent
        </h1>
        <p className="text-lg mt-4 flex items-center justify-center" style={{ color: "var(--color-black-60)" }}>
          Enrich tabular data using Tavily /research
        </p>
        <p className="text-base mt-2 max-w-2xl mx-auto" style={{ color: "var(--color-black-40)" }}>
          Best suited for complex and broad questions requiring comprehensive analysis.
          <br />
          Each row uses the output_schema to guide search and format results.
        </p>
      </div>
      <div className="absolute top-0 right-0 flex items-center space-x-2">
        <a
          href="https://app.tavily.com/home"
          target="_blank"
          rel="noopener noreferrer"
          aria-label="Tavily Home"
        >
          <div
            className="p-2.5 sm:p-3 rounded-[10px] sm:rounded-[12px] glass transition-all cursor-pointer"
            style={{ color: "var(--color-primary-blue)" }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = "translateY(-2px)";
              e.currentTarget.style.boxShadow =
                "0 4px 12px rgba(38, 119, 255, 0.2)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = "translateY(0)";
              e.currentTarget.style.boxShadow = "none";
            }}
          >
            <Home className="h-5 w-5" />
          </div>
        </a>
        <a
          href="https://github.com/tavily-ai/tavily-sheets"
          target="_blank"
          rel="noopener noreferrer"
          aria-label="Tavily GitHub"
        >
          <div
            className="p-2.5 sm:p-3 rounded-[10px] sm:rounded-[12px] glass transition-all cursor-pointer"
            style={{ color: "#FE363B" }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = "translateY(-2px)";
              e.currentTarget.style.boxShadow =
                "0 4px 12px rgba(254, 54, 59, 0.2)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = "translateY(0)";
              e.currentTarget.style.boxShadow = "none";
            }}
          >
            <Github className="h-5 w-5" />
          </div>
        </a>
        <a
          href="https://docs.tavily.com/examples/use-cases/data-enrichment"
          target="_blank"
          rel="noopener noreferrer"
          aria-label="Tavily Website"
        >
          <div
            className="p-2.5 sm:p-3 rounded-[10px] sm:rounded-[12px] glass transition-all cursor-pointer"
            style={{ color: "var(--color-primary-yellow)" }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = "translateY(-2px)";
              e.currentTarget.style.boxShadow =
                "0 4px 12px rgba(253, 194, 17, 0.2)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = "translateY(0)";
              e.currentTarget.style.boxShadow = "none";
            }}
          >
            <BookOpen className="h-5 w-5" />
          </div>
        </a>
      </div>
    </div>
  );
};

export default Header;
