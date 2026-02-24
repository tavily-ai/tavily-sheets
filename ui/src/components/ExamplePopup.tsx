import React, { useState } from "react";
import { Sparkles, FlaskConical, Dna, TrendingUp, LucideIcon } from "lucide-react";
import { SpreadsheetData } from "../types";

// Sample companies for examples
const EXAMPLE_DATA: Array<{
  name: string;
  icon: LucideIcon;
  data: SpreadsheetData;
}> = [
  {
    name: "Drug Trials / Biomedical Research",
    icon: FlaskConical,
    data: {
      headers: ["Trial Name", "Drug / Intervention", "Indication", "Phase", "Sponsor / Organization", "Mechanism of Action", "Primary Endpoints", "Trial Status", "Key Results Summary", "Safety Signals", "Regulatory Notes", "Biomarker / Stratification Criteria"],
      rows: [
        [
          { value: "KEYNOTE-671 (Pembrolizumab in NSCLC)" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
        ],
        [
          { value: "CARTITUDE-1 (Cilta-cel for Multiple Myeloma)" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
        ],
        [
          { value: "DESTINY-Breast04" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
        ],
        [
          { value: "EMPA-REG OUTCOME" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
        ],
        [
          { value: "DAPA-HF" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
        ],
        [
          { value: "CheckMate-577" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
        ],
        [
          { value: "MONARCH-E" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
        ],
        [
          { value: "COV-BARRIER" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
        ],
        [
          { value: "IMpower010" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
        ],
        [
          { value: "LEADER Trial (Liraglutide)" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
        ],
      ],
    },
  },
  {
    name: "Biomedical / Scientific Landscape",
    icon: Dna,
    data: {
      headers: ["Entity Name (Gene / Protein / Pathway)", "Associated Conditions", "Therapeutic Area", "Key Findings", "Evidence Strength", "Recent Publications", "Active Trials", "Open Questions", "Research Momentum", "Source Citations"],
      rows: [
        [
          { value: "KRAS G12C" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
        ],
        [
          { value: "PD-1 / PD-L1 Pathway" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
        ],
        [
          { value: "Amyloid-β" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
        ],
        [
          { value: "Tau Protein" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
        ],
        [
          { value: "BRCA1 / BRCA2" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
        ],
        [
          { value: "IL-6" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
        ],
        [
          { value: "EGFR" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
        ],
        [
          { value: "APOE ε4" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
        ],
        [
          { value: "JAK-STAT Pathway" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
        ],
        [
          { value: "mRNA Vaccine Platforms" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
        ],
      ],
    },
  },
  {
    name: "Financial / Market Analysis",
    icon: TrendingUp,
    data: {
      headers: ["Asset / Company", "Sector / Market", "Key Drivers", "Recent Events", "Bull Case Summary", "Bear Case Summary", "Risk Factors", "Correlation Profile", "Volatility Regime", "Forward-Looking Indicators"],
      rows: [
        [
          { value: "NVIDIA (NVDA)" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
        ],
        [
          { value: "S&P 500 Semiconductors Index" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
        ],
        [
          { value: "US Treasury 10Y Yield" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
        ],
        [
          { value: "Bitcoin (BTC)" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
        ],
        [
          { value: "Crude Oil (WTI)" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
        ],
        [
          { value: "EUR/USD" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
        ],
        [
          { value: "AI Infrastructure Market" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
        ],
        [
          { value: "Electric Vehicle Supply Chain" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
        ],
        [
          { value: "US Regional Banks" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
        ],
        [
          { value: "China Semiconductor Export Controls" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
          { value: "" },
        ],
      ],
    },
  },
];

interface ExamplePopupProps {
  visible: boolean;
  onExampleSelect: React.Dispatch<React.SetStateAction<SpreadsheetData>>;
}

// Example Popup Component
const ExamplePopup: React.FC<ExamplePopupProps> = ({
  visible,
  onExampleSelect,
}) => {
  const [selectedExample, setSelectedExample] = useState(0);

  if (!visible) return null;

  return (
    <div className="mb-4">
      <div className="flex items-center gap-2 mb-3">
        <Sparkles
          className="w-4 h-4"
          style={{ color: "var(--color-black-40)" }}
        />
        <span
          className="text-sm font-medium"
          style={{ color: "var(--color-black-60)" }}
        >
          Try an example
        </span>
      </div>

      {/* Example chips */}
      <div className="flex flex-wrap gap-2">
        {EXAMPLE_DATA.map((example, idx) => {
          const isSelected = selectedExample === idx;
          const IconComponent = example.icon;

          return (
            <button
              key={idx}
              onClick={() => {
                setSelectedExample(idx);
                onExampleSelect(example.data);
              }}
              className="px-3 py-2 rounded-xl text-sm transition-all inline-flex items-center"
              style={{
                background: "var(--color-black-5)",
                color: "var(--color-black-60)",
                border: `1px solid ${isSelected ? "var(--color-black-60)" : "transparent"}`,
              }}
            >
              <IconComponent
                className="w-3.5 h-3.5 mr-1.5 inline-block"
                style={{
                  color: isSelected
                    ? "var(--color-black-60)"
                    : "var(--color-black-40)",
                }}
              />
              {example.name}
            </button>
          );
        })}
      </div>
    </div>
  );
};

export default ExamplePopup;
