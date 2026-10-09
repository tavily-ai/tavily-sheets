import React, { useState } from "react";
import {
  Sparkles,
  FlaskConical,
  Dna,
  TrendingUp,
  Building2,
  Cpu,
  ShieldCheck,
  ShoppingCart,
  Plane,
  Leaf,
  Scale,
  GraduationCap,
  Newspaper,
  Factory,
  MapPin,
  Briefcase,
  Trophy,
  HeartHandshake,
  Film,
  Zap,
  LucideIcon,
} from "lucide-react";
import { SpreadsheetData } from "../types";

const createExampleData = (
  headers: string[],
  subjects: string[]
): SpreadsheetData => ({
  headers,
  rows: subjects.map((subject) => [
    { value: subject },
    ...headers.slice(1).map(() => ({ value: "" })),
  ]),
});

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
  {
    name: "Company Profiles",
    icon: Building2,
    data: createExampleData(
      [
        "Company",
        "Headquarters",
        "Founded",
        "Leadership",
        "Business Model",
        "Products",
        "Funding / Revenue",
        "Recent News",
      ],
      ["Anthropic", "Canva", "Databricks", "Figma", "Stripe"]
    ),
  },
  {
    name: "AI Tools / Platforms",
    icon: Cpu,
    data: createExampleData(
      [
        "Platform",
        "Primary Use Case",
        "Key Features",
        "Model Support",
        "Pricing",
        "Integrations",
        "Security / Compliance",
        "Best For",
      ],
      ["Hugging Face", "LangChain", "Pinecone", "Replicate", "Weights & Biases"]
    ),
  },
  {
    name: "Cybersecurity Incidents",
    icon: ShieldCheck,
    data: createExampleData(
      [
        "Incident / Organization",
        "Date Disclosed",
        "Attack Type",
        "Affected Systems",
        "Impact",
        "Threat Actor",
        "Response",
        "Current Status",
      ],
      [
        "Change Healthcare cyberattack",
        "MOVEit Transfer campaign",
        "MGM Resorts cyberattack",
        "SolarWinds supply-chain attack",
        "Colonial Pipeline ransomware attack",
      ]
    ),
  },
  {
    name: "Consumer Brand Research",
    icon: ShoppingCart,
    data: createExampleData(
      [
        "Brand",
        "Category",
        "Target Customer",
        "Price Positioning",
        "Key Products",
        "Distribution",
        "Competitors",
        "Recent Campaigns",
      ],
      ["Allbirds", "Glossier", "Liquid Death", "Oura", "Whoop"]
    ),
  },
  {
    name: "Travel Destinations",
    icon: Plane,
    data: createExampleData(
      [
        "Destination",
        "Best Time to Visit",
        "Top Attractions",
        "Typical Daily Budget",
        "Getting Around",
        "Weather",
        "Entry Requirements",
        "Travel Tips",
      ],
      ["Kyoto, Japan", "Lisbon, Portugal", "Mexico City, Mexico", "Reykjavik, Iceland", "Cape Town, South Africa"]
    ),
  },
  {
    name: "Climate / Sustainability",
    icon: Leaf,
    data: createExampleData(
      [
        "Company / Initiative",
        "Climate Commitments",
        "Target Year",
        "Emissions Progress",
        "Renewable Energy",
        "Reporting Framework",
        "Controversies",
        "Latest Update",
      ],
      ["Apple", "IKEA", "Microsoft", "Patagonia", "Ørsted"]
    ),
  },
  {
    name: "Legal / Regulatory Tracker",
    icon: Scale,
    data: createExampleData(
      [
        "Regulation / Case",
        "Jurisdiction",
        "Effective Date",
        "Scope",
        "Key Requirements",
        "Organizations Affected",
        "Penalties",
        "Latest Development",
      ],
      ["EU AI Act", "Digital Markets Act", "California Consumer Privacy Act", "SEC climate disclosure rules", "FTC noncompete rule"]
    ),
  },
  {
    name: "Universities / Programs",
    icon: GraduationCap,
    data: createExampleData(
      [
        "Program",
        "Institution",
        "Location",
        "Curriculum Focus",
        "Program Length",
        "Tuition",
        "Admissions Requirements",
        "Application Deadline",
      ],
      ["MIT MBA", "Stanford MS Computer Science", "Carnegie Mellon MS Robotics", "Oxford MSc AI", "Georgia Tech Online MSCS"]
    ),
  },
  {
    name: "News / Event Briefings",
    icon: Newspaper,
    data: createExampleData(
      [
        "Topic",
        "What Happened",
        "Date",
        "Key Organizations",
        "Why It Matters",
        "Market Reaction",
        "Open Questions",
        "Source Summary",
      ],
      ["AI chip export controls", "Commercial lunar missions", "Global shipping disruptions", "Major central bank rate decisions", "Quantum computing milestones"]
    ),
  },
  {
    name: "Supply Chain Intelligence",
    icon: Factory,
    data: createExampleData(
      [
        "Product / Material",
        "Major Producers",
        "Key Regions",
        "Supply Constraints",
        "Demand Drivers",
        "Price Trends",
        "Geopolitical Risks",
        "Alternatives",
      ],
      ["Advanced AI GPUs", "Lithium carbonate", "Rare earth magnets", "Semiconductor-grade neon", "Solar polysilicon"]
    ),
  },
  {
    name: "Real Estate / Location Intel",
    icon: MapPin,
    data: createExampleData(
      [
        "Market",
        "Median Home Price",
        "Rent Trends",
        "Population Growth",
        "Major Employers",
        "Development Pipeline",
        "Transit Access",
        "Market Outlook",
      ],
      ["Austin, Texas", "Charlotte, North Carolina", "Denver, Colorado", "Nashville, Tennessee", "Raleigh, North Carolina"]
    ),
  },
  {
    name: "Hiring / Talent Markets",
    icon: Briefcase,
    data: createExampleData(
      [
        "Role",
        "Typical Salary Range",
        "Top Hiring Markets",
        "In-Demand Skills",
        "Experience Level",
        "Remote Availability",
        "Hiring Companies",
        "Market Outlook",
      ],
      ["AI Research Engineer", "Cybersecurity Analyst", "Data Engineer", "Product Designer", "Solutions Architect"]
    ),
  },
  {
    name: "Sports Teams / Leagues",
    icon: Trophy,
    data: createExampleData(
      [
        "Team / League",
        "Location",
        "Recent Performance",
        "Key Players",
        "Head Coach",
        "Championships",
        "Venue",
        "Latest News",
      ],
      ["Arsenal FC", "Boston Celtics", "Kansas City Chiefs", "Los Angeles Dodgers", "McLaren Formula 1 Team"]
    ),
  },
  {
    name: "Nonprofits / Foundations",
    icon: HeartHandshake,
    data: createExampleData(
      [
        "Organization",
        "Mission",
        "Programs",
        "Regions Served",
        "Leadership",
        "Funding",
        "Impact Metrics",
        "Recent Initiatives",
      ],
      ["charity: water", "Gates Foundation", "Kiva", "The Nature Conservancy", "World Central Kitchen"]
    ),
  },
  {
    name: "Films / Streaming",
    icon: Film,
    data: createExampleData(
      [
        "Title",
        "Release Date",
        "Director / Creator",
        "Cast",
        "Platform / Distributor",
        "Critical Reception",
        "Awards",
        "Box Office / Viewership",
      ],
      ["Dune: Part Two", "Oppenheimer", "Severance", "Shōgun", "The Bear"]
    ),
  },
  {
    name: "Energy Projects",
    icon: Zap,
    data: createExampleData(
      [
        "Project",
        "Energy Type",
        "Location",
        "Capacity",
        "Developer",
        "Project Status",
        "Expected Completion",
        "Key Challenges",
      ],
      ["Dogger Bank Wind Farm", "Gemini Solar Project", "Hinkley Point C", "NEOM Green Hydrogen Project", "Vogtle Units 3 and 4"]
    ),
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
