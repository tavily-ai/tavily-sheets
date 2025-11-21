import { Source } from "./components/Tooltip";

export type GlassStyle = {
  base: string;
  card: string;
  input: string;
};

export type CellData = {
  value: string;
  sources?: Source[];
  enriched?: boolean;
  loading?: boolean;
};

// Configuration for each column, supporting agent chains
export type ColumnConfig = {
  name: string; // The column header name
  enrichmentType?: "predefined" | "ai_agent"; // Type of enrichment for this column
  mappedColumnIndex?: number; // Index of the column to use as input (for agent chains)
  customPrompt?: string; // Custom prompt/question for AI agent
};

// Spreadsheet data structure
export type SpreadsheetData = {
  headers: ColumnConfig[]; // Column configurations (was: string[])
  rows: CellData[][];
};

export type Position = {
  row: number;
  col: number;
};
