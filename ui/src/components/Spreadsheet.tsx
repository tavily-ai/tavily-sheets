import React, {
  useState,
  useRef,
  useEffect,
  SetStateAction,
  Dispatch,
} from "react";
import { SpreadsheetData, Position } from "../types";
import { Sparkles, Trash2, Pencil, Plus, Search, Loader2, CheckCircle2, Zap, Upload, Download } from "lucide-react";
import { motion } from "framer-motion";
import { ToastDetail } from "../App";
import ExamplePopup from "./ExamplePopup";
import UnifiedSourcesPanel from "./UnifiedSourcesPanel";
import ApiKeyInput from "./ApiKeyInput";
import { Source } from "./Tooltip";
import { exportToCSV } from "../utils";
import Papa from "papaparse";
import * as XLSX from "xlsx";

interface SpreadsheetProps {
  data: SpreadsheetData;
  setData: Dispatch<SetStateAction<SpreadsheetData>>;
  setToast: Dispatch<SetStateAction<ToastDetail>>;
  apiKey: string;
  checkApiKey: () => boolean | 0 | undefined;
  isApiKeyDropdownOpen: boolean;
  setIsApiKeyDropdownOpen: (open: boolean) => void;
  setApiKey: (key: string) => void;
}

// Add API URL from environment
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const Spreadsheet: React.FC<SpreadsheetProps> = ({
  setToast,
  data,
  setData,
  apiKey,
  checkApiKey,
  isApiKeyDropdownOpen,
  setIsApiKeyDropdownOpen,
  setApiKey,
}) => {
  const [activeCell, setActiveCell] = useState<Position | null>(null);
  const [editValue, setEditValue] = useState("");
  const [isEditing, setIsEditing] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const tableRef = useRef<HTMLTableElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isEnrichingTable, setIsEnrichingTable] = useState(false);
  const [enrichmentProgress, setEnrichmentProgress] = useState<string>("Starting research...");
  const [isSourcesPanelOpen, setIsSourcesPanelOpen] = useState(false);
  const [collectedQueries, setCollectedQueries] = useState<string[]>([]);
  const [currentQueries, setCurrentQueries] = useState<string[]>([]);
  const [eventHistory, setEventHistory] = useState<Array<{type: string; message: string; timestamp: number; completed: boolean}>>([]);
  const [collectedSources, setCollectedSources] = useState<Source[]>([]);

  // Track previous headers to detect when example data changes
  const prevHeadersRef = useRef<string[]>(data.headers);

  // Reset queries when data changes (e.g., when example is selected)
  useEffect(() => {
    // Check if headers changed (indicates new example was selected)
    const headersChanged = JSON.stringify(prevHeadersRef.current) !== JSON.stringify(data.headers);
    
      if (headersChanged && !isEnrichingTable) {
      // Reset queries when data changes (but not during active enrichment)
      setCollectedQueries([]);
      setCurrentQueries([]);
      setCollectedSources([]);
      setEventHistory([]);
      prevHeadersRef.current = data.headers;
    }
  }, [data.headers, isEnrichingTable]);

  // Add a new row
  const addRow = () => {
    const newRows = [...data.rows];
    newRows.push(Array(data.headers.length).fill({ value: "" }));
    setData({ ...data, rows: newRows });
  };

  // Clear all table data
  const clearTable = () => {
    if (window.confirm("Are you sure you want to clear all data from the table?")) {
      setData({
        headers: Array(5).fill(""),
        rows: Array(5)
          .fill(0)
          .map(() => Array(5).fill({ value: "" })),
      });
      setCollectedQueries([]);
      setCurrentQueries([]);
      setCollectedSources([]);
      setEventHistory([]);
      setToast({
        message: "Table cleared successfully",
        type: "success",
        isShowing: true,
      });
    }
  };

  // File upload functions
  const parseCSV = (file: File): Promise<SpreadsheetData> => {
    return new Promise((resolve, reject) => {
      Papa.parse(file, {
        header: true,
        skipEmptyLines: true,
        complete: (results) => {
          try {
            const data = results.data as Record<string, any>[];
            if (data.length === 0) {
              reject(new Error("CSV file is empty"));
              return;
            }

            const headers = results.meta.fields || Object.keys(data[0]);
            const rows: any[][] = data.map((row) =>
              headers.map((header) => ({
                value: String(row[header] || "").trim(),
                sources: [],
                enriched: false,
                loading: false,
              }))
            );

            resolve({ headers, rows });
          } catch (error) {
            reject(error);
          }
        },
        error: (error) => {
          reject(new Error(`Failed to parse CSV: ${error.message}`));
        },
      });
    });
  };

  const parseExcel = (file: File): Promise<SpreadsheetData> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      
      reader.onload = (e) => {
        try {
          const data = new Uint8Array(e.target?.result as ArrayBuffer);
          const workbook = XLSX.read(data, { type: "array" });
          
          const firstSheetName = workbook.SheetNames[0];
          const worksheet = workbook.Sheets[firstSheetName];
          
          const jsonData = XLSX.utils.sheet_to_json(worksheet, {
            header: 1,
            defval: "",
          }) as any[][];

          if (jsonData.length === 0) {
            reject(new Error("Excel file is empty"));
            return;
          }

          const headers = jsonData[0].map((h) => String(h || "").trim());
          
          if (headers.length === 0 || headers.every(h => !h)) {
            reject(new Error("Excel file has no valid headers"));
            return;
          }
          
          const rows: any[][] = jsonData.length > 1
            ? jsonData.slice(1).map((row) =>
                headers.map((_, index) => ({
                  value: String(row[index] || "").trim(),
                  sources: [],
                  enriched: false,
                  loading: false,
                }))
              )
            : [];

          resolve({ headers, rows });
        } catch (error) {
          reject(new Error(`Failed to parse Excel file: ${error instanceof Error ? error.message : "Unknown error"}`));
        }
      };

      reader.onerror = () => {
        reject(new Error("Failed to read file"));
      };

      reader.readAsArrayBuffer(file);
    });
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const fileExtension = file.name.split('.').pop()?.toLowerCase();
    if (!fileExtension || !['csv', 'xlsx', 'xls'].includes(fileExtension)) {
      setToast({
        message: "Please upload a CSV, XLSX, or XLS file",
        type: "error",
        isShowing: true,
      });
      return;
    }

    try {
      setToast({
        message: "Parsing file...",
        type: "info",
        isShowing: true,
      });

      let parsedData: SpreadsheetData;

      if (fileExtension === 'csv') {
        parsedData = await parseCSV(file);
      } else {
        parsedData = await parseExcel(file);
      }

      setData(parsedData);
      setCollectedQueries([]);
      setCurrentQueries([]);
      setEventHistory([]);

      setToast({
        message: "File uploaded successfully!",
        type: "success",
        isShowing: true,
      });
    } catch (error) {
      console.error("Error parsing file:", error);
      setToast({
        message: error instanceof Error ? error.message : "Failed to parse file",
        type: "error",
        isShowing: true,
      });
    } finally {
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  // Add a new column
  const addColumn = () => {
    const newHeaders = [...data.headers, ""];
    const newRows = data.rows.map((row) => [...row, { value: "" }]);
    setData({ headers: newHeaders, rows: newRows });
  };

  // Delete a column
  const deleteColumn = (colIndex: number) => {
    if (data.headers.length <= 1) return; // Prevent deleting the last column

    const newHeaders = [...data.headers];
    newHeaders.splice(colIndex, 1);

    const newRows = data.rows.map((row) => {
      const newRow = [...row];
      newRow.splice(colIndex, 1);
      return newRow;
    });

    setData({ headers: newHeaders, rows: newRows });
  };

  // Focus on cell
  const focusCell = (row: number, col: number) => {
    setActiveCell({ row, col });
    setEditValue(data.rows[row][col].value);
    setIsEditing(true);
  };

  // Handle cell change
  const handleCellChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setEditValue(e.target.value);
  };

  // Save cell value
  const saveCell = () => {
    if (activeCell) {
      const { row, col } = activeCell;
      const newRows = [...data.rows];
      newRows[row][col] = {
        ...newRows[row][col],
        value: editValue,
        sources: [],
      };
      setData({ ...data, rows: newRows });
      setIsEditing(false);
      setActiveCell(null);
    }
  };

  // Handle key events
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (isEditing) {
      if (e.key === "Enter") {
        saveCell();
      } else if (e.key === "Escape") {
        setIsEditing(false);
        setActiveCell(null);
      }
    }
  };

  const enrichTable = async () => {
    if (!data.headers[0]?.trim()) {
      setToast({
        message: "Please set the first column header",
        type: "error",
        isShowing: true,
      });
      return;
    }

    if (!data.rows.map((row) => row[0]).some((cell) => cell?.value?.length)) {
      setToast({
        message: "Please specify a key in the first column",
        type: "error",
        isShowing: true,
      });
      return;
    }

    if (!checkApiKey()) {
      setToast({
        message: "Please set a valid API Key",
        type: "error",
        isShowing: true,
      });
      return;
    }

    // Set table-wide loading state
    setIsEnrichingTable(true);
    setEnrichmentProgress("Starting research...");
    setCollectedQueries([]); // Reset queries for new enrichment
    setCurrentQueries([]);
    setCollectedSources([]); // Reset sources for new enrichment
    setEventHistory([{ type: "start", message: "Starting research...", timestamp: Date.now(), completed: false }]);

    try {
      // Convert table data to the required format
      const headers = data.headers;
      const rows = data.rows.map((row) =>
        row.map((cell) => cell.value || "")
      );

      // Find a column with data to use as context (first non-empty column)
      const contextColumn = headers.find((_, idx) =>
        data.rows.some((row) => row[idx]?.value?.trim())
      );

      // Create abort controller for cancellation
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 180000); // 3 minute timeout

      const response = await fetch(`${API_URL}/api/enrich/table-wide/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: apiKey,
        },
        body: JSON.stringify({
          headers: headers,
          rows: rows,
          context_column: contextColumn,
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        clearTimeout(timeoutId);
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || response.statusText || "Request failed");
      }

      // Check if response body exists
      if (!response.body) {
        clearTimeout(timeoutId);
        throw new Error("No response body");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          break;
        }

        // Decode chunk and add to buffer
        buffer += decoder.decode(value, { stream: true });

        // Process complete SSE messages (lines ending with \n)
        const lines = buffer.split("\n");
        buffer = lines.pop() || ""; // Keep incomplete line in buffer

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const eventData = JSON.parse(line.slice(6)); // Remove "data: " prefix
              handleStreamEvent(eventData, headers);
            } catch (parseError) {
              console.error("Error parsing SSE data:", parseError, line);
            }
          }
        }
      }

      clearTimeout(timeoutId);
    } catch (error) {
      console.error("Error during table enrichment:", error);
      
      if (error instanceof Error && error.name === "AbortError") {
        setToast({
          message: "Request timed out. Please try again.",
          type: "error",
          isShowing: true,
        });
      } else {
        setToast({
          message: error instanceof Error ? error.message : "Table enrichment failed",
          type: "error",
          isShowing: true,
        });
      }
      setIsEnrichingTable(false);
      setEnrichmentProgress("");
      setCurrentQueries([]);
      setEventHistory([]);
    }
  };

  const handleStreamEvent = (
    event: any,
    headers: string[]
  ) => {
    switch (event.type) {
      case "start":
        setEnrichmentProgress(event.message || "Starting research...");
        setEventHistory((prev) => [...prev, { type: "start", message: event.message || "Starting research...", timestamp: Date.now(), completed: true }]);
        break;

      case "progress":
        setEnrichmentProgress(event.message || "Processing...");
        break;

      case "tool_call":
        if (event.tool === "WebSearch" && event.queries) {
          // Collect queries
          const newQueries = Array.isArray(event.queries) ? event.queries : [];
          setCollectedQueries((prev) => [...prev, ...newQueries]);
          // Accumulate queries from all batches instead of replacing
          setCurrentQueries((prev) => {
            // Combine previous and new queries, removing duplicates
            const combined = [...prev, ...newQueries];
            // Remove duplicates based on query text
            const unique = combined.filter((query, index, self) => 
              self.findIndex(q => q === query) === index
            );
            return unique;
          });
          
          // Collect sources if available - check multiple possible locations
          const sourcesToAdd: Source[] = [];
          
          // Check event.sources (direct)
          if (event.sources && Array.isArray(event.sources)) {
            sourcesToAdd.push(...event.sources);
          }
          
          // Check event.reasoning_steps for sources
          if (event.reasoning_steps && Array.isArray(event.reasoning_steps)) {
            event.reasoning_steps.forEach((step: any) => {
              if (step.sources && Array.isArray(step.sources)) {
                sourcesToAdd.push(...step.sources);
              }
              // Check sub-steps
              if (step.sub_steps && Array.isArray(step.sub_steps)) {
                step.sub_steps.forEach((subStep: any) => {
                  if (subStep.sources && Array.isArray(subStep.sources)) {
                    sourcesToAdd.push(...subStep.sources);
                  }
                });
              }
            });
          }
          
          // Add unique sources
          if (sourcesToAdd.length > 0) {
            setCollectedSources((prev) => {
              const seenUrls = new Set(prev.map(s => s.url));
              const newSources = sourcesToAdd.filter((s: Source) => s && s.url && !seenUrls.has(s.url));
              return [...prev, ...newSources];
            });
          }
          
          const message = `Searching the web (${newQueries.length} ${newQueries.length === 1 ? "query" : "queries"})...`;
          setEnrichmentProgress(message);
          setEventHistory((prev) => {
            // Mark previous active events as completed
            const updated = prev.map(e => e.completed ? e : { ...e, completed: true });
            return [...updated, { type: "search", message, timestamp: Date.now(), completed: false }];
          });
        } else if (event.tool === "Planning") {
          setEnrichmentProgress("Planning research strategy...");
          setCurrentQueries([]);
          setEventHistory((prev) => {
            // Only add Planning if it doesn't already exist (keep first one)
            const hasPlanning = prev.some(e => e.type === "planning");
            if (hasPlanning) {
              return prev; // Don't add duplicate Planning
            }
            const updated = prev.map(e => e.completed ? e : { ...e, completed: true });
            return [...updated, { type: "planning", message: "Planning research strategy...", timestamp: Date.now(), completed: false }];
          });
        } else if (event.tool === "Generating") {
          setEnrichmentProgress("Generating research report...");
          setCurrentQueries([]);
          // Don't add "Generating" to eventHistory - user doesn't want it in Research Progress
        } else if (event.tool === "ResearchSubtopic") {
          setEnrichmentProgress("Researching subtopics...");
          setCurrentQueries([]);
          setEventHistory((prev) => {
            const updated = prev.map(e => e.completed ? e : { ...e, completed: true });
            return [...updated, { type: "research", message: "Researching subtopics...", timestamp: Date.now(), completed: false }];
          });
        } else {
          setEnrichmentProgress(`Executing ${event.tool || "tool"}...`);
          setCurrentQueries([]);
        }
        break;

      case "sources_found":
        // Collect sources in real-time from the event
        if (event.sources && Array.isArray(event.sources)) {
          setCollectedSources((prev) => {
            const seenUrls = new Set(prev.map(s => s.url));
            const newSources = event.sources.filter((s: Source) => s && s.url && !seenUrls.has(s.url));
            return [...prev, ...newSources];
          });
          // Update progress message with source count
          setEnrichmentProgress(`Found ${event.count || event.sources.length} sources from ${event.tool || "research"}...`);
        }
        break;

      case "sources_complete":
        // Accumulate sources from complete event (don't replace, just add new ones)
        if (event.sources && Array.isArray(event.sources)) {
          setCollectedSources((prev) => {
            const seenUrls = new Set(prev.map(s => s.url));
            const newSources = event.sources.filter((s: Source) => s && s.url && !seenUrls.has(s.url));
            const updated = [...prev, ...newSources];
            // Update progress message with total count
            const totalCount = event.count || updated.length;
            setEnrichmentProgress(`Research complete: ${totalCount} total sources`);
            return updated;
          });
        } else {
          setEnrichmentProgress(`Research complete: ${event.count || collectedSources.length} total sources`);
        }
        break;

      case "content_chunk":
        // Optionally show content generation progress
        setEnrichmentProgress("Generating content...");
        break;

      case "complete":
        // Update all rows at once with enriched values
        if (event.enriched_values && event.sources) {
          // Debug logging
          console.log("Enrichment complete event:", {
            enriched_values_keys: Object.keys(event.enriched_values),
            row_count: data.rows.length,
            enriched_row_counts: Object.keys(event.enriched_values).map(col => ({
              column: col,
              count: event.enriched_values[col]?.length || 0
            }))
          });

          // Collect all sources from the complete event
          const allSourcesFromEvent: Source[] = [];
          Object.keys(event.sources).forEach((columnName) => {
            const columnSources = event.sources[columnName];
            if (Array.isArray(columnSources)) {
              columnSources.forEach((rowSources: Source[]) => {
                if (Array.isArray(rowSources)) {
                  rowSources.forEach((source: Source) => {
                    if (source && source.url) {
                      allSourcesFromEvent.push(source);
                    }
                  });
                }
              });
            }
          });
          
          // Add unique sources to collected sources
          if (allSourcesFromEvent.length > 0) {
            setCollectedSources((prev) => {
              const seenUrls = new Set(prev.map(s => s.url));
              const newSources = allSourcesFromEvent.filter(s => s && s.url && !seenUrls.has(s.url));
              return [...prev, ...newSources];
            });
          }
          
          const updatedRows = data.rows.map((row, rowIndex) => {
            const newRow = [...row];

            // Update each column for this row
            Object.keys(event.enriched_values).forEach((columnName) => {
              const colIndex = headers.indexOf(columnName);
              const columnValues = event.enriched_values[columnName];
              
              // Check if column exists and if there's a value for this row
              if (
                colIndex >= 0 &&
                Array.isArray(columnValues) &&
                rowIndex < columnValues.length
              ) {
                const enrichedValue = columnValues[rowIndex];
                // Update the cell if we have a value (even if it's an empty string)
                // Empty string means enrichment was attempted but no value found
                if (enrichedValue !== undefined && enrichedValue !== null) {
                  newRow[colIndex] = {
                    value: String(enrichedValue || ""),
                    sources: event.sources[columnName]?.[rowIndex] || [],
                    enriched: String(enrichedValue || "").trim() !== "",
                    loading: false,
                  };
                }
              }
            });

            return newRow;
          });

          console.log("Updated rows:", updatedRows);
          setData({ ...data, rows: updatedRows });
          setToast({
            message: "Table enriched successfully!",
            type: "success",
            isShowing: true,
          });
        }
        // Mark all events as completed and add completion event
        setEventHistory((prev) => {
          const completed = prev.map(e => ({ ...e, completed: true }));
          return [...completed, { type: "complete", message: "Enrichment complete!", timestamp: Date.now(), completed: true }];
        });
        setIsEnrichingTable(false);
        setEnrichmentProgress("");
        setCurrentQueries([]);
        // Clear event history after a delay
        setTimeout(() => {
          setEventHistory([]);
          setCollectedSources([]);
        }, 2000);
        break;

      case "error":
        setToast({
          message: event.message || "Enrichment failed",
          type: "error",
          isShowing: true,
        });
        setIsEnrichingTable(false);
        setEnrichmentProgress("");
        setCurrentQueries([]);
        break;

      default:
        // Unknown event type, log for debugging
        console.log("Unknown event type:", event.type, event);
    }
  };

  // Edit column header
  const [editingHeader, setEditingHeader] = useState<number | null>(null);
  const [headerEditValue, setHeaderEditValue] = useState("");
  const headerInputRef = useRef<HTMLInputElement>(null);

  const startEditingHeader = (index: number) => {
    setEditingHeader(index);
    setHeaderEditValue(data.headers[index]);
  };

  const saveHeaderEdit = () => {
    if (editingHeader !== null) {
      const newHeaders = [...data.headers];
      newHeaders[editingHeader] = headerEditValue;
      setData({ ...data, headers: newHeaders });
      setEditingHeader(null);
    }
  };

  const handleHeaderKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      saveHeaderEdit();
    } else if (e.key === "Escape") {
      setEditingHeader(null);
    }
  };

  // Focus the input when editing starts
  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isEditing]);

  // Focus header input when editing header
  useEffect(() => {
    if (editingHeader !== null && headerInputRef.current) {
      headerInputRef.current.focus();
    }
  }, [editingHeader]);

  // Collect all sources from all cells and deduplicate by URL
  const getAllSources = (): Source[] => {
    const allSources: Source[] = [];
    const seenUrls = new Set<string>();
    
    data.rows.forEach((row) => {
      row.forEach((cell) => {
        if (cell.sources && Array.isArray(cell.sources) && cell.sources.length > 0) {
          // Filter out any invalid sources (must have url) and deduplicate
          cell.sources.forEach((source) => {
            if (source && source.url && typeof source.url === "string") {
              // Only add if we haven't seen this URL before
              if (!seenUrls.has(source.url)) {
                seenUrls.add(source.url);
                allSources.push(source);
              }
            }
          });
        }
      });
    });
    return allSources;
  };

  const allSources = getAllSources();
  const hasSources = allSources.length > 0;
  
  // Get unique queries
  const uniqueQueries = Array.from(new Set(collectedQueries));
  const hasQueries = uniqueQueries.length > 0;

  return (
    <motion.div
      className="w-full mb-40 relative"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      style={{ marginTop: "25px" }}
    >
      {/* Loading overlay for table-wide enrichment */}
      {isEnrichingTable && (
        <div className="fixed inset-0 flex items-center justify-center z-50" style={{ background: "rgba(60, 58, 57, 0.6)", backdropFilter: "blur(4px)" }}>
          <motion.div
            className="glass rounded-[18px] shadow-2xl w-full max-w-5xl mx-4"
            style={{ 
              background: "rgba(255, 255, 255, 0.95)",
              backdropFilter: "blur(20px)",
              border: "1px solid var(--color-black-10)",
              maxHeight: "85vh",
              boxShadow: "0 20px 60px rgba(0, 0, 0, 0.15)"
            }}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.3 }}
          >
            <div className="flex flex-col p-6 sm:p-8 h-full">
              {/* Header */}
              <div className="flex items-center gap-3 mb-4 flex-shrink-0">
                <div className="w-10 h-10 border-4 rounded-full animate-spin flex-shrink-0" style={{ 
                  borderColor: "var(--color-primary-blue)",
                  borderTopColor: "transparent"
                }}></div>
                <div className="flex-1">
                  <p className="text-lg font-medium" style={{ color: "var(--color-black)" }}>
                    Enriching entire table...
                  </p>
                  <p className="text-sm" style={{ color: "var(--color-black-60)" }}>
                    {enrichmentProgress || "Processing..."}
                  </p>
                </div>
              </div>

              {/* Favicons at the top - Max 2 rows */}
              {(() => {
                const sourcesWithFavicons = collectedSources.filter(s => s.favicon && s.favicon !== null);
                // Estimate: favicon (24px) + gap (8px) = 32px per item
                // Container width ~400px, so ~12 per row = 24 total for 2 rows
                const maxFavicons = 24;
                const visibleFavicons = sourcesWithFavicons.slice(0, maxFavicons);
                const remainingCount = sourcesWithFavicons.length - maxFavicons;
                
                return sourcesWithFavicons.length > 0 ? (
                  <div className="flex flex-wrap items-center gap-2 mb-4 pb-4 border-b flex-shrink-0" style={{ 
                    borderColor: "var(--color-black-10)",
                    maxHeight: "52px", // 2 rows: (24px favicon + 8px gap) * 2 = 64px, but using 52px for tighter spacing
                    overflow: "hidden"
                  }}>
                    {visibleFavicons.map((source, idx) => (
                      <motion.a
                        key={source.url || idx}
                        href={source.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: idx * 0.02 }}
                        className="flex items-center justify-center w-6 h-6 rounded transition-colors flex-shrink-0"
                        style={{
                          background: "var(--color-black-5)",
                          border: "1px solid var(--color-black-10)"
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.background = "var(--color-beige)";
                          e.currentTarget.style.borderColor = "var(--color-primary-blue)";
                          e.currentTarget.style.transform = "scale(1.1)";
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.background = "var(--color-black-5)";
                          e.currentTarget.style.borderColor = "var(--color-black-10)";
                          e.currentTarget.style.transform = "scale(1)";
                        }}
                        title={source.title || source.url}
                      >
                        <img
                          src={source.favicon!}
                          alt=""
                          className="w-4 h-4 rounded-sm"
                          onError={(e) => {
                            (e.target as HTMLImageElement).style.display = "none";
                          }}
                        />
                      </motion.a>
                    ))}
                    {remainingCount > 0 && (
                      <div className="flex items-center justify-center min-w-[24px] h-6 px-1.5 rounded flex-shrink-0" style={{
                        background: "var(--color-black-5)",
                        border: "1px solid var(--color-black-10)"
                      }}>
                        <span className="text-xs font-medium whitespace-nowrap" style={{ color: "var(--color-black-60)" }}>
                          +{remainingCount}
                        </span>
                      </div>
                    )}
                  </div>
                ) : null;
              })()}
              
              {/* Two-column layout: Research Progress (1/3 left), Active Queries (2/3 right) - Fixed sizes with scrolling */}
              <div className="flex gap-4" style={{ height: "60vh" }}>
                {/* Left: Research Progress - 1/3 width */}
                <div className="flex-[1] flex flex-col rounded-xl border overflow-hidden" style={{
                  background: "var(--color-white)",
                  borderColor: "var(--color-black-10)",
                  minWidth: 0,
                  height: "100%"
                }}>
                  <div className="px-4 py-3 border-b flex-shrink-0" style={{ 
                    borderColor: "var(--color-black-10)",
                    background: "var(--color-light-gray)"
                  }}>
                    <p className="text-sm font-medium" style={{ color: "var(--color-black)" }}>
                      Research Progress
                    </p>
                  </div>
                  <div className="flex-1 overflow-y-auto p-4" style={{ minHeight: 0, maxHeight: "100%" }}>
                    <div className="space-y-3">
                      {eventHistory.length > 0 ? (
                        eventHistory.map((event, index) => {
                          const getEventIcon = () => {
                            if (event.completed) {
                              return <CheckCircle2 className="w-4 h-4" style={{ color: "#22C55E" }} />;
                            }
                            return <Loader2 className="w-4 h-4 animate-spin" style={{ color: "var(--color-primary-blue)" }} />;
                          };

                          const getEventLabel = () => {
                            switch (event.type) {
                              case "start": return "Started";
                              case "planning": return "Planning";
                              case "search": return "Searching";
                              case "research": return "Researching";
                              case "generating": return "Generating";
                              case "complete": return "Complete";
                              default: return "Processing";
                            }
                          };

                          return (
                            <motion.div
                              key={index}
                              initial={{ opacity: 0, x: -10 }}
                              animate={{ opacity: 1, x: 0 }}
                              transition={{ delay: index * 0.1 }}
                              className="flex items-start gap-3"
                            >
                              <div className="flex-shrink-0 mt-0.5">
                                {getEventIcon()}
                              </div>
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2 mb-0.5">
                                  <span className="text-sm font-medium" style={{ color: "var(--color-black)" }}>
                                    {getEventLabel()}
                                  </span>
                                </div>
                                <p className="text-xs leading-relaxed" style={{ color: "var(--color-black-60)" }}>
                                  {event.message}
                                </p>
                              </div>
                            </motion.div>
                          );
                        })
                      ) : (
                        <p className="text-sm" style={{ color: "var(--color-black-40)" }}>
                          Waiting for events...
                        </p>
                      )}
                    </div>
                  </div>
                </div>

                {/* Right: Active Queries - 2/3 width, Always visible */}
                <div className="flex-[2] flex flex-col rounded-xl border overflow-hidden" style={{
                  background: "var(--color-white)",
                  borderColor: "var(--color-black-10)",
                  minWidth: 0,
                  height: "100%"
                }}>
                  <div className="px-4 py-3 border-b flex-shrink-0" style={{ 
                    borderColor: "var(--color-black-10)",
                    background: "var(--color-light-gray)"
                  }}>
                    <div className="flex items-center gap-2">
                      <Search className="w-4 h-4" style={{ color: "var(--color-primary-blue)" }} />
                      <span className="text-sm font-medium" style={{ color: "var(--color-black)" }}>
                        Active Queries
                      </span>
                      {currentQueries.length > 0 && (
                        <span className="text-xs px-2 py-0.5 rounded-full" style={{ 
                          background: "rgba(38, 119, 255, 0.1)",
                          color: "var(--color-primary-blue)"
                        }}>
                          {currentQueries.length}
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="flex-1 overflow-y-auto p-4" style={{ minHeight: 0, maxHeight: "100%" }}>
                    {currentQueries.length > 0 ? (
                      <div className="space-y-2">
                        {currentQueries.map((query, index) => (
                          <motion.div
                            key={index}
                            initial={{ opacity: 0, y: 5 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: index * 0.05 }}
                            className="flex items-start gap-2 p-3 rounded-lg"
                            style={{
                              background: "var(--color-black-5)"
                            }}
                          >
                            <Zap className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" style={{ color: "var(--color-primary-blue)" }} />
                            <p className="text-xs leading-relaxed flex-1" style={{ color: "var(--color-black)" }}>
                              {query}
                            </p>
                          </motion.div>
                        ))}
                      </div>
                    ) : (
                      <p className="text-sm" style={{ color: "var(--color-black-40)" }}>
                        No active queries yet...
                      </p>
                    )}
                  </div>
                </div>
              </div>

            </div>
          </motion.div>
        </div>
      )}
      {/* Unified Sources Panel */}
      <UnifiedSourcesPanel
        sources={allSources}
        isOpen={isSourcesPanelOpen}
        onClose={() => setIsSourcesPanelOpen(false)}
      />

      {/* Table at the top */}
      <div 
        className="w-full mb-6 glass rounded-[14px] sm:rounded-[18px] p-3 sm:p-4" 
        style={{ 
          opacity: 0.95,
          maxHeight: "calc(100vh - 250px)",
          display: "inline-flex",
          flexDirection: "column",
          width: "100%"
        }}
      >
        <div style={{ 
          overflowX: "auto", 
          overflowY: "auto",
          flexShrink: 0,
          maxHeight: "calc(100vh - 350px)",
          position: "relative",
          width: "100%"
        }}>
          <table
            ref={tableRef}
            className="border-separate border-spacing-0 w-full"
            style={{ 
              minWidth: "max-content",
              width: "100%"
            }}
          >
          <thead>
            <tr>
              <th className="w-14 p-2 sticky left-0 top-0 z-30 first-cell" style={{
                background: "var(--color-light-gray)",
                border: "1px solid var(--color-black-10)",
                position: "sticky",
                left: 0,
                top: 0
              }}>
              </th>
              {data.headers.map((header, index) => (
                <th
                  key={index}
                  className={`w-40 max-w-[150px] p-2 text-left relative h-12 sticky top-0 ${
                    index === data.headers.length - 1
                      ? "last-header"
                      : ""
                  }`}
                  style={{
                    background: "var(--color-light-gray)",
                    border: "1px solid var(--color-black-10)",
                    position: "sticky",
                    top: 0,
                    zIndex: 25,
                    overflow: "hidden",
                    wordBreak: "break-word"
                  }}
                >
                  <div className="flex justify-between items-center group">
                    <div className="flex items-center w-full">
                      {editingHeader === index ? (
                        <input
                          ref={headerInputRef}
                          type="text"
                          value={headerEditValue}
                          onChange={(e) => setHeaderEditValue(e.target.value)}
                          onBlur={saveHeaderEdit}
                          onKeyDown={handleHeaderKeyDown}
                          className="w-full text-sm font-medium outline-none focus:outline-none focus:ring-0 focus:border-transparent"
                          style={{ 
                            background: "transparent",
                            color: "var(--color-black)"
                          }}
                          placeholder="Enter column name..."
                        />
                      ) : (
                        <div
                          className="flex items-center max-w-[140px]"
                          onClick={() => startEditingHeader(index)}
                        >
                          <span className="font-medium overflow-hidden text-ellipsis whitespace-nowrap">
                            {header}
                          </span>
                          <button
                            className="ml-2 p-1 rounded-full transition-colors"
                            style={{ 
                              color: "var(--color-black-40)"
                            }}
                            onMouseEnter={(e) => {
                              e.currentTarget.style.color = "var(--color-primary-blue)";
                              e.currentTarget.style.background = "var(--color-beige)";
                            }}
                            onMouseLeave={(e) => {
                              e.currentTarget.style.color = "var(--color-black-40)";
                              e.currentTarget.style.background = "transparent";
                            }}
                            title="Edit column name"
                          >
                            <Pencil size={14} />
                          </button>
                        </div>
                      )}
                    </div>
                    {index !== 0 && (
                      <div className="flex space-x-1 opacity-0 group-hover:opacity-100 transition-opacity">
                        <button
                          className="p-1 rounded-full transition-colors"
                          style={{ color: "#FE363B" }}
                          onMouseEnter={(e) => {
                            e.currentTarget.style.background = "rgba(254, 54, 59, 0.1)";
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.background = "transparent";
                          }}
                          onClick={() => deleteColumn(index)}
                          title="Delete column"
                        >
                          <Trash2 size={14} />
                        </button>
                      </div>
                    )}
                  </div>
                </th>
              ))}
              <th
                className={`w-14 p-2 text-center sticky top-0 ${
                  data.headers.length === data.headers.length - 1 ? "last-header" : ""
                }`}
                style={{
                  background: "var(--color-light-gray)",
                  border: "1px solid var(--color-black-10)",
                  position: "sticky",
                  top: 0,
                  zIndex: 25
                }}
              >
                <button
                  onClick={addColumn}
                  className="w-8 h-8 rounded-full flex items-center justify-center transition-colors mx-auto"
                  style={{ 
                    background: "var(--color-black-5)",
                    color: "var(--color-black-40)"
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = "var(--color-beige)";
                    e.currentTarget.style.color = "var(--color-primary-blue)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = "var(--color-black-5)";
                    e.currentTarget.style.color = "var(--color-black-40)";
                  }}
                  title="Add column"
                >
                  <Plus size={16} />
                </button>
              </th>
            </tr>
          </thead>
          <tbody>
            {data.rows.map((row, rowIndex) => (
              <tr
                key={rowIndex}
                className="transition-colors"
                style={{
                  backgroundColor: "var(--color-white)"
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = "var(--color-beige)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = "var(--color-white)";
                }}
              >
                <td
                  className={`p-2 text-center font-medium sticky left-0 ${
                    rowIndex === data.rows.length - 1
                      ? "last-row-first-cell"
                      : ""
                  }`}
                  style={{
                    background: "var(--color-white)",
                    border: "1px solid var(--color-black-10)",
                    color: "var(--color-black)",
                    position: "sticky",
                    left: 0,
                    zIndex: 20
                  }}
                >
                  {rowIndex + 1}
                </td>
                {row.map((cell, colIndex) => (
                  <motion.td
                    key={colIndex}
                    className={`max-w-[150px] p-2 relative ${
                      rowIndex === data.rows.length - 1 &&
                      colIndex === row.length - 1
                        ? "last-cell"
                        : ""
                    }`}
                    style={{
                      background: "var(--color-white)",
                      border: "1px solid var(--color-black-10)",
                      overflow: "hidden",
                      wordBreak: "break-word"
                    }}
                    onClick={() => focusCell(rowIndex, colIndex)}
                    data-enriched={cell.enriched ? "true" : "false"}
                    title={cell.value || ""}
                  >
                    {activeCell?.row === rowIndex &&
                    activeCell?.col === colIndex &&
                    isEditing ? (
                      <input
                        ref={inputRef}
                        type="text"
                        value={editValue}
                        onChange={handleCellChange}
                        onBlur={saveCell}
                        onKeyDown={handleKeyDown}
                        className="w-full h-full p-0 border-0 outline-none bg-transparent focus:outline-none focus:ring-0 focus:border-transparent"
                      />
                    ) : (
                      <motion.div
                        className="w-full min-h-6 flex items-start"
                        initial={
                          cell.enriched ? { opacity: 0 } : { opacity: 1 }
                        }
                        animate={{ opacity: 1 }}
                        transition={{ duration: 0.3 }}
                        style={{
                          maxHeight: "7.5rem", // ~6 lines at 1.25rem line-height
                          overflow: "hidden"
                        }}
                      >
                        {cell.loading ? (
                          <div className="flex items-center justify-left w-full">
                            <div className="w-5 h-5 border-2 border-t-transparent rounded-full animate-spin" style={{
                              borderColor: "var(--color-primary-blue)"
                            }}></div>
                            <span className="ml-2 text-sm" style={{ color: "var(--color-black-60)" }}>
                              Researching... (10-30s)
                            </span>
                          </div>
                        ) : (
                          <>
                            <span
                              className="w-full text-sm leading-5"
                              style={{
                                color: cell.enriched ? "var(--color-primary-blue)" : "var(--color-black)",
                                fontWeight: cell.enriched ? 500 : 400,
                                display: "-webkit-box",
                                WebkitLineClamp: 6,
                                WebkitBoxOrient: "vertical",
                                overflow: "hidden",
                                textOverflow: "ellipsis",
                                wordBreak: "break-word"
                              }}
                              title={cell.value || ""}
                            >
                              {cell.value}
                            </span>
                          </>
                        )}
                      </motion.div>
                    )}
                  </motion.td>
                ))}
                <td
                  className={`w-14 p-2 text-center ${
                    rowIndex === data.rows.length - 1 ? "last-cell" : ""
                  }`}
                  style={{
                    background: "var(--color-white)",
                    border: "1px solid var(--color-black-10)"
                  }}
                ></td>
              </tr>
            ))}
          </tbody>
        </table>
        </div>
        <div className="mt-2 flex items-center justify-end relative">
          {/* Enrich button - centered */}
          <div className="absolute left-1/2 transform -translate-x-1/2">
            <div
              className="px-3 py-1.5 text-center rounded-md cursor-pointer transition-colors flex items-center gap-2"
              style={{
                background: isEnrichingTable ? "var(--color-black-10)" : "var(--color-primary-blue)",
                color: "white",
                opacity: isEnrichingTable ? 0.5 : 1,
                pointerEvents: isEnrichingTable ? "none" : "auto"
              }}
              onMouseEnter={(e) => {
                if (!isEnrichingTable) {
                  e.currentTarget.style.background = "#8FBCFA";
                }
              }}
              onMouseLeave={(e) => {
                if (!isEnrichingTable) {
                  e.currentTarget.style.background = "var(--color-primary-blue)";
                }
              }}
              onClick={enrichTable}
            >
              <Sparkles size={16} />
              <span className="text-sm font-medium">
                {isEnrichingTable ? "Enriching..." : "Enrich"}
              </span>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv,.xlsx,.xls"
              onChange={handleFileUpload}
              className="hidden"
              aria-label="Upload file"
            />
            {/* Upload button */}
            <div
              className="p-0 text-center rounded-md cursor-pointer transition-colors"
              style={{
                background: "var(--color-black-5)",
                color: "var(--color-black-40)"
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = "var(--color-beige)";
                e.currentTarget.style.color = "var(--color-primary-blue)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = "var(--color-black-5)";
                e.currentTarget.style.color = "var(--color-black-40)";
              }}
              onClick={handleUploadClick}
            >
              <button
                className="w-8 h-8 flex items-center justify-center transition-colors"
                title="Upload file"
              >
                <Upload size={16} />
              </button>
            </div>
            {/* Export button */}
            <div
              className="p-0 text-center rounded-md cursor-pointer transition-colors"
              style={{
                background: "var(--color-black-5)",
                color: "var(--color-black-40)"
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = "var(--color-beige)";
                e.currentTarget.style.color = "var(--color-primary-blue)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = "var(--color-black-5)";
                e.currentTarget.style.color = "var(--color-black-40)";
              }}
              onClick={() => exportToCSV(data)}
            >
              <button
                className="w-8 h-8 flex items-center justify-center transition-colors"
                title="Export to CSV"
              >
                <Download size={16} />
              </button>
            </div>
            {/* Clear button */}
            <div
              className="p-0 text-center rounded-md cursor-pointer transition-colors"
              style={{
                background: "var(--color-black-5)",
                color: "var(--color-black-40)"
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = "rgba(254, 54, 59, 0.1)";
                e.currentTarget.style.color = "#FE363B";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = "var(--color-black-5)";
                e.currentTarget.style.color = "var(--color-black-40)";
              }}
              onClick={clearTable}
            >
              <button
                className="w-8 h-8 flex items-center justify-center transition-colors"
                title="Clear table"
              >
                <Trash2 size={16} />
              </button>
            </div>
            {/* Add row button */}
            <div
              className="p-0 text-center rounded-md cursor-pointer transition-colors"
              style={{
                background: "var(--color-black-5)",
                color: "var(--color-black-40)"
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = "var(--color-beige)";
                e.currentTarget.style.color = "var(--color-primary-blue)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = "var(--color-black-5)";
                e.currentTarget.style.color = "var(--color-black-40)";
              }}
              onClick={addRow}
            >
              <button
                className="w-8 h-8 flex items-center justify-center transition-colors"
                title="Add row"
              >
                <Plus size={16} />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Controls below the table */}
      <div className="flex flex-col gap-4 mt-6">
        {/* Example */}
        <div className="flex items-center justify-between gap-4">
          <ExamplePopup visible={true} onExampleSelect={setData} />
        </div>

        {/* Sources section at the bottom - like research page */}
        {hasSources && (
          <div className="mt-6 glass rounded-[14px] sm:rounded-[18px] p-3 sm:p-4">
            <div
              className="text-sm font-semibold mb-2 sm:mb-3 flex items-center gap-2"
              style={{ color: "var(--color-black)" }}
            >
              <span>Sources</span>
              <span style={{ color: "var(--color-black-60)" }}>
                ({allSources.length})
              </span>
            </div>
            <div className="space-y-2 max-h-[300px] overflow-y-auto">
              {allSources.map((source, idx) => (
                <a
                  key={idx}
                  href={source.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block p-2.5 sm:p-3 rounded-[10px] sm:rounded-[12px] transition active:scale-[0.98]"
                  style={{
                    background: "var(--color-white)",
                    border: "1px solid var(--color-black-10)",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor =
                      "var(--color-primary-blue)";
                    e.currentTarget.style.boxShadow =
                      "0 2px 4px rgba(38, 119, 255, 0.1)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor =
                      "var(--color-black-10)";
                    e.currentTarget.style.boxShadow = "none";
                  }}
                >
                  <div className="flex items-center gap-2">
                    {source.favicon && (
                      <img
                        src={source.favicon}
                        alt=""
                        className="w-4 h-4 flex-shrink-0"
                        onError={(e) => {
                          e.currentTarget.style.display = "none";
                        }}
                      />
                    )}
                    <div
                      className="text-sm font-medium line-clamp-1"
                      style={{ color: "var(--color-primary-blue)" }}
                    >
                      {source.title}
                    </div>
                  </div>
                  {(source as any).snippet && (
                    <div
                      className="text-xs mt-1 line-clamp-2 hidden sm:block"
                      style={{ color: "var(--color-black-60)" }}
                    >
                      {(source as any).snippet}
                    </div>
                  )}
                  <div
                    className="text-xs mt-1"
                    style={{ color: "var(--color-black-40)" }}
                  >
                    {new URL(source.url).hostname}
                  </div>
                </a>
              ))}
            </div>
          </div>
        )}

        {/* Search Queries section at the bottom - like research page */}
        {hasQueries && (
          <div className="mt-6 glass rounded-[14px] sm:rounded-[18px] p-3 sm:p-4">
            <div
              className="text-sm font-semibold mb-2 sm:mb-3 flex items-center gap-2"
              style={{ color: "var(--color-black)" }}
            >
              <span>Search Queries</span>
              <span style={{ color: "var(--color-black-60)" }}>
                ({uniqueQueries.length})
              </span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 max-h-[300px] overflow-y-auto">
              {uniqueQueries.map((query, index) => (
                <div
                  key={index}
                  className="flex items-start gap-2 p-2.5 sm:p-3 rounded-[10px] sm:rounded-[12px] transition"
                  style={{
                    background: "var(--color-white)",
                    border: "1px solid var(--color-black-10)",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor =
                      "var(--color-primary-blue)";
                    e.currentTarget.style.boxShadow =
                      "0 2px 4px rgba(38, 119, 255, 0.1)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor =
                      "var(--color-black-10)";
                    e.currentTarget.style.boxShadow = "none";
                  }}
                >
                  <Search className="w-4 h-4 flex-shrink-0 mt-0.5" style={{ color: "var(--color-primary-blue)" }} />
                  <div
                    className="text-sm leading-relaxed flex-1"
                    style={{ color: "var(--color-black)" }}
                  >
                    {query}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </motion.div>
  );
};

export default Spreadsheet;
