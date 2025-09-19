import React, {
  useState,
  useRef,
  useEffect,
  SetStateAction,
  Dispatch,
} from "react";
import { SpreadsheetData, Position } from "../types";
import { Sparkles, Trash2, Pencil, Plus } from "lucide-react";
import { motion } from "framer-motion";
import { ToastDetail } from "../App";
import SourcesTooltip from "./Tooltip";

interface SpreadsheetProps {
  data: SpreadsheetData;
  setData: Dispatch<SetStateAction<SpreadsheetData>>;
  setToast: Dispatch<SetStateAction<ToastDetail>>;
  apiKey: string;
}

// Add API URL from environment
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const Spreadsheet: React.FC<SpreadsheetProps> = ({
  setToast,
  data,
  setData,
  apiKey,
}) => {
  const [activeCell, setActiveCell] = useState<Position | null>(null);
  const [editValue, setEditValue] = useState("");
  const [isEditing, setIsEditing] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const tableRef = useRef<HTMLTableElement>(null);
  const [tooltipOpenCell, setTooltipOpenCell] = useState<{
    row: number;
    col: number;
  } | null>(null);

  // Add a new row
  const addRow = () => {
    const newRows = [...data.rows];
    newRows.push(Array(data.headers.length).fill({ value: "" }));
    setData({ ...data, rows: newRows });
  };

  // Add a new column
  const addColumn = () => {
    if (data.headers.length >= 5) return; // Limit to 5 columns

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
    if (tooltipOpenCell?.row === row && tooltipOpenCell?.col === col) {
      return; // If tooltip is open for this cell, don't trigger edit mode
    }

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

  // Enrichment function that calls our streaming API
  const enrichColumn = async (colIndex: number) => {
    if (!data.headers[colIndex]) {
      setToast({
        message: "Please set the column header",
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

    // Set all cells in the column to loading state at once
    const newRows = data.rows.map((row) => {
      const newRow = [...row];
      if (newRow[0].value?.length) {
        newRow[colIndex] = { ...newRow[colIndex], loading: true };
      }
      return newRow;
    });
    setData({ ...data, rows: newRows });

    try {
      // Prepare surgeons data with context from other columns
      const surgeons = data.rows
        .filter(row => row[0].value?.length) // Only include rows with names
        .map((row) => {
          const surgeon: any = { name: row[0].value };
          
          // Add context from other columns
          data.headers.forEach((header, idx) => {
            if (idx !== 0 && idx !== colIndex && header.trim() !== "") {
              const headerKey = header.toLowerCase().replace(/\s+/g, '_');
              if (row[idx]?.value) {
                surgeon[headerKey] = row[idx].value;
              }
            }
          });
          
          return surgeon;
        });

      const targetFields = [data.headers[colIndex]]; // Single field enrichment

      // Use fetch with streaming instead of EventSource since we need POST
      const response = await fetch(`${API_URL}/api/enrich-medical/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': apiKey ? `Bearer ${apiKey}` : '',
        },
        body: JSON.stringify({
          surgeons,
          target_fields: targetFields,
          provider: 'vertex'
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('Failed to get response reader');
      }

      let processedCount = 0;
      const totalCount = surgeons.length;
      let buffer = '';

      // Process the stream
      try {
        while (true) {
          const { done, value } = await reader.read();
          
          if (done) break;

          // Convert bytes to text and add to buffer
          const chunk = new TextDecoder().decode(value, { stream: true });
          buffer += chunk;

          // Process complete SSE messages in buffer
          let lines = buffer.split('\n');
          buffer = lines.pop() || ''; // Keep incomplete line in buffer

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const eventData = JSON.parse(line.slice(6)); // Remove 'data: ' prefix
                
                switch (eventData.type) {
                  case 'connected':
                    setToast({
                      message: "Starting enrichment...",
                      type: "info",
                      isShowing: true,
                    });
                    break;
                    
                  case 'field_complete':
                    // Update the specific cell immediately
                    const surgeonIdx = eventData.surgeon_idx;
                    const value = eventData.value || "";
                    const sources = eventData.sources || [];
                    
                    setData(prevData => {
                      const updatedRows = [...prevData.rows];
                      if (updatedRows[surgeonIdx]) {
                        updatedRows[surgeonIdx][colIndex] = {
                          value: value,
                          sources: sources,
                          enriched: value !== "" && value !== "Information not found",
                          loading: false,
                        };
                      }
                      return { ...prevData, rows: updatedRows };
                    });
                    
                    processedCount++;
                    
                    // Show progress
                    if (processedCount % 10 === 0 || processedCount === totalCount) {
                      setToast({
                        message: `Enriched ${processedCount}/${totalCount} cells...`,
                        type: "info",
                        isShowing: true,
                      });
                    }
                    break;
                    
                  case 'field_error':
                    // Handle individual field errors
                    const errorSurgeonIdx = eventData.surgeon_idx;
                    setData(prevData => {
                      const updatedRows = [...prevData.rows];
                      if (updatedRows[errorSurgeonIdx]) {
                        updatedRows[errorSurgeonIdx][colIndex] = {
                          value: "Error during enrichment",
                          sources: [],
                          enriched: false,
                          loading: false,
                        };
                      }
                      return { ...prevData, rows: updatedRows };
                    });
                    break;
                    
                  case 'complete':
                    setToast({
                      message: `Enrichment completed! Processed ${processedCount} cells`,
                      type: "success",
                      isShowing: true,
                    });
                    return; // Exit the processing loop
                }
              } catch (error) {
                console.error("Error parsing streaming data:", error);
              }
            }
          }
        }
      } catch (streamError) {
        console.error("Error reading stream:", streamError);
        throw streamError;
      } finally {
        reader.releaseLock();
      }

    } catch (error) {
      console.error("Error during enrichment:", error);
      // Reset loading state on error for all cells at once
      const errorRows = data.rows.map((row) => {
        const newRow = [...row];
        newRow[colIndex] = {
          ...newRow[colIndex],
          enriched: false,
          loading: false,
        };
        return newRow;
      });
      setData({ ...data, rows: errorRows });
      setToast({
        message: "Enrichment failed",
        type: "error",
        isShowing: true,
      });
    }
  };

  const enrichTable = async () => {
    if (
      !data.headers[0]?.trim() ||
      !data.headers.slice(1).some((header: string) => header?.trim().length > 0)
    ) {
      setToast({
        message: "Please set the first column and at least one other header",
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

    const loadingRows = data.rows.map((row) => {
      if (!row[0]?.value?.trim()) return row;

      return row.map((cell, colIndex) => {
        const hasHeader = data.headers[colIndex]?.trim();
        return hasHeader && colIndex !== 0 && !cell.enriched
          ? { ...cell, loading: true }
          : cell;
      });
    });

    setData({ ...data, rows: loadingRows });

    try {
      // Prepare payload for enrichment
      const requestData: Record<
        string,
        { rows: string[]; context_values: Record<string, string> }
      > = {};

      data.headers.forEach((header, colIndex) => {
        if (header && colIndex > 0) {
          const rows = data.rows.map((row) => row[0].value);
          const columnContext: Record<string, string> = {};

          // Generate context from other columns using row 0 as sample (or could build dynamic per-row later)
          data.headers.forEach((otherHeader, otherIdx) => {
            if (otherIdx !== colIndex && otherHeader.trim() !== "") {
              columnContext[otherHeader] = data.rows[0][otherIdx].value;
            }
          });

          requestData[header] = {
            rows,
            context_values: columnContext,
          };
        }
      });

      const response = await fetch(`${API_URL}/api/enrich-table`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(apiKey && { "Authorization": `Bearer ${apiKey}` }),
        },
        body: JSON.stringify({ data: requestData }),
      });

      if (!response.ok) {
        let errorMessage = "Table enrichment failed";
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorData.message || errorMessage;
        } catch (e) {
          // Use default message if JSON parsing fails
        }
        throw new Error(errorMessage);
      }

      const result = await response.json();

      // Update all cells with enriched values
      const updatedRows = data.rows.map((row, rowIndex) =>
        row.map((cell, colIndex) => {
          const colName = data.headers[colIndex];
          const enrichedValue =
            result.enriched_values?.[colName]?.[rowIndex] ?? "";
          const sources = result.sources?.[colName]?.[rowIndex] ?? [];

          return {
            value: enrichedValue || cell.value,
            sources,
            enriched: enrichedValue !== "",
            loading: false,
          };
        })
      );

      setData({ ...data, rows: updatedRows });
      setToast({ message: "Table enriched", type: "success", isShowing: true });
    } catch (error) {
      // Reset all loading states on error
      const errorRows = data.rows.map((row) =>
        row.map((cell) => ({ ...cell, loading: false }))
      );

      setData({ ...data, rows: errorRows });
      setToast({
        message: error instanceof Error ? error.message : "Table enrichment failed",
        type: "error",
        isShowing: true,
      });
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

  return (
    <motion.div
      className="w-full mb-40"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="w-full overflow-x-auto overflow-y-visible">
        <table
          ref={tableRef}
          className="min-w-full border-separate border-spacing-0"
        >
          <thead>
            <tr>
              <th className="w-14 bg-white border border-gray-200 p-2 sticky left-0 top-0 z-20 first-cell text-blue-500 hover:text-blue-700 cursor-pointer">
                <div
                  className="flex align-center justify-center w-full"
                  onClick={enrichTable}
                >
                  <Sparkles size={18} />
                </div>
              </th>
              {data.headers.map((header, index) => (
                                <th
                  key={index}
                  className={`w-56 max-w-[220px] bg-white border border-gray-200 p-2 text-left relative h-12 ${
                    editingHeader === index
                      ? "z-30"
                      : index === 0
                      ? "sticky left-14 z-10"
                      : ""
                  }`}
                >
                  <div className="flex justify-between items-center group">
                    <div className="flex items-center flex-1 min-w-0 mr-2">
                      {editingHeader === index ? (
                        <input
                          ref={headerInputRef}
                          type="text"
                          value={headerEditValue}
                          onChange={(e) => setHeaderEditValue(e.target.value)}
                          onBlur={saveHeaderEdit}
                          onKeyDown={handleHeaderKeyDown}
                          className="w-full text-sm bg-white font-medium outline-none focus:outline-none focus:ring-0 focus:border-transparent"
                          placeholder="Enter column name..."
                        />
                      ) : (
                        <div
                          className="flex items-center min-w-0 flex-1"
                          onClick={() => startEditingHeader(index)}
                        >
                          <span className="font-medium overflow-hidden text-ellipsis whitespace-nowrap flex-1">
                            {header}
                          </span>
                          <button
                            className="ml-2 text-gray-400 hover:text-blue-500 p-1 rounded-full hover:bg-blue-50 transition-colors flex-shrink-0"
                            title="Edit column name"
                          >
                            <Pencil size={14} />
                          </button>
                        </div>
                      )}
                    </div>
                    {index !== 0 && (
                      <div className="flex space-x-1 flex-shrink-0">
                        <button
                          className="text-red-500 hover:text-red-700 p-1 rounded-full hover:bg-red-50"
                          onClick={() => deleteColumn(index)}
                          title="Delete column"
                        >
                          <Trash2 size={14} />
                        </button>
                        <button
                          className="text-blue-500 hover:text-blue-700 p-1 rounded-full hover:bg-blue-50"
                          onClick={() => enrichColumn(index)}
                          title="Enrich column using Tavily"
                        >
                          <Sparkles size={14} />
                        </button>
                      </div>
                    )}
                  </div>
                </th>
              ))}
              {data.headers.length < 5 && (
                <th
                  className={`w-14 bg-white border border-gray-200 p-2 text-center ${
                    data.headers.length === 4 ? "last-header" : ""
                  }`}
                >
                  <button
                    onClick={addColumn}
                    className="w-8 h-8 rounded-full bg-gray-50 hover:bg-blue-50 text-gray-400 hover:text-blue-500 flex items-center justify-center transition-colors mx-auto"
                    title="Add column (max 5)"
                  >
                    <Plus size={16} />
                  </button>
                </th>
              )}
            </tr>
          </thead>
          <tbody>
            {data.rows.map((row, rowIndex) => (
              <tr
                key={rowIndex}
                className="hover:bg-blue-50/30 transition-colors"
              >
                <td
                  className={`bg-white border border-gray-200 p-2 text-center font-medium sticky left-0 ${
                    rowIndex === data.rows.length - 1
                      ? "last-row-first-cell"
                      : ""
                  }`}
                >
                  {rowIndex + 1}
                </td>
                {row.map((cell, colIndex) => (
                  <motion.td
                    key={colIndex}
                    className={`max-w-[150px] bg-white border border-gray-200 p-2 relative overflow-visible ${
                      rowIndex === data.rows.length - 1 &&
                      colIndex === row.length - 1 &&
                      data.headers.length === 5
                        ? "last-cell"
                        : ""
                    }`}
                    onClick={() => focusCell(rowIndex, colIndex)}
                    data-enriched={cell.enriched ? "true" : "false"}
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
                        className="w-full h-full min-h-6 flex items-center justify-between"
                        initial={
                          cell.enriched ? { opacity: 0 } : { opacity: 1 }
                        }
                        animate={{ opacity: 1 }}
                        transition={{ duration: 0.3 }}
                      >
                        {cell.loading ? (
                          <div className="flex items-center justify-left w-full">
                            <div className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                            <span className="ml-2 text-sm text-gray-500">
                              Enriching...
                            </span>
                          </div>
                        ) : (
                          <>
                            <span
                              className={`w-full
                              ${
                                cell.enriched
                                  ? "text-green-700 font-medium"
                                  : ""
                              }`}
                            >
                              {cell.value}
                            </span>

                            {cell.sources?.length ? (
                              <>
                                <SourcesTooltip
                                  sources={cell.sources}
                                  open={
                                    tooltipOpenCell?.row === rowIndex &&
                                    tooltipOpenCell?.col === colIndex
                                  }
                                  setOpen={(isOpen: boolean) => {
                                    console.log("isOpen", isOpen);
                                    setTooltipOpenCell(
                                      isOpen
                                        ? { row: rowIndex, col: colIndex }
                                        : null
                                    );
                                  }}
                                />
                              </>
                            ) : null}
                          </>
                        )}
                      </motion.div>
                    )}
                  </motion.td>
                ))}
                {data.headers.length < 5 && (
                  <td
                    className={`w-14 bg-white border border-gray-200 p-2 text-center ${
                      rowIndex === data.rows.length - 1 ? "last-cell" : ""
                    }`}
                  ></td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
        <div
          className="mt-2 bg-gray-100 p-0 text-gray-400 text-center rounded-md hover:bg-gray-200 cursor-pointer transition-colors"
          onClick={addRow}
        >
          <button
            className="w-8 h-8 flex items-center justify-center transition-colors mx-auto"
            title="Add row"
          >
            <Plus size={16} />
          </button>
        </div>
      </div>
    </motion.div>
  );
};

export default Spreadsheet;
