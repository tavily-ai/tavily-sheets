import React, {
  useState,
  useRef,
  useEffect,
  SetStateAction,
  Dispatch,
} from "react";
import { SpreadsheetData, Position, ColumnConfig } from "../types";
import { Sparkles, Trash2, Pencil, Plus } from "lucide-react";
import { motion } from "framer-motion";
import { ToastDetail } from "../App";
import SourcesTooltip from "./Tooltip";

interface SpreadsheetProps {
  data: SpreadsheetData;
  setData: Dispatch<SetStateAction<SpreadsheetData>>;
  setToast: Dispatch<SetStateAction<ToastDetail>>;
  apiKey: string;
  checkApiKey: () => boolean | 0 | undefined;
}

// Add API URL from environment
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// Add at the top of the Spreadsheet component
const ENRICHMENT_TYPES = [
  { value: "predefined", label: "Predefined (e.g. Find Website)" },
  { value: "ai_agent", label: "AI Agent (custom chain)" },
];

// Helper to create a default column config
const defaultColumnConfig = (name = ""): ColumnConfig => ({
  name,
  enrichmentType: "predefined",
});

const Spreadsheet: React.FC<SpreadsheetProps> = ({
  setToast,
  data,
  setData,
  apiKey,
  checkApiKey,
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

  // Modal state for column config
  const [configModalOpen, setConfigModalOpen] = useState(false);
  const [configModalIndex, setConfigModalIndex] = useState<number | null>(null);
  const [configForm, setConfigForm] = useState<ColumnConfig>({ name: "" });

  // Add a new row
  const addRow = () => {
    const newRows = [...data.rows];
    newRows.push(Array(data.headers.length).fill({ value: "" }));
    setData({ ...data, rows: newRows });
  };

  // Update addColumn to use defaultColumnConfig
  const addColumn = () => {
    if (data.headers.length >= 5) return; // Limit to 5 columns
    const newHeaders = [...data.headers, defaultColumnConfig()];
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

  // Enrichment function that calls our API
  const enrichColumn = async (colIndex: number) => {
    const colConfig = data.headers[colIndex];
    if (!colConfig.name) {
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

    if (!checkApiKey()) {
      setToast({
        message: "Please set a valid API Key",
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
      // Get context from other columns
      const contextValues: Record<string, string> = {};
      data.headers.forEach((header, idx) => {
        const headerName = header?.name;
        if (
          idx !== colIndex &&
          typeof headerName === "string" &&
          headerName.trim() !== ""
        ) {
          contextValues[headerName] = data.rows[0][idx].value;
        }
      });

      // Extract all target values from first column
      const targetValues = data.rows.map((row) => row[0].value);

      // Prepare payload for each row
      const rowsPayload = data.rows.map((row, rowIndex) => {
        // Default: use entity from first column
        let input_source_type = "ENTITY";
        let input_data = row[0].value;
        let custom_prompt = undefined;
        if (colConfig.enrichmentType === "ai_agent") {
          input_source_type = "TEXT_FROM_COLUMN";
          if (
            typeof colConfig.mappedColumnIndex === "number" &&
            data.headers[colConfig.mappedColumnIndex]
          ) {
            input_data = row[colConfig.mappedColumnIndex].value;
          }
          custom_prompt = colConfig.customPrompt;
        }
        return {
          column_name: colConfig.name,
          target_value: row[0].value,
          context_values: contextValues,
          input_source_type,
          input_data,
          custom_prompt,
        };
      });

      // Make a single batch request
      const response = await fetch(`${API_URL}/api/enrich/batch`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: apiKey,
        },
        body: JSON.stringify({
          column_name: colConfig.name,
          rows: data.rows.map(row => row[0].value),
          context_values: contextValues
        }),
      });

      if (!response.ok) {
        setToast({
          message: "Enrichment failed",
          type: "error",
          isShowing: true,
        });
        throw new Error("Batch enrichment failed");
      }

      const result = await response.json();

      // Update all cells at once with the enriched values
      const enrichedRows = data.rows.map((row, rowIndex) => {
        const newRow = [...row];
        newRow[colIndex] = {
          value: result.enriched_values[rowIndex],
          sources: result.sources[rowIndex],
          enriched: result.enriched_values[rowIndex] !== "",
          loading: false,
        };
        return newRow;
      });

      setData({ ...data, rows: enrichedRows });
      setToast({ message: "Cells enriched", type: "success", isShowing: true });
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
      !data.headers[0]?.name ||
      !data.headers.slice(1).some((header: ColumnConfig) => header?.name.trim().length > 0)
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

    if (!checkApiKey()) {
      setToast({
        message: "Please set a valid API Key",
        type: "error",
        isShowing: true,
      });
      return;
    }

    const loadingRows = data.rows.map((row) => {
      if (!row[0]?.value?.trim()) return row;

      return row.map((cell, colIndex) => {
        const headerName = data.headers[colIndex]?.name;
        const hasHeader = typeof headerName === "string" && headerName.trim() !== "";
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
            const otherHeaderName = otherHeader?.name;
            if (
              otherIdx !== colIndex &&
              typeof otherHeaderName === "string" &&
              otherHeaderName.trim() !== ""
            ) {
              columnContext[otherHeaderName] = data.rows[0][otherIdx].value;
            }
          });

          requestData[header.name] = {
            rows,
            context_values: columnContext,
          };
        }
      });

      const response = await fetch(`${API_URL}/api/enrich-table`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: apiKey,
        },
        body: JSON.stringify({ data: requestData }),
      });

      if (!response.ok) {
        throw new Error("Table enrichment failed");
      }

      const result = await response.json();

      // Update all cells with enriched values
      const updatedRows = data.rows.map((row, rowIndex) =>
        row.map((cell, colIndex) => {
          const colName = data.headers[colIndex];
          const enrichedValue =
            result.enriched_values?.[colName.name]?.[rowIndex] ?? "";
          const sources = result.sources?.[colName.name]?.[rowIndex] ?? [];

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
        message: "Table enrichment failed",
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
    setHeaderEditValue(data.headers[index].name);
  };

  const saveHeaderEdit = () => {
    if (editingHeader !== null) {
      const newHeaders = [...data.headers];
      newHeaders[editingHeader] = { ...data.headers[editingHeader], name: headerEditValue };
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

  // Open the config modal for a column
  const openConfigModal = (index: number) => {
    setConfigModalIndex(index);
    setConfigForm({ ...data.headers[index] });
    setConfigModalOpen(true);
  };

  // Save the config changes
  const saveConfigModal = () => {
    if (configModalIndex !== null) {
      const newHeaders = [...data.headers];
      newHeaders[configModalIndex] = { ...configForm };
      setData({ ...data, headers: newHeaders });
      setConfigModalOpen(false);
      setConfigModalIndex(null);
    }
  };

  // Handle config form changes
  const handleConfigChange = (field: keyof ColumnConfig, value: any) => {
    setConfigForm((prev) => ({ ...prev, [field]: value }));
  };

  return (
    <motion.div
      className="w-full mb-40"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="w-full">
        <table
          ref={tableRef}
          className="w-full border-separate border-spacing-0"
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
                  className={`w-40 max-w-[150px] bg-white border border-gray-200 p-2 text-left relative h-12 ${
                    index === data.headers.length - 1 &&
                    data.headers.length === 5
                      ? "last-header"
                      : ""
                  }`}
                >
                  <div className="flex justify-between items-center group">
                    <div className="flex items-center w-full">
                      {editingHeader === index ? (
                        <input
                          ref={headerInputRef}
                          type="text"
                          value={headerEditValue ?? ""}
                          onChange={(e) => setHeaderEditValue(e.target.value)}
                          onBlur={saveHeaderEdit}
                          onKeyDown={handleHeaderKeyDown}
                          className="w-full text-sm bg-white font-medium outline-none focus:outline-none focus:ring-0 focus:border-transparent"
                          placeholder="Enter column name..."
                        />
                      ) : (
                        <div
                          className="flex items-center max-w-[140px]"
                          onClick={() => openConfigModal(index)}
                        >
                          <span className="font-medium overflow-hidden text-ellipsis whitespace-nowrap">
                            {header.name}
                          </span>
                          <button
                            className="ml-2 text-gray-400 hover:text-blue-500 p-1 rounded-full hover:bg-blue-50 transition-colors"
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
                        value={editValue ?? ""}
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
      {configModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-30">
          <div className="bg-white rounded-lg shadow-lg p-6 w-full max-w-md">
            <h2 className="text-lg font-semibold mb-4">Configure Column</h2>
            <div className="mb-3">
              <label className="block text-sm font-medium mb-1">Column Name</label>
              <input
                type="text"
                className="w-full border rounded px-2 py-1"
                value={configForm.name ?? ""}
                onChange={(e) => handleConfigChange("name", e.target.value)}
              />
            </div>
            <div className="mb-3">
              <label className="block text-sm font-medium mb-1">Enrichment Type</label>
              <select
                className="w-full border rounded px-2 py-1"
                value={configForm.enrichmentType || "predefined"}
                onChange={(e) => handleConfigChange("enrichmentType", e.target.value)}
              >
                {ENRICHMENT_TYPES.map((type) => (
                  <option key={type.value} value={type.value}>{type.label}</option>
                ))}
              </select>
            </div>
            {configForm.enrichmentType === "ai_agent" && (
              <>
                <div className="mb-3">
                  <label className="block text-sm font-medium mb-1">Map Input From Column</label>
                  <select
                    className="w-full border rounded px-2 py-1"
                    value={configForm.mappedColumnIndex ?? ""}
                    onChange={(e) => handleConfigChange("mappedColumnIndex", e.target.value === "" ? undefined : Number(e.target.value))}
                  >
                    <option value="">Select column</option>
                    {data.headers.map((header, idx) =>
                      idx !== configModalIndex ? (
                        <option key={idx} value={idx}>{header.name || `Column ${idx + 1}`}</option>
                      ) : null
                    )}
                  </select>
                </div>
                <div className="mb-3">
                  <label className="block text-sm font-medium mb-1">Custom Prompt</label>
                  <textarea
                    className="w-full border rounded px-2 py-1"
                    value={configForm.customPrompt ?? ""}
                    onChange={(e) => handleConfigChange("customPrompt", e.target.value)}
                    rows={3}
                    placeholder="e.g. Using the website from Column B, who is their target audience?"
                  />
                </div>
              </>
            )}
            <div className="flex justify-end gap-2 mt-4">
              <button
                className="px-4 py-2 rounded bg-gray-200 hover:bg-gray-300"
                onClick={() => setConfigModalOpen(false)}
              >
                Cancel
              </button>
              <button
                className="px-4 py-2 rounded bg-blue-600 text-white hover:bg-blue-700"
                onClick={saveConfigModal}
              >
                Save
              </button>
            </div>
          </div>
        </div>
      )}
    </motion.div>
  );
};

export default Spreadsheet;
