import { useState, useEffect } from "react";
import { Header, Spreadsheet, InfoPanel, ColumnMappingModal, ApiKeyInput } from "./components";
import { GlassStyle, SpreadsheetData, CellData } from "./types";
import { motion } from "framer-motion";
import Toast from "./components/Toast";
import { ColumnMapping } from "./components/ColumnMappingModal";
import Papa from 'papaparse';
import * as XLSX from 'xlsx';

const API_URL = import.meta.env.VITE_API_URL;
const WS_URL = import.meta.env.VITE_WS_URL;

// Target fields that the backend can process
const TARGET_FIELDS = [
  'name',
  'specialty',
  'subspecialty',
  'email',
  'phone',
  'hospital_name',
  'address',
  'credentials',
  'linkedin_url',
  'influence_summary',
  'strategic_summary',
  'additional_contacts',
  'publication_summary',
  'personal_website'
];

if (!API_URL || !WS_URL) {
  throw new Error(
    "Environment variables VITE_API_URL and VITE_WS_URL must be set"
  );
}

// Add this near the top of the file, after the imports
const writingAnimation = `
@keyframes writing {
  0% {
    stroke-dashoffset: 1000;
  }
  100% {
    stroke-dashoffset: 0;
  }
}

.animate-writing {
  animation: writing 1.5s linear infinite;
}
`;

// Add this right after the imports
const style = document.createElement("style");
style.textContent = writingAnimation;
document.head.appendChild(style);

// Add DM Sans font import
const dmSansStyle = document.createElement("style");
dmSansStyle.textContent = `
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600;9..40,700&display=swap');
  
  /* Apply DM Sans globally */
  body {
    font-family: 'DM Sans', sans-serif;
  }
`;
document.head.appendChild(dmSansStyle);

export type ToastDetail = {
  message?: string;
  type?: "success" | "error" | "info";
  isShowing?: boolean;
};

function App() {
  const [isInfoPanelOpen, setIsInfoPanelOpen] = useState<boolean>(true);
  const [toastDetail, setToastDetail] = useState<ToastDetail>({});
  const [apiKey, setApiKey] = useState<string>('');
  const [isApiKeyOpen, setIsApiKeyOpen] = useState<boolean>(true);
  
  // Separate storage for original data vs enriched data
  const [originalData, setOriginalData] = useState<SpreadsheetData>(() => {
    // Default data - always fresh on load
    return {
      headers: TARGET_FIELDS,
      rows: Array(5)
        .fill(0)
        .map(() => Array(TARGET_FIELDS.length).fill({ value: "" })),
    };
  });

  // Store only enriched cells separately
  const [enrichedCells, setEnrichedCells] = useState<Map<string, {value: string, sources: any[]}>>(() => {
    const saved = localStorage.getItem('tavily-sheets-enriched');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        return new Map(Object.entries(parsed));
      } catch (error) {
        console.error('Error parsing enriched data:', error);
      }
    }
    return new Map();
  });

  // Computed data that merges original + enriched
  const [data, setData] = useState<SpreadsheetData>(() => {
    return mergeOriginalWithEnriched(originalData, enrichedCells);
  });

  // Helper function to merge original data with enriched cells
  function mergeOriginalWithEnriched(original: SpreadsheetData, enriched: Map<string, any>): SpreadsheetData {
    const mergedRows = original.rows.map((row, rowIndex) => {
      return row.map((cell, colIndex) => {
        const cellKey = `${rowIndex}-${colIndex}`;
        const enrichedValue = enriched.get(cellKey);
        
        if (enrichedValue) {
          return {
            value: enrichedValue.value,
            enriched: true,
            sources: enrichedValue.sources || []
          };
        }
        
        return { ...cell, enriched: false };
      });
    });

    return {
      headers: original.headers,
      rows: mergedRows
    };
  }

  // Save only enriched cells to localStorage
  useEffect(() => {
    const enrichedObj = Object.fromEntries(enrichedCells);
    localStorage.setItem('tavily-sheets-enriched', JSON.stringify(enrichedObj));
  }, [enrichedCells]);

  // Update computed data when original or enriched changes
  useEffect(() => {
    setData(mergeOriginalWithEnriched(originalData, enrichedCells));
  }, [originalData, enrichedCells]);

  // New state for file import functionality
  const [isMappingModalOpen, setIsMappingModalOpen] = useState<boolean>(false);
  const [parsedInfo, setParsedInfo] = useState<{
    headers: string[];
    data: any[];
  } | null>(null);

  // Function to handle enrichment updates
  const handleEnrichmentUpdate = (rowIndex: number, colIndex: number, value: string, sources: any[] = []) => {
    const cellKey = `${rowIndex}-${colIndex}`;
    setEnrichedCells(prev => {
      const newMap = new Map(prev);
      if (value && value.trim() && value !== "Information not found") {
        newMap.set(cellKey, { value, sources });
      } else {
        newMap.delete(cellKey); // Remove if empty or not found
      }
      return newMap;
    });
  };

  // Function to clear enriched data
  const handleClearEnrichment = () => {
    setEnrichedCells(new Map());
  };

  // Function to handle manual cell updates
  const handleCellUpdate = (rowIndex: number, colIndex: number, value: string) => {
    setOriginalData(prev => {
      const newRows = [...prev.rows];
      newRows[rowIndex] = [...newRows[rowIndex]];
      newRows[rowIndex][colIndex] = { value, enriched: false };
      return { ...prev, rows: newRows };
    });
  };

  // Add these styles at the top of the component, before the return statement
  const glassStyle: GlassStyle = {
    base: "backdrop-filter backdrop-blur-lg bg-white/80 border border-gray-200 shadow-xl",
    card: "backdrop-filter backdrop-blur-lg bg-white/80 border border-gray-200 shadow-xl rounded-2xl p-6",
    input:
      "backdrop-filter backdrop-blur-lg bg-white/80 border border-gray-200 shadow-xl pl-10 w-full rounded-lg py-3 px-4 text-gray-900 focus:border-[#468BFF]/50 focus:outline-none focus:ring-1 focus:ring-[#468BFF]/50 placeholder-gray-400 bg-white/80 shadow-none",
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    
    reader.onload = (e) => {
      try {
        const result = e.target?.result;
        if (!result) return;

        let headers: string[];
        let parsedData: any[];

        if (file.name.endsWith('.csv')) {
          // Parse CSV
          const csvData = Papa.parse(result as string, {
            header: false,
            skipEmptyLines: true
          });
          
          if (csvData.errors.length > 0) {
            throw new Error(`CSV parsing error: ${csvData.errors[0].message}`);
          }
          
          headers = csvData.data[0] as string[];
          parsedData = csvData.data.slice(1).map((row: any) => {
            const obj: any = {};
            headers.forEach((header, index) => {
              obj[header] = row[index] || '';
            });
            return obj;
          });
        } else if (file.name.endsWith('.xlsx') || file.name.endsWith('.xls')) {
          // Parse Excel
          const workbook = XLSX.read(result, { type: 'binary' });
          const sheetName = workbook.SheetNames[0];
          const worksheet = workbook.Sheets[sheetName];
          const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 });
          
          if (jsonData.length === 0) {
            throw new Error('Excel file is empty');
          }
        
          headers = jsonData[0] as string[];
          parsedData = jsonData.slice(1).map((row: any[]) => {
            const obj: any = {};
            headers.forEach((header, index) => {
              obj[header] = row[index] || '';
            });
            return obj;
          });
        } else {
          throw new Error('Unsupported file type. Please upload CSV or Excel files.');
        }
        
        setParsedInfo({ headers, data: parsedData });
        setIsMappingModalOpen(true);
      } catch (error) {
        console.error('File processing error:', error);
        setToastDetail({
          message: error instanceof Error ? error.message : 'Failed to process file',
          type: 'error',
          isShowing: true
        });
      }
    };

    reader.readAsBinaryString(file);
  };

  const handleConfirmMapping = (mapping: ColumnMapping) => {
    try {
      if (!parsedInfo) {
        throw new Error('No file data available');
      }

      // Transform the data according to the mapping - ALWAYS include ALL target fields
      const transformedData: CellData[][] = parsedInfo.data.map((row) => {
        const transformedRow: CellData[] = [];
        TARGET_FIELDS.forEach((targetField) => {
          // Find which file header maps to this target field
          const fileHeader = Object.entries(mapping).find(([_, target]) => target === targetField)?.[0];
          const value = fileHeader ? (row[fileHeader] || '') : '';
          transformedRow.push({ value: String(value), enriched: false });
        });
        return transformedRow;
      });

      // Update the original data (this will trigger recomputation)
      setOriginalData({
        headers: TARGET_FIELDS,
        rows: transformedData
      });

      // Clear any existing enriched data since we're importing fresh
      setEnrichedCells(new Map());

      // Close modal and clear temporary state
      setIsMappingModalOpen(false);
      setParsedInfo(null);

      setToastDetail({
        message: `Successfully imported ${parsedInfo.data.length} rows with ${TARGET_FIELDS.length} columns`,
        type: 'success',
        isShowing: true
      });

    } catch (error) {
      console.error('Mapping error:', error);
      setToastDetail({
        message: error instanceof Error ? error.message : 'Failed to apply mapping',
        type: 'error',
        isShowing: true
      });
    }
  };

  return (
    <>
      <ApiKeyInput 
        isOpen={isApiKeyOpen}
        onSubmit={(key) => {
          setApiKey(key);
          setIsApiKeyOpen(false);
        }}
        onClose={() => setIsApiKeyOpen(false)}
      />

      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 p-4">
        <Header 
          glassStyle={glassStyle}
          onFileUpload={handleFileUpload}
          onClearData={handleClearEnrichment}
        />
        
        <motion.div
          className="max-w-7xl mx-auto mt-8"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          {isInfoPanelOpen && (
            <InfoPanel
              glassStyle={glassStyle.card}
              onDismiss={() => setIsInfoPanelOpen(false)}
            />
          )}

          <Spreadsheet
            data={data}
            onEnrichmentUpdate={handleEnrichmentUpdate}
            onCellUpdate={handleCellUpdate}
            setToast={setToastDetail}
            apiKey={apiKey}
          />
        </motion.div>
      </div>

      {/* Column Mapping Modal */}
      {parsedInfo && (
        <ColumnMappingModal
          isOpen={isMappingModalOpen}
          onClose={() => {
            setIsMappingModalOpen(false);
            setParsedInfo(null);
          }}
          fileHeaders={parsedInfo.headers}
          targetFields={TARGET_FIELDS}
          onConfirm={handleConfirmMapping}
        />
      )}

      {/* Toast Notification */}
      <Toast toastDetail={toastDetail} setToastDetail={setToastDetail} />
    </>
  );
}

export default App;