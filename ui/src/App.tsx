import { useState, useEffect } from "react";
import { Header, Spreadsheet, InfoPanel, ColumnMappingModal, ApiKeyInput } from "./components";
import { GlassStyle, SpreadsheetData, CellData } from "./types";
import { motion } from "framer-motion";
import Toast from "./components/Toast";
import { ColumnMapping } from "./components/ColumnMappingModal";
import Papa from 'papaparse';
import * as XLSX from 'xlsx';
import { dataCache } from './utils/dataCache';

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
  
  // Initialize default data
  const defaultData: SpreadsheetData = {
    headers: TARGET_FIELDS,
    rows: Array(5)
      .fill(0)
      .map(() => Array(TARGET_FIELDS.length).fill({ value: "" })),
  };
  
  // Initialize data FRESH every time; don't auto-restore from cache
  const [data, setData] = useState<SpreadsheetData>(defaultData);

  // Restore prompt state (user-controlled cache)
  const [showRestorePrompt, setShowRestorePrompt] = useState<boolean>(false);
  const [restoreCandidate, setRestoreCandidate] = useState<{
    key: string;
    data: SpreadsheetData;
    timestamp: number;
  } | null>(null);

  // Enhanced setData that automatically caches
  const setDataWithCache = (newData: SpreadsheetData | ((prevState: SpreadsheetData) => SpreadsheetData)) => {
    const resolvedData = typeof newData === 'function' ? newData(data) : newData;
    setData(resolvedData);
    
    // Auto-save to cache when data changes (debounced)
    setTimeout(() => {
      dataCache.saveToCache(resolvedData);
    }, 1000);
  };

  // On mount, check if a valid cache exists and ask user whether to restore
  useEffect(() => {
    // Effects run client-side only; safe to access localStorage
    const latest = dataCache.getLatest();
    if (latest) {
      setRestoreCandidate(latest);
      setShowRestorePrompt(true);
    }
  }, []);

  // New state for file import functionality
  const [isMappingModalOpen, setIsMappingModalOpen] = useState<boolean>(false);
  const [parsedInfo, setParsedInfo] = useState<{
    headers: string[];
    data: any[];
  } | null>(null);

  // Add these styles at the top of the component, before the return statement
  const glassStyle: GlassStyle = {
    base: "backdrop-filter backdrop-blur-lg bg-white/80 border border-gray-200 shadow-xl",
    card: "backdrop-filter backdrop-blur-lg bg-white/80 border border-gray-200 shadow-xl rounded-2xl p-6",
    input:
      "backdrop-filter backdrop-blur-lg bg-white/80 border border-gray-200 shadow-xl pl-10 w-full rounded-lg py-3 px-4 text-gray-900 focus:border-[#468BFF]/50 focus:outline-none focus:ring-1 focus:ring-[#468BFF]/50 placeholder-gray-400 bg-white/80 shadow-none",
  };

  const handleFileSelect = async (file: File) => {
    try {
      const fileExtension = file.name.toLowerCase().split('.').pop();
      let headers: string[] = [];
      let parsedData: any[] = [];

      if (fileExtension === 'csv') {
        // Parse CSV file
        Papa.parse(file, {
          header: true,
          skipEmptyLines: true,
          complete: (results) => {
            if (results.errors.length > 0) {
              throw new Error('CSV parsing error: ' + results.errors[0].message);
            }
            headers = results.meta.fields || [];
            parsedData = results.data as any[];
            setParsedInfo({ headers, data: parsedData });
            setIsMappingModalOpen(true);
          },
          error: (error) => {
            throw new Error('Failed to parse CSV: ' + error.message);
          }
        });
      } else if (fileExtension === 'xlsx' || fileExtension === 'xls') {
        // Parse Excel file
        const arrayBuffer = await file.arrayBuffer();
        const workbook = XLSX.read(arrayBuffer, { type: 'array' });
        const firstSheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[firstSheetName];
        
        // Convert to JSON with header row
        const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 }) as any[][];
        
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
        
        setParsedInfo({ headers, data: parsedData });
        setIsMappingModalOpen(true);
      } else {
        throw new Error('Unsupported file type. Please upload CSV or Excel files.');
      }
      
    } catch (error) {
      console.error('File processing error:', error);
      setToastDetail({
        message: error instanceof Error ? error.message : 'Failed to process file',
        type: 'error',
        isShowing: true
      });
    }
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

      // Update the spreadsheet with the transformed data - ALL target fields included
      // Update state with new spreadsheet data
      setDataWithCache({
        headers: TARGET_FIELDS,
        rows: transformedData
      });

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
        message: error instanceof Error ? error.message : 'Failed to process mapping',
        type: 'error',
        isShowing: true
      });
    }
  };

  return (
    <div className="min-h-screen bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-white via-gray-50 to-white p-8 relative overflow-hidden">
      {/* Enhanced background with multiple layers */}
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_1px_1px,rgba(70,139,255,0.35)_1px,transparent_0)] bg-[length:24px_24px] bg-center"></div>

      {toastDetail.isShowing && (
        <Toast
          message={toastDetail.message}
          type={toastDetail.type}
          onClose={() => setToastDetail({})}
        />
      )}

      {/* Unsaved session restore banner */}
      {showRestorePrompt && restoreCandidate && (
        <div className="fixed top-5 left-1/2 -translate-x-1/2 z-50">
          <div className="glass border border-gray-300/60 bg-white/90 text-gray-900 rounded-xl shadow-lg px-4 py-3 flex items-center gap-3">
            <span className="text-sm">
              You have an unsaved session from {new Date(restoreCandidate.timestamp).toLocaleString()}. 
            </span>
            <button
              onClick={() => {
                setDataWithCache(restoreCandidate.data);
                setShowRestorePrompt(false);
                setToastDetail({
                  message: 'Session restored',
                  type: 'success',
                  isShowing: true
                });
              }}
              className="text-sm px-3 py-1 rounded-md bg-[#468BFF] text-white hover:bg-[#8FBCFA] transition-colors"
            >
              Restore Session
            </button>
            <button
              onClick={() => {
                if (restoreCandidate) {
                  dataCache.deleteCacheKey(restoreCandidate.key);
                }
                setShowRestorePrompt(false);
                setRestoreCandidate(null);
                setToastDetail({
                  message: 'Starting fresh — old session discarded',
                  type: 'info',
                  isShowing: true
                });
              }}
              className="text-sm px-3 py-1 rounded-md bg-gray-200 text-gray-800 hover:bg-gray-300 transition-colors"
            >
              Start Fresh
            </button>
          </div>
        </div>
      )}

      {/* Add floating gradient orbs for visual interest */}
      <motion.div
        className="absolute top-1/4 right-1/4 w-64 h-64 rounded-full bg-gradient-to-br from-blue-300/20 to-purple-300/10 blur-3xl pointer-events-none"
        animate={{
          y: [0, -15, 0],
          x: [0, 10, 0],
          scale: [1, 1.05, 1],
        }}
        transition={{
          duration: 10,
          repeat: Infinity,
          repeatType: "reverse",
        }}
      />
      <motion.div
        className="absolute bottom-1/3 left-1/4 w-48 h-48 rounded-full bg-gradient-to-tr from-green-300/10 to-blue-300/20 blur-3xl pointer-events-none"
        animate={{
          y: [0, 20, 0],
          x: [0, -15, 0],
          scale: [1, 1.1, 1],
        }}
        transition={{
          duration: 15,
          repeat: Infinity,
          repeatType: "reverse",
        }}
      />

      {/* Add a subtle animated glow around the main content */}
      <motion.div
        className="absolute inset-0 mx-auto max-w-7xl h-full bg-gradient-to-b from-blue-50/10 to-purple-50/10 blur-3xl rounded-[40px] pointer-events-none"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1.5 }}
      />

      <div className="max-w-7xl mx-auto space-y-8 relative">
        {/* API Key Input Component */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <ApiKeyInput
            apiKey={apiKey}
            setApiKey={setApiKey}
            isOpen={isApiKeyOpen}
            setIsOpen={setIsApiKeyOpen}
            glassStyle={glassStyle.card}
          />
        </motion.div>

        {/* Header Component */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
        >
          <Header glassStyle={glassStyle.card} data={data} onFileSelect={handleFileSelect} />
        </motion.div>

        {/* Spreadsheet Component */}
        <motion.div
          className="relative"
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
            setData={setDataWithCache}
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
    </div>
  );
}

export default App;
