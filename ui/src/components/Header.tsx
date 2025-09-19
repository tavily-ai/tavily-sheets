import React, { useRef } from "react";
import { Download, Upload } from "lucide-react";
import { exportToCSV } from "../utils";
import { SpreadsheetData } from "../types";

interface HeaderProps {
  glassStyle: string;
  data: SpreadsheetData;
  onFileSelect: (file: File) => void;
}

const Header: React.FC<HeaderProps> = ({ glassStyle, data, onFileSelect }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageError = (
    e: React.SyntheticEvent<HTMLImageElement, Event>
  ) => {
    console.error("Failed to load Sylke banner");
    console.log("Image path:", e.currentTarget.src);
    e.currentTarget.style.display = "none";
  };

  const handleFileImport = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      onFileSelect(file);
      // Reset the input value to allow re-uploading the same file
      event.target.value = '';
    }
  };

  return (
    <div className="relative mb-16">
      <div className="text-center pt-4">
        <h1 className="text-[48px] font-medium text-[#1a202c] font-['DM_Sans'] tracking-[-1px] leading-[52px] text-center mx-auto antialiased">
          Surgeon Data Enrichment
        </h1>
        <p className="text-gray-600 text-lg font-['DM_Sans'] mt-4 flex items-center justify-center">
          <img
            src="/banner.png"
            alt="Sylke"
            className="h-6"
            onError={handleImageError}
          />
        </p>
      </div>
      <div className="absolute top-0 right-0 flex items-center space-x-2">
        {/* Hidden file input */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,.xlsx,.xls"
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />
        
        {/* Import button */}
        <button
          onClick={handleFileImport}
          className={`bg-[#468BFF] text-white hover:bg-[#8FBCFA] transition-colors rounded-lg flex items-center justify-center gap-2 text-sm`}
          style={{ width: "auto", height: "40px", padding: "8px 12px" }}
          aria-label="Import CSV or Excel file"
        >
          <Upload style={{ width: "16px", height: "auto" }} />
          Import
        </button>
        
        <button
          onClick={() => exportToCSV(data)}
          className={`bg-[#468BFF] text-white hover:bg-[#8FBCFA] transition-colors rounded-lg flex items-center justify-center gap-2 text-sm`}
          style={{ width: "auto", height: "40px", padding: "8px 12px" }}
          aria-label="Export to CSV"
        >
          <Download style={{ width: "16px", height: "auto" }} />
          Export
        </button>
      </div>
    </div>
  );
};

export default Header;
