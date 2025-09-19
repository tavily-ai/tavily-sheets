import React, { useState, useEffect } from "react";
import { X } from "lucide-react";

export interface ColumnMapping {
  [fileHeader: string]: string;
}

interface ColumnMappingModalProps {
  isOpen: boolean;
  onClose: () => void;
  fileHeaders: string[];
  targetFields: string[];
  onConfirm: (mapping: ColumnMapping) => void;
}

const ColumnMappingModal: React.FC<ColumnMappingModalProps> = ({
  isOpen,
  onClose,
  fileHeaders,
  targetFields,
  onConfirm,
}) => {
  const [mapping, setMapping] = useState<ColumnMapping>({});

  // Helper function to get display names for target fields
  const getFieldDisplayName = (field: string): string => {
    const displayNames: { [key: string]: string } = {
      'name': 'Name',
      'specialty': 'Medical Specialty',
      'subspecialty': 'Subspecialty',
      'email': 'Email Address',
      'phone': 'Phone Number',
      'hospital_name': 'Hospital/Institution',
      'address': 'Address',
      'credentials': 'Medical Credentials',
      'linkedin_url': 'LinkedIn URL',
      'influence_summary': 'Influence Summary',
      'strategic_summary': 'Strategic Summary',
      'additional_contacts': 'Additional Contacts',
      'publication_summary': 'Publication Summary',
      'personal_website': 'Personal Website'
    };
    return displayNames[field] || field;
  };

  // Smart defaulting logic - attempt to find best-fit matches
  useEffect(() => {
    if (fileHeaders.length > 0 && targetFields.length > 0) {
      const initialMapping: ColumnMapping = {};
      
      fileHeaders.forEach((header) => {
        const headerLower = header.toLowerCase().trim();
        
        // Enhanced matching logic for specific fields
        let match = targetFields.find(field => {
          const fieldLower = field.toLowerCase();
          
          // Exact matches
          if (fieldLower === headerLower) return true;
          
          // Smart field-specific matching
          if (fieldLower === 'publication_summary' && 
              (headerLower.includes('publication') || headerLower.includes('research') || 
               headerLower.includes('papers') || headerLower.includes('articles'))) return true;
               
          if (fieldLower === 'personal_website' && 
              (headerLower.includes('website') || headerLower.includes('web') || 
               headerLower.includes('personal') || headerLower.includes('site'))) return true;
               
          if (fieldLower === 'hospital_name' && 
              (headerLower.includes('hospital') || headerLower.includes('institution') || 
               headerLower.includes('medical center') || headerLower.includes('clinic'))) return true;
               
          // General matching
          return headerLower.includes(fieldLower) || fieldLower.includes(headerLower);
        });
        
        initialMapping[header] = match || "-- Ignore this column --";
      });
      
      setMapping(initialMapping);
    }
  }, [fileHeaders, targetFields]);

  const handleMappingChange = (fileHeader: string, targetField: string) => {
    setMapping(prev => ({
      ...prev,
      [fileHeader]: targetField
    }));
  };

  const handleConfirm = () => {
    // Filter out ignored columns
    const filteredMapping: ColumnMapping = {};
    Object.entries(mapping).forEach(([fileHeader, targetField]) => {
      if (targetField !== "-- Ignore this column --") {
        filteredMapping[fileHeader] = targetField;
      }
    });
    
    onConfirm(filteredMapping);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-xl max-w-2xl w-full max-h-[80vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <div>
            <h2 className="text-xl font-semibold text-gray-900">
              Map Columns
            </h2>
            <p className="text-sm text-gray-600 mt-1">
              Map your file columns to the target data fields
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            aria-label="Close modal"
          >
            <X className="w-5 h-5 text-gray-500" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-[50vh]">
          <div className="space-y-4">
            {fileHeaders.map((header) => (
              <div key={header} className="flex items-center gap-4">
                <div className="flex-1 min-w-0">
                  <label className="block text-sm font-medium text-gray-700 truncate">
                    {header}
                  </label>
                  <span className="text-xs text-gray-500">File column</span>
                </div>
                
                <div className="flex items-center text-gray-400">
                  →
                </div>
                
                <div className="flex-1">
                  <select
                    value={mapping[header] || "-- Ignore this column --"}
                    onChange={(e) => handleMappingChange(header, e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-[#468BFF] focus:border-transparent"
                  >
                    <option value="-- Ignore this column --">
                      -- Ignore this column --
                    </option>
                    {targetFields.map((field) => (
                      <option key={field} value={field}>
                        {getFieldDisplayName(field)}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 p-6 border-t border-gray-200 bg-gray-50">
          <button
            onClick={onClose}
            className="px-4 py-2 text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleConfirm}
            className="px-4 py-2 bg-[#468BFF] text-white rounded-lg hover:bg-[#8FBCFA] transition-colors"
          >
            Confirm and Import
          </button>
        </div>
      </div>
    </div>
  );
};

export default ColumnMappingModal;
