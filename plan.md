# Plan: Implement CSV/Excel File Import with Dynamic Column Mapping

This document provides a detailed, step-by-step implementation plan for a coder agent. The objective is to introduce a file import feature that supports CSV and Excel files, with a crucial user-facing step for dynamic column mapping.

## Guiding Principles

-   **Client-Centric**: The entire file reading, parsing, and mapping process will be handled on the client-side to provide a fast, interactive, and seamless user experience. No backend changes are needed for this feature.
-   **Component-Based**: New functionality will be encapsulated in new, reusable components.
-   **State-Driven**: The UI will reactively update based on a centralized state managed in the main `App.tsx` component.

---

## Phase 1: Project Setup and Dependencies

**Objective:** Prepare the frontend environment by adding the necessary third-party libraries for file parsing.

-   **Step 1.1: Modify `package.json`:**
    -   Locate and open the `package.json` file within the `ui` directory.
    -   Add `papaparse` and `xlsx` to the `dependencies` section. These are essential for parsing CSV and Excel files, respectively.
    -   Add `@types/papaparse` to the `devDependencies` section to ensure TypeScript support.

-   **Step 1.2: Install Dependencies:**
    -   Execute `npm install` within the `ui` directory to download and install the newly added packages into the project.

---

## Phase 2: UI Component Implementation

**Objective:** Create the user-facing elements for file import and column mapping.

-   **Step 2.1: Enhance the `Header` Component (`ui/src/components/Header.tsx`)**
    -   **Props Interface:** Modify the `HeaderProps` interface to accept a new callback function prop named `onFileSelect`. This function will be invoked with the selected `File` object.
    -   **File Input:** Add a hidden `<input type="file">` element within the component's render method. Configure its `accept` attribute to allow only CSV and Excel file types (`.csv`, `.xlsx`, `.xls`).
    -   **Import Button:** Add a new "Import" button to the UI, visually grouping it with the existing "Export" button. This button should be styled consistently with the application's theme and include an `Upload` icon from the `lucide-react` library.
    -   **Event Handling:** Create a click handler for the "Import" button that programmatically triggers a click on the hidden file input. The file input's `onChange` event should call the `onFileSelect` prop with the selected file and then reset its own value to allow for re-uploading the same file.

-   **Step 2.2: Create the `ColumnMappingModal` Component**
    -   **File Creation:** Create a new component file at `ui/src/components/ColumnMappingModal.tsx`.
    -   **Props Interface:** Define the component's props. It must accept:
        -   `isOpen`: A boolean to control the modal's visibility.
        -   `onClose`: A callback function to handle closing the modal.
        -   `fileHeaders`: An array of strings from the parsed uploaded file.
        -   `targetFields`: An array of strings representing the valid fields the application can enrich.
        -   `onConfirm`: A callback function that returns the user-defined mapping object.
    -   **UI Layout:** The component should render a modal dialog that overlays the main UI. Inside, it should display a title, instructional text, and a scrollable list.
    -   **Mapping UI:** For each header in the `fileHeaders` prop, render a row containing the header's name and a corresponding `<select>` dropdown menu.
    -   **Dropdown Population:** Each dropdown should be populated with all available `targetFields`, plus a default "-- Ignore this column --" option.
    -   **Smart Defaulting Logic:** Implement a `useEffect` hook that runs when the component receives `fileHeaders`. This hook should create an initial mapping state by attempting to find a "best-fit" match for each file header from the `targetFields` list (e.g., by case-insensitive comparison). This provides a smart default for the user.
    -   **Action Buttons:** Include "Cancel" and "Confirm and Import" buttons. `Cancel` should trigger the `onClose` prop. `Confirm` should finalize the current mapping state and pass it to the `onConfirm` prop.

---

## Phase 3: Application Logic and State Orchestration

**Objective:** Integrate the new UI components and file-processing logic into the main application.

-   **Step 3.1: Update the Main `App` Component (`ui/src/App.tsx`)**
    -   **Imports:** Import the new `ColumnMappingModal` component and the `papaparse` and `xlsx` libraries.
    -   **Constants:** Define a constant array named `TARGET_FIELDS` that contains the canonical list of all fields the backend can process (e.g., 'name', 'specialty', 'email'). This list will be passed to the mapping modal.
    -   **State Management:** Introduce new state variables to manage the mapping modal's visibility (`isMappingModalOpen`) and to temporarily store the parsed file's headers and data (`parsedInfo`).

-   **Step 3.2: Implement the `onFileSelect` Handler**
    -   Create this function within `App.tsx`. It will be passed down to the `Header` component.
    -   This function will use a `FileReader` to read the selected file's content.
    -   It must differentiate between CSV and Excel files based on their file extension.
    -   Use the appropriate library (`papaparse` or `xlsx`) to parse the file content into a structured format (an array of objects is preferred).
    -   Upon successful parsing, update the `parsedInfo` state with the headers and data from the file, and set the state to open the `ColumnMappingModal`.
    -   Implement robust error handling to catch parsing errors and display a user-friendly toast notification.

-   **Step 3.3: Implement the `handleConfirmMapping` Handler**
    -   Create this function within `App.tsx`. It will be passed to the `ColumnMappingModal`.
    -   This function receives the final mapping object from the modal.
    -   It must then transform the data stored in the `parsedInfo` state into the `SpreadsheetData` format required by the main `Spreadsheet` component. This involves creating new `headers` and `rows` arrays according to the user's mapping.
    -   After the transformation, update the application's primary `data` state with the new spreadsheet data, close the modal, and clear the temporary `parsedInfo` state.

-   **Step 3.4: Render the Modal**
    -   In the JSX of the `App` component, conditionally render the `ColumnMappingModal` based on the `isMappingModalOpen` state variable. Ensure all required props (headers, target fields, callbacks, etc.) are passed to it.