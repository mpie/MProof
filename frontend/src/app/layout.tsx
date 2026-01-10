import type { Metadata } from "next";
import { Inter } from "next/font/google";
import Link from "next/link";
import "./globals.css";
import { QueryProvider } from "@/components/QueryProvider";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { config, library } from "@fortawesome/fontawesome-svg-core";
import {
  faRobot, faFileShield, faPlus, faEdit, faTrash, faSave, faTimes, faCog,
  faExclamationTriangle, faCloudUploadAlt, faFile, faCheck, faUser, faBuilding,
  faFolder, faQuestion, faSearch, faEye, faDownload, faRedo, faInfoCircle,
  faHighlighter, faCopy, faChevronDown, faChevronUp, faSpinner, faFilePdf,
  faFileImage, faFileWord, faFileExcel, faHome, faFileAlt, faList, faBook,
  faExternalLinkAlt
} from "@fortawesome/free-solid-svg-icons";
import "@fortawesome/fontawesome-svg-core/styles.css";
import { Navigation } from "@/components/Navigation";

// Add all commonly used icons to the library to prevent layout shifts
library.add(
  faRobot, faFileShield, faPlus, faEdit, faTrash, faSave, faTimes, faCog,
  faExclamationTriangle, faCloudUploadAlt, faFile, faCheck, faUser, faBuilding,
  faFolder, faQuestion, faSearch, faEye, faDownload, faRedo, faInfoCircle,
  faHighlighter, faCopy, faChevronDown, faChevronUp, faSpinner, faFilePdf,
  faFileImage, faFileWord, faFileExcel, faHome, faFileAlt, faList, faBook,
  faExternalLinkAlt
);

// Prevent FontAwesome from adding its CSS since we did it manually above
config.autoAddCss = false;

// Prevent layout shifts by ensuring icons are rendered consistently
config.autoReplaceSvg = 'nest';

// Ensure icons are available immediately to prevent layout shifts
config.measurePerformance = false;

// Disable async loading to prevent layout shifts
config.observeMutations = false;

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "MProof - Document Analysis System",
  description: "Upload, analyze, and manage documents with AI-powered processing",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.variable} antialiased`}>
        <QueryProvider>
          <div className="min-h-screen" style={{ background: 'linear-gradient(135deg, #0f766e 0%, #0e7490 30%, #1e40af 70%, #1e3a8a 100%)' }}>
            <header className="bg-white/10 backdrop-blur-xl border-b border-white/15 shadow-xl">
              <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex items-center justify-between h-16">
                  <Link
                    href="/"
                    className="flex items-center space-x-3 hover:opacity-80 transition-opacity"
                    title="Go to Dashboard"
                  >
                    <div className="relative w-10 h-10 flex items-center justify-center rounded-lg bg-gradient-to-br from-teal-500 via-cyan-500 to-blue-500 p-2 shadow-lg">
                      <FontAwesomeIcon
                        icon={faFileShield}
                        className="text-white text-lg"
                      />
                    </div>
                    <div className="flex flex-col">
                      <span className="text-white text-xl font-bold tracking-tight">MProof</span>
                      <span className="text-white/70 text-xs font-normal">Document Analysis</span>
                    </div>
                  </Link>
                  <div className="flex items-center space-x-4">
                    <OllamaStatus />
                    <Navigation />
                  </div>
                </div>
              </div>
            </header>
            <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
              {children}
            </main>
          </div>
        </QueryProvider>
      </body>
    </html>
  );
}

function OllamaStatus() {
  // This will be implemented with real health check
  return (
    <div className="flex items-center space-x-2 text-white">
      <FontAwesomeIcon icon={faRobot} className="text-green-400" />
      <span className="text-sm">Ollama: Online</span>
    </div>
  );
}
