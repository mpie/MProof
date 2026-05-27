import type { Metadata } from "next";
import { Inter } from "next/font/google";
import Link from "next/link";
import "./globals.css";
import { QueryProvider } from "@/components/QueryProvider";
import { ModelProvider } from "@/context/ModelContext";
import { AuthProvider } from "@/context/AuthContext";
import { UserMenu } from "@/components/UserMenu";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { config, library } from "@fortawesome/fontawesome-svg-core";
import {
  faRobot, faFileShield, faPlus, faEdit, faTrash, faSave, faTimes, faCog,
  faExclamationTriangle, faCloudUploadAlt, faFile, faCheck, faUser, faBuilding,
  faFolder, faQuestion, faSearch, faEye, faDownload, faRedo, faInfoCircle,
  faHighlighter, faCopy, faChevronDown, faChevronUp, faSpinner, faFilePdf,
  faFileImage, faFileWord, faFileExcel, faHome, faFileAlt, faList, faBook,
  faExternalLinkAlt, faUsers, faCrown, faShieldAlt, faToggleOn, faToggleOff,
  faSignOutAlt, faKey, faEyeSlash, faStop,
} from "@fortawesome/free-solid-svg-icons";
import "@fortawesome/fontawesome-svg-core/styles.css";
import { Navigation } from "@/components/Navigation";
import { LLMStatus } from "@/components/LLMStatus";

// Add all commonly used icons to the library to prevent layout shifts
library.add(
  faRobot, faFileShield, faPlus, faEdit, faTrash, faSave, faTimes, faCog,
  faExclamationTriangle, faCloudUploadAlt, faFile, faCheck, faUser, faBuilding,
  faFolder, faQuestion, faSearch, faEye, faDownload, faRedo, faInfoCircle,
  faHighlighter, faCopy, faChevronDown, faChevronUp, faSpinner, faFilePdf,
  faFileImage, faFileWord, faFileExcel, faHome, faFileAlt, faList, faBook,
  faExternalLinkAlt, faUsers, faCrown, faShieldAlt, faToggleOn, faToggleOff,
  faSignOutAlt, faKey, faEyeSlash, faStop,
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
      <body className={`${inter.variable} antialiased min-w-[320px]`}>
        <QueryProvider>
          <AuthProvider>
            <ModelProvider>
              <div className="min-h-screen min-w-[320px]" style={{ background: 'var(--background)' }}>
                <header className="section-header backdrop-blur-xl relative z-[100]">
                  <div className="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="flex items-center justify-between h-14 sm:h-16">
                      <Link
                        href="/"
                        className="flex items-center gap-2.5 hover:opacity-80 transition-opacity"
                        title="Dashboard"
                      >
                        <div className="w-9 h-9 flex items-center justify-center rounded-xl bg-gradient-to-br from-[#FFC1F3] via-[#f5b8f0] to-[#FCE2CE] shadow-[0_2px_12px_rgba(255,193,243,0.55)] border border-white/40">
                          <FontAwesomeIcon icon={faFileShield} className="text-white" style={{ fontSize: '15px' }} />
                        </div>
                        <div className="flex flex-col leading-none gap-0.5">
                          <span className="text-slate-800 text-base font-bold tracking-tight">MProof</span>
                          <span className="text-[9px] font-semibold tracking-[0.1em] text-slate-400 uppercase">Document Analysis</span>
                        </div>
                      </Link>
                      <div className="flex items-center gap-2 sm:gap-3">
                        <div className="hidden sm:block">
                          <LLMStatus />
                        </div>
                        <Navigation />
                        <UserMenu />
                      </div>
                    </div>
                  </div>
                </header>
                <main className="max-w-[1600px] mx-auto px-3 sm:px-4 lg:px-8 py-4 sm:py-8">
                  {children}
                </main>
              </div>
            </ModelProvider>
          </AuthProvider>
        </QueryProvider>
      </body>
    </html>
  );
}
