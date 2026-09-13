import React from 'react';
import { HeartPulse, ShieldCheck, Building2, LineChart, RefreshCw, Lock, Stethoscope, Network } from 'lucide-react';
import { SystemStatus } from '../types';

interface HeaderProps {
  status: SystemStatus | null;
  onTriggerTraining: () => void;
  isTriggering: boolean;
  activeTab: string;
  setActiveTab: (tab: string) => void;
}

export const Header: React.FC<HeaderProps> = ({
  status,
  onTriggerTraining,
  isTriggering,
  activeTab,
  setActiveTab,
}) => {
  return (
    <header className="border-b border-[#2C2C2E] bg-[#161618]/95 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-[1920px] mx-auto px-6 lg:px-12 py-5">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-5">
          
          {/* Portal Brand Logo */}
          <div className="flex items-center gap-4">
            <div className="h-12 w-12 rounded-xl border border-[#3C3C3E] bg-[#1F1F22] flex items-center justify-center text-[#F2F2F2] shadow-md shrink-0">
              <HeartPulse className="h-6 w-6 text-[#F2F2F2]" />
            </div>
            <div>
              <h1 className="font-serif-editorial text-2xl lg:text-3xl font-bold tracking-tight text-[#F2F2F2]">
                Clinical Intelligence Portal
              </h1>
            </div>
          </div>

          {/* Clinical Badges & Sync Button */}
          <div className="flex flex-wrap items-center gap-4">

            {/* Medical Centers Badge */}
            <div className="hidden sm:flex items-center gap-2 px-4 py-2 rounded-lg border border-[#2C2C2E] bg-[#1A1A1D] text-xs sm:text-sm text-[#B3B3B3]">
              <Building2 className="h-4 w-4 text-[#E6E2D8]" />
              <span>20 Hospitals</span>
            </div>

            {/* HIPAA Compliance Badge */}
            <div className="hidden lg:flex items-center gap-2 px-4 py-2 rounded-lg border border-[#2C2C2E] bg-[#1A1A1D] text-xs sm:text-sm text-[#B3B3B3]">
              <ShieldCheck className="h-4 w-4 text-[#E6E2D8]" />
              <span>HIPAA Privacy Protected</span>
            </div>

            {/* Sync Clinical Model */}
            <button
              onClick={onTriggerTraining}
              disabled={isTriggering}
              className="group inline-flex items-center justify-center gap-2.5 rounded-lg border border-[#3A3A3C] bg-[#222225] hover:bg-[#2C2C30] px-4 py-2 text-xs sm:text-sm font-semibold text-[#F2F2F2] transition-colors disabled:opacity-50"
            >
              <RefreshCw className={`h-4 w-4 ${isTriggering ? 'animate-spin' : ''}`} />
              <span>{isTriggering ? 'Updating Model...' : 'Sync Network Model'}</span>
            </button>
          </div>

        </div>

        {/* Tab Navigation — Unique Relevant Icons per Tab */}
        <div className="flex items-center gap-2 mt-5 border-t border-[#222225] pt-4 overflow-x-auto">
          {[
            { id: 'inference', label: 'Clinical Triage & Risk Assessment', icon: Stethoscope },
            { id: 'hospitals', label: 'Participating Medical Centers (20)', icon: Network },
            { id: 'analytics', label: 'Diagnostic Accuracy & Trends', icon: LineChart },
            { id: 'mps', label: 'Data Privacy & Network Security', icon: Lock },
          ].map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`inline-flex items-center gap-2.5 px-5 py-2.5 rounded-lg text-xs sm:text-sm font-semibold transition-all whitespace-nowrap ${
                  isActive
                    ? 'bg-[#F2F2F2] text-[#161618] shadow-md font-bold'
                    : 'text-[#9E9E9E] hover:text-[#F2F2F2] hover:bg-[#222225]'
                }`}
              >
                <Icon className="h-4 w-4" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>

      </div>
    </header>
  );
};
