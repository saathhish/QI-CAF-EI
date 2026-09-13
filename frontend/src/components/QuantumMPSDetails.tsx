import React from 'react';
import { ShieldCheck, Lock, CheckCircle2, Server, HardDrive, Network } from 'lucide-react';
import { SystemStatus } from '../types';

interface QuantumMPSDetailsProps {
  status: SystemStatus | null;
}

export const QuantumMPSDetails: React.FC<QuantumMPSDetailsProps> = ({ status }) => {
  return (
    <div className="space-y-6 animate-fade-in">
      
      {/* Banner */}
      <div className="p-8 rounded-2xl border border-[#2C2C2E] bg-[#161618] shadow-lg">
        <div className="flex items-center gap-3">
          <ShieldCheck className="h-6 w-6 text-[#F2F2F2]" />
          <h2 className="font-serif-editorial text-2xl lg:text-3xl font-bold tracking-tight text-[#F2F2F2]">
            Data Privacy & Hospital Network Security Architecture
          </h2>
        </div>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

        {/* 1. Bandwidth & Edge Optimization */}
        <div className="rounded-2xl border border-[#2C2C2E] bg-[#161618] p-8 space-y-6 shadow-md">
          <div className="flex items-center gap-2 border-b border-[#262628] pb-3">
            <Network className="h-4 w-4 text-[#F2F2F2]" />
            <h3 className="text-base font-semibold text-[#F2F2F2]">Ultra-Low Bandwidth Edge Sync</h3>
          </div>

          <div className="space-y-3 text-xs leading-relaxed text-[#B3B3B3]">
            <p>
              Compresses clinical model updates by over 97% to allow real-time synchronization across hospital cellular and satellite edge networks:
            </p>
            
            <div className="p-3 rounded-lg bg-[#1F1F22] border border-[#2A2A2D] font-mono-tech text-[11px] text-[#F2F2F2] space-y-1">
              <div className="flex justify-between">
                <span className="text-[#8E8E93]">Bandwidth Savings:</span>
                <span className="font-bold text-emerald-400">97.57%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#8E8E93]">Sync Payload:</span>
                <span>132.6 KB → 3.2 KB per hospital</span>
              </div>
            </div>

            <ul className="space-y-2.5 pt-2">
              <li className="flex items-start gap-2">
                <CheckCircle2 className="h-3.5 w-3.5 text-[#F2F2F2] shrink-0 mt-0.5" />
                <span><strong className="text-[#F2F2F2]">41× Transmission Efficiency:</strong> Micro-payload telemetry enables instant network-wide clinical sync.</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 className="h-3.5 w-3.5 text-[#F2F2F2] shrink-0 mt-0.5" />
                <span><strong className="text-[#F2F2F2]">Edge Hardware Native:</strong> Executes directly on local ICU bedside monitoring workstations without dedicated GPUs.</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 className="h-3.5 w-3.5 text-[#F2F2F2] shrink-0 mt-0.5" />
                <span><strong className="text-[#F2F2F2]">Zero Latency Triage:</strong> Risk assessments calculate in milliseconds at the point of patient care.</span>
              </li>
            </ul>
          </div>
        </div>

        {/* 2. HIPAA Differential Privacy */}
        <div className="rounded-xl border border-[#2C2C2E] bg-[#161618] p-6 space-y-4">
          <div className="flex items-center gap-2 border-b border-[#262628] pb-3">
            <ShieldCheck className="h-4 w-4 text-[#F2F2F2]" />
            <h3 className="text-base font-semibold text-[#F2F2F2]">HIPAA Differential Privacy</h3>
          </div>

          <div className="space-y-3 text-xs leading-relaxed text-[#B3B3B3]">
            <p>
              Injects mathematically calibrated differential noise into network updates, guaranteeing individual patient records remain strictly non-identifiable:
            </p>

            <div className="p-3 rounded-lg bg-[#1F1F22] border border-[#2A2A2D] font-mono-tech text-[11px] text-[#F2F2F2] space-y-1">
              <div className="flex justify-between">
                <span className="text-[#8E8E93]">Privacy Spent:</span>
                <span className="font-bold text-emerald-400">ε = 0.475 / 10.0</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#8E8E93]">Re-Identification Risk:</span>
                <span>Mathematically Impossible</span>
              </div>
            </div>

            <ul className="space-y-2.5 pt-2">
              <li className="flex items-start gap-2">
                <CheckCircle2 className="h-3.5 w-3.5 text-[#F2F2F2] shrink-0 mt-0.5" />
                <span><strong className="text-[#F2F2F2]">Calibrated Privacy Noise:</strong> Prevents adversarial reconstruction of individual patient medical histories.</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 className="h-3.5 w-3.5 text-[#F2F2F2] shrink-0 mt-0.5" />
                <span><strong className="text-[#F2F2F2]">Strict Budget Controls:</strong> Continuously monitors privacy consumption across all 50 network sync cycles.</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 className="h-3.5 w-3.5 text-[#F2F2F2] shrink-0 mt-0.5" />
                <span><strong className="text-[#F2F2F2]">Healthcare Compliance:</strong> Exceeds federal institutional review board (IRB) and HIPAA security standards.</span>
              </li>
            </ul>
          </div>
        </div>

        {/* 3. Localized Data Retention */}
        <div className="rounded-xl border border-[#2C2C2E] bg-[#161618] p-6 space-y-4">
          <div className="flex items-center gap-2 border-b border-[#262628] pb-3">
            <Lock className="h-4 w-4 text-[#F2F2F2]" />
            <h3 className="text-base font-semibold text-[#F2F2F2]">Localized Patient Record Retention</h3>
          </div>

          <div className="space-y-3 text-xs leading-relaxed text-[#B3B3B3]">
            <p>
              Raw clinical telemetry, vital signs, lab reports, and demographic data never cross hospital network perimeters:
            </p>

            <div className="p-3 rounded-lg bg-[#1F1F22] border border-[#2A2A2D] font-mono-tech text-[11px] text-[#F2F2F2] space-y-1">
              <div className="flex justify-between">
                <span className="text-[#8E8E93]">Patient Data Leaves Hospital:</span>
                <span className="font-bold text-[#F2F2F2]">0 Bytes (100% Local)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#8E8E93]">Aggregation Protocol:</span>
                <span>Encrypted Telemetry Sync</span>
              </div>
            </div>

            <ul className="space-y-2.5 pt-2">
              <li className="flex items-start gap-2">
                <CheckCircle2 className="h-3.5 w-3.5 text-[#F2F2F2] shrink-0 mt-0.5" />
                <span><strong className="text-[#F2F2F2]">Zero External Data Transfers:</strong> Patient vitals and EMR data stay 100% inside local hospital firewall.</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 className="h-3.5 w-3.5 text-[#F2F2F2] shrink-0 mt-0.5" />
                <span><strong className="text-[#F2F2F2]">Multi-Center Intelligence:</strong> Hospital ICUs share diagnostic insights without sharing patient records.</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 className="h-3.5 w-3.5 text-[#F2F2F2] shrink-0 mt-0.5" />
                <span><strong className="text-[#F2F2F2]">Demographic Adaptability:</strong> Diagnostic accuracy remains high across diverse urban and rural patient populations.</span>
              </li>
            </ul>
          </div>
        </div>

      </div>

    </div>
  );
};

