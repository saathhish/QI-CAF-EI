import React, { useState, useEffect } from 'react';
import { Building2, ShieldCheck, Database, Search, MapPin } from 'lucide-react';
import { HospitalNode } from '../types';
import { fetchHospitalNodes } from '../api';

export const HospitalNetwork: React.FC = () => {
  const [hospitals, setHospitals] = useState<HospitalNode[]>([]);
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(true);

  useEffect(() => {
    fetchHospitalNodes()
      .then((data) => setHospitals(data))
      .catch((err) => console.error(err))
      .finally(() => setIsLoading(false));
  }, []);

  const filteredHospitals = hospitals.filter((h) =>
    h.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    h.location.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="space-y-6 animate-fade-in">
      
      {/* Header Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between p-8 rounded-2xl border border-[#2C2C2E] bg-[#161618] shadow-lg gap-6">
        <div className="flex items-center gap-3">
          <Building2 className="h-6 w-6 text-[#F2F2F2]" />
          <h2 className="font-serif-editorial text-2xl lg:text-3xl font-bold tracking-tight text-[#F2F2F2]">
            20 Participating Medical Centers
          </h2>
        </div>

        {/* Search Filter Input */}
        <div className="relative w-full md:w-80">
          <Search className="absolute left-3.5 top-3 h-4 w-4 text-[#77777A]" />
          <input
            type="text"
            placeholder="Search medical center or state..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full rounded-lg border border-[#3C3C3E] bg-[#222225] text-sm text-[#F2F2F2] pl-10 pr-4 py-2.5 focus:outline-none focus:border-[#6C6C70] shadow-sm"
          />
        </div>
      </div>

      {/* Network Overview Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="p-6 rounded-2xl border border-[#2C2C2E] bg-[#161618] shadow-md">
          <div className="text-xs text-[#8E8E93] font-mono-tech uppercase tracking-wider">Connected Hospitals</div>
          <div className="text-3xl font-bold text-[#F2F2F2] mt-1 font-serif-editorial">20 Centers</div>
          <div className="text-xs text-[#B3B3B3] mt-1">16 Active Participants / Round</div>
        </div>

        <div className="p-6 rounded-2xl border border-[#2C2C2E] bg-[#161618] shadow-md">
          <div className="text-xs text-[#8E8E93] font-mono-tech uppercase tracking-wider">Diagnostic Accuracy</div>
          <div className="text-3xl font-bold text-[#F2F2F2] mt-1 font-serif-editorial">88.5%</div>
          <div className="text-xs text-[#B3B3B3] mt-1">Validated on Non-IID Patients</div>
        </div>

        <div className="p-6 rounded-2xl border border-[#2C2C2E] bg-[#161618] shadow-md">
          <div className="text-xs text-[#8E8E93] font-mono-tech uppercase tracking-wider">Privacy Compliance</div>
          <div className="text-3xl font-bold text-[#F2F2F2] mt-1 font-serif-editorial">95.2% Protected</div>
          <div className="text-xs text-[#B3B3B3] mt-1">Differential Privacy Guarantee</div>
        </div>

        <div className="p-6 rounded-2xl border border-[#2C2C2E] bg-[#161618] shadow-md">
          <div className="text-xs text-[#8E8E93] font-mono-tech uppercase tracking-wider">Data Confidentiality</div>
          <div className="text-3xl font-bold text-[#F2F2F2] mt-1 font-serif-editorial">0 Bytes Shared</div>
          <div className="text-xs text-[#B3B3B3] mt-1">No Patient Records Leave Hospital</div>
        </div>
      </div>

      {/* Grid of 20 Hospitals */}
      {isLoading ? (
        <div className="p-12 text-center text-[#8E8E93]">Loading medical center status...</div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredHospitals.map((node) => (
            <div
              key={node.id}
              className="group rounded-xl border border-[#2C2C2E] bg-[#161618] p-5 hover:border-[#4A4A4E] transition-all hover:shadow-lg space-y-4"
            >
              {/* Card Header */}
              <div className="flex items-start justify-between gap-2">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] font-mono-tech px-2 py-0.5 rounded border border-[#3C3C3E] bg-[#222225] text-[#CCCCCC]">
                      CENTER #{node.id < 10 ? `0${node.id}` : node.id}
                    </span>
                    <span className={`h-2 w-2 rounded-full ${node.status.includes('Active') ? 'bg-emerald-400' : 'bg-amber-400'}`} />
                  </div>
                  <h3 className="text-base font-semibold text-[#F2F2F2] mt-2 group-hover:text-white transition-colors">
                    {node.name}
                  </h3>
                  <div className="flex items-center gap-1 text-xs text-[#8E8E93] mt-0.5">
                    <MapPin className="h-3 w-3" />
                    <span>{node.location}</span>
                  </div>
                </div>

                <span className={`text-[10px] font-mono-tech px-2 py-1 rounded-md border ${
                  node.status.includes('Active')
                    ? 'border-emerald-900/60 bg-emerald-950/40 text-emerald-300'
                    : 'border-amber-900/60 bg-amber-950/40 text-amber-300'
                }`}>
                  {node.status.includes('Active') ? 'CONNECTED' : 'STANDBY'}
                </span>
              </div>

              <hr className="border-[#242426]" />

              {/* Center Metrics */}
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div>
                  <div className="text-[#8E8E93] font-mono-tech uppercase">PATIENT COHORT</div>
                  <div className="font-semibold text-[#F2F2F2] mt-0.5">{node.sample_size} ICU Cases</div>
                </div>
                <div>
                  <div className="text-[#8E8E93] font-mono-tech uppercase">ICU BEDS</div>
                  <div className="font-semibold text-[#F2F2F2] mt-0.5">{node.capacity_beds} Beds</div>
                </div>
                <div>
                  <div className="text-[#8E8E93] font-mono-tech uppercase">DIAGNOSTIC ACCURACY</div>
                  <div className="font-semibold text-[#F2F2F2] mt-0.5">{node.local_accuracy}%</div>
                </div>
                <div>
                  <div className="text-[#8E8E93] font-mono-tech uppercase">PRIVACY RATING</div>
                  <div className="font-semibold text-[#F2F2F2] mt-0.5">Grade A (Secure)</div>
                </div>
              </div>

              {/* Privacy Compliance Bar */}
              <div>
                <div className="flex justify-between text-[11px] font-mono-tech text-[#8E8E93] mb-1">
                  <span>HIPAA Privacy Protection (ε = {node.privacy_budget_eps})</span>
                  <span>{(100 - node.privacy_pct_used).toFixed(1)}% Protected</span>
                </div>
                <div className="w-full h-1.5 rounded-full bg-[#262629] overflow-hidden border border-[#3C3C3E]">
                  <div 
                    className="h-full bg-[#F2F2F2]" 
                    style={{ width: `${Math.min(100, Math.max(10, 100 - node.privacy_pct_used))}%` }}
                  />
                </div>
              </div>

            </div>
          ))}
        </div>
      )}

    </div>
  );
};
