import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { PatientInference } from './components/PatientInference';
import { HospitalNetwork } from './components/HospitalNetwork';
import { TrainingAnalytics } from './components/TrainingAnalytics';
import { QuantumMPSDetails } from './components/QuantumMPSDetails';
import { SystemStatus } from './types';
import { fetchSystemStatus, triggerTraining } from './api';

export function App() {
  const [activeTab, setActiveTab] = useState<string>('inference');
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [isTriggering, setIsTriggering] = useState<boolean>(false);
  const [triggerMsg, setTriggerMsg] = useState<string | null>(null);

  const loadStatus = () => {
    fetchSystemStatus()
      .then((data) => setStatus(data))
      .catch((err) => console.error('Failed to load status:', err));
  };

  useEffect(() => {
    loadStatus();
    const interval = setInterval(loadStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleTriggerTraining = async () => {
    setIsTriggering(true);
    setTriggerMsg(null);
    try {
      const res = await triggerTraining(5);
      setTriggerMsg(res.message);
      setTimeout(() => setTriggerMsg(null), 5000);
    } catch (err: any) {
      setTriggerMsg(err.message || 'Failed to trigger training');
    } finally {
      setIsTriggering(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#141416] text-[#F2F2F2] flex flex-col">
      
      {/* Header */}
      <Header
        status={status}
        onTriggerTraining={handleTriggerTraining}
        isTriggering={isTriggering}
        activeTab={activeTab}
        setActiveTab={setActiveTab}
      />

      {/* Trigger Notification Toast */}
      {triggerMsg && (
        <div className="max-w-[1920px] mx-auto w-full px-6 lg:px-12 mt-4">
          <div className="p-4 rounded-xl border border-emerald-900/60 bg-emerald-950/40 text-emerald-300 text-sm font-mono-tech flex items-center justify-between shadow-md">
            <span>[NETWORK SYNC] {triggerMsg}</span>
            <button onClick={() => setTriggerMsg(null)} className="text-emerald-400 hover:text-white text-base">✕</button>
          </div>
        </div>
      )}

      {/* Main Container */}
      <main className="flex-1 max-w-[1920px] w-full mx-auto px-6 lg:px-12 py-8">
        {activeTab === 'inference' && <PatientInference />}
        {activeTab === 'hospitals' && <HospitalNetwork />}
        {activeTab === 'analytics' && <TrainingAnalytics />}
        {activeTab === 'mps' && <QuantumMPSDetails status={status} />}
      </main>

      {/* Footer */}
      <footer className="border-t border-[#222225] py-6 text-center text-xs text-[#7C7C80] font-mono-tech">
        <p>QI-CAF-EI • Hospital Edge Intelligence Portal • 20 Participating Medical Centers</p>
      </footer>

    </div>
  );
}

export default App;
