import React, { useState, useEffect } from 'react';
import { Stethoscope, AlertTriangle, Sliders, ShieldCheck, Heart, FlaskConical, Play, ShieldAlert } from 'lucide-react';
import { PatientCase, PatientVitals, InferenceResult } from '../types';
import { fetchSamplePatients, runPatientInference } from '../api';

export const PatientInference: React.FC = () => {
  const [patients, setPatients] = useState<PatientCase[]>([]);
  const [selectedPatientId, setSelectedPatientId] = useState<string>('');
  const [vitals, setVitals] = useState<PatientVitals>({
    age: 74,
    gender: 0,
    heart_rate_mean: 118,
    sysbp_mean: 88,
    diasbp_mean: 54,
    meanbp_mean: 65,
    resprate_mean: 29,
    tempc_mean: 38.6,
    spo2_mean: 89,
    glucose_mean: 210,
    sodium_mean: 145,
    potassium_mean: 5.2,
    creatinine_mean: 2.8,
    bun_mean: 42,
    wbc_mean: 19.4,
    hgb_mean: 9.1,
    icu_los_hours: 72,
  });

  const [inferenceResult, setInferenceResult] = useState<InferenceResult | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchSamplePatients()
      .then((data) => {
        setPatients(data);
        if (data.length > 0) {
          setSelectedPatientId(data[0].id);
          setVitals(data[0].vitals);
          handleRunInference(data[0].vitals);
        }
      })
      .catch((err) => console.error(err));
  }, []);

  const handlePatientSelect = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const pId = e.target.value;
    setSelectedPatientId(pId);
    const p = patients.find((item) => item.id === pId);
    if (p) {
      setVitals(p.vitals);
      handleRunInference(p.vitals);
    }
  };

  const handleVitalChange = (key: keyof PatientVitals, val: number) => {
    const updated = { ...vitals, [key]: val };
    setVitals(updated);
  };

  const handleRunInference = async (currentVitals: PatientVitals = vitals) => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await runPatientInference(currentVitals);
      setInferenceResult(result);
    } catch (err: any) {
      setError(err.message || 'Failed to analyze patient risk profile.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6 animate-fade-in">
      
      {/* Clinical Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between p-8 rounded-2xl border border-[#2C2C2E] bg-[#161618] shadow-lg gap-6">
        <div className="flex items-center gap-3">
          <Stethoscope className="h-6 w-6 text-[#F2F2F2]" />
          <h2 className="font-serif-editorial text-2xl lg:text-3xl font-bold tracking-tight text-[#F2F2F2]">
            ICU Patient Risk Triage & Deterioration Assessment
          </h2>
        </div>

        {/* Load Preset Patient */}
        <div className="flex items-center gap-4">
          <label className="text-xs font-mono-tech uppercase text-[#8E8E93] tracking-wider whitespace-nowrap">Patient Profile:</label>
          <select
            value={selectedPatientId}
            onChange={handlePatientSelect}
            className="rounded-lg border border-[#3C3C3E] bg-[#222225] text-sm font-semibold text-[#F2F2F2] px-4 py-2.5 focus:outline-none focus:border-[#6C6C70] shadow-sm"
          >
            {patients.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name} ({p.id}) — Age {Number(p.age).toFixed(0)}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Grid: Patient Vital Sign Controls (5 cols) vs Assessment Report (7 cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">

        {/* Left Column: Clinical Parameters Input (5 cols) */}
        <div className="lg:col-span-5 rounded-2xl border border-[#2C2C2E] bg-[#161618] p-8 space-y-6 shadow-md">
          <div className="flex items-center justify-between border-b border-[#262628] pb-4">
            <div className="flex items-center gap-2.5">
              <Sliders className="h-5 w-5 text-[#B3B3B3]" />
              <h3 className="text-base font-bold text-[#F2F2F2]">Clinical Vitals & Laboratory Values</h3>
            </div>
            <button
              onClick={() => handleRunInference()}
              disabled={isLoading}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-[#F2F2F2] text-[#161618] text-xs sm:text-sm font-bold hover:bg-[#D9D9D9] transition-colors disabled:opacity-50 shadow-md"
            >
              <Play className="h-4 w-4 fill-[#161618]" />
              <span>{isLoading ? 'Analyzing...' : 'Assess Patient Risk'}</span>
            </button>
          </div>

          <div className="space-y-5 max-h-[720px] overflow-y-auto pr-3">
            
            {/* Vital Signs Section */}
            <div className="space-y-4">
              <p className="text-xs font-mono-tech text-[#8E8E93] uppercase tracking-wider flex items-center gap-2">
                <Heart className="h-4 w-4 text-[#B3B3B3]" />
                <span>Cardiovascular & Respiratory Vitals</span>
              </p>
              
              {/* Age */}
              <div>
                <div className="flex justify-between text-xs sm:text-sm mb-1.5 font-medium">
                  <span className="text-[#CCCCCC]">Age</span>
                  <span className="font-mono-tech text-[#F2F2F2] font-semibold">{Number(vitals.age).toFixed(0)} years</span>
                </div>
                <input
                  type="range" min="18" max="95" value={vitals.age}
                  onChange={(e) => handleVitalChange('age', parseFloat(e.target.value))}
                  className="w-full accent-[#F2F2F2] bg-[#2C2C2E] h-2 rounded-lg cursor-pointer"
                />
              </div>

              {/* Heart Rate */}
              <div>
                <div className="flex justify-between text-xs sm:text-sm mb-1.5 font-medium">
                  <span className="text-[#CCCCCC]">Heart Rate (Normal: 60–100 bpm)</span>
                  <span className="font-mono-tech text-[#F2F2F2] font-semibold">{Number(vitals.heart_rate_mean).toFixed(1)} bpm</span>
                </div>
                <input
                  type="range" min="40" max="170" value={vitals.heart_rate_mean}
                  onChange={(e) => handleVitalChange('heart_rate_mean', parseFloat(e.target.value))}
                  className="w-full accent-[#F2F2F2] bg-[#2C2C2E] h-2 rounded-lg cursor-pointer"
                />
              </div>

              {/* Systolic BP */}
              <div>
                <div className="flex justify-between text-xs sm:text-sm mb-1.5 font-medium">
                  <span className="text-[#CCCCCC]">Systolic Blood Pressure (Normal: 90–120 mmHg)</span>
                  <span className="font-mono-tech text-[#F2F2F2] font-semibold">{Number(vitals.sysbp_mean).toFixed(1)} mmHg</span>
                </div>
                <input
                  type="range" min="60" max="210" value={vitals.sysbp_mean}
                  onChange={(e) => handleVitalChange('sysbp_mean', parseFloat(e.target.value))}
                  className="w-full accent-[#F2F2F2] bg-[#2C2C2E] h-2 rounded-lg cursor-pointer"
                />
              </div>

              {/* Respiratory Rate */}
              <div>
                <div className="flex justify-between text-xs sm:text-sm mb-1.5 font-medium">
                  <span className="text-[#CCCCCC]">Respiratory Rate (Normal: 12–20 /min)</span>
                  <span className="font-mono-tech text-[#F2F2F2] font-semibold">{Number(vitals.resprate_mean).toFixed(1)} /min</span>
                </div>
                <input
                  type="range" min="8" max="45" value={vitals.resprate_mean}
                  onChange={(e) => handleVitalChange('resprate_mean', parseFloat(e.target.value))}
                  className="w-full accent-[#F2F2F2] bg-[#2C2C2E] h-2 rounded-lg cursor-pointer"
                />
              </div>

              {/* Temperature */}
              <div>
                <div className="flex justify-between text-xs sm:text-sm mb-1.5 font-medium">
                  <span className="text-[#CCCCCC]">Body Temperature (Normal: 36.5–37.5 °C)</span>
                  <span className="font-mono-tech text-[#F2F2F2] font-semibold">{Number(vitals.tempc_mean).toFixed(1)} °C</span>
                </div>
                <input
                  type="range" min="34" max="41" step="0.1" value={vitals.tempc_mean}
                  onChange={(e) => handleVitalChange('tempc_mean', parseFloat(e.target.value))}
                  className="w-full accent-[#F2F2F2] bg-[#2C2C2E] h-2 rounded-lg cursor-pointer"
                />
              </div>

              {/* SpO2 */}
              <div>
                <div className="flex justify-between text-xs sm:text-sm mb-1.5 font-medium">
                  <span className="text-[#CCCCCC]">Oxygen Saturation SpO2 (Normal: 95–100%)</span>
                  <span className="font-mono-tech text-[#F2F2F2] font-semibold">{Number(vitals.spo2_mean).toFixed(1)} %</span>
                </div>
                <input
                  type="range" min="70" max="100" value={vitals.spo2_mean}
                  onChange={(e) => handleVitalChange('spo2_mean', parseFloat(e.target.value))}
                  className="w-full accent-[#F2F2F2] bg-[#2C2C2E] h-2 rounded-lg cursor-pointer"
                />
              </div>

            </div>

            <hr className="border-[#262628]" />

            {/* Laboratory Measurements */}
            <div className="space-y-4">
              <p className="text-xs font-mono-tech text-[#8E8E93] uppercase tracking-wider flex items-center gap-2">
                <FlaskConical className="h-4 w-4 text-[#B3B3B3]" />
                <span>Laboratory Diagnostic Measurements</span>
              </p>
              
              {/* Glucose */}
              <div>
                <div className="flex justify-between text-xs sm:text-sm mb-1.5 font-medium">
                  <span className="text-[#CCCCCC]">Blood Glucose (Normal: 70–140 mg/dL)</span>
                  <span className="font-mono-tech text-[#F2F2F2] font-semibold">{Number(vitals.glucose_mean).toFixed(1)} mg/dL</span>
                </div>
                <input
                  type="range" min="60" max="350" value={vitals.glucose_mean}
                  onChange={(e) => handleVitalChange('glucose_mean', parseFloat(e.target.value))}
                  className="w-full accent-[#F2F2F2] bg-[#2C2C2E] h-2 rounded-lg cursor-pointer"
                />
              </div>

              {/* WBC */}
              <div>
                <div className="flex justify-between text-xs sm:text-sm mb-1.5 font-medium">
                  <span className="text-[#CCCCCC]">White Blood Count WBC (Normal: 4.5–11.0 k/uL)</span>
                  <span className="font-mono-tech text-[#F2F2F2] font-semibold">{Number(vitals.wbc_mean).toFixed(1)} k/uL</span>
                </div>
                <input
                  type="range" min="2" max="40" step="0.5" value={vitals.wbc_mean}
                  onChange={(e) => handleVitalChange('wbc_mean', parseFloat(e.target.value))}
                  className="w-full accent-[#F2F2F2] bg-[#2C2C2E] h-2 rounded-lg cursor-pointer"
                />
              </div>

              {/* Creatinine */}
              <div>
                <div className="flex justify-between text-xs sm:text-sm mb-1.5 font-medium">
                  <span className="text-[#CCCCCC]">Serum Creatinine (Normal: 0.6–1.2 mg/dL)</span>
                  <span className="font-mono-tech text-[#F2F2F2] font-semibold">{Number(vitals.creatinine_mean).toFixed(1)} mg/dL</span>
                </div>
                <input
                  type="range" min="0.4" max="8.0" step="0.1" value={vitals.creatinine_mean}
                  onChange={(e) => handleVitalChange('creatinine_mean', parseFloat(e.target.value))}
                  className="w-full accent-[#F2F2F2] bg-[#2C2C2E] h-2 rounded-lg cursor-pointer"
                />
              </div>

              {/* Potassium */}
              <div>
                <div className="flex justify-between text-xs sm:text-sm mb-1.5 font-medium">
                  <span className="text-[#CCCCCC]">Potassium (Normal: 3.5–5.0 mEq/L)</span>
                  <span className="font-mono-tech text-[#F2F2F2] font-semibold">{Number(vitals.potassium_mean).toFixed(1)} mEq/L</span>
                </div>
                <input
                  type="range" min="2.5" max="7.5" step="0.1" value={vitals.potassium_mean}
                  onChange={(e) => handleVitalChange('potassium_mean', parseFloat(e.target.value))}
                  className="w-full accent-[#F2F2F2] bg-[#2C2C2E] h-2 rounded-lg cursor-pointer"
                />
              </div>

            </div>

          </div>
        </div>

        {/* Right Column: Physician Assessment Output Report (7 cols) */}
        <div className="lg:col-span-7 space-y-6">

          {error && (
            <div className="p-4 rounded-lg border border-rose-900/50 bg-rose-950/30 text-rose-200 text-sm flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-rose-400 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {inferenceResult ? (
            <div className="space-y-6 animate-fade-in-up">
              
              {/* Triage Summary Box */}
              <div className="rounded-xl border border-[#3C3C3E] bg-[#1C1C1E] p-6 space-y-4 shadow-lg">
                <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#2C2C2E] pb-4">
                  <div>
                    <span className="text-xs font-mono-tech uppercase text-[#8E8E93] tracking-wider">
                      Clinical Risk Assessment Report
                    </span>
                    <div className="flex items-center gap-3 mt-1">
                      <h3 className="font-serif-editorial text-3xl font-bold text-[#F2F2F2]">
                        {inferenceResult.mortality_risk_percentage}%
                      </h3>
                      <span className="text-sm font-medium text-[#B3B3B3]">ICU Mortality Risk Index</span>
                    </div>
                  </div>

                  {/* Clinical Triage Status Badge */}
                  <div className={`px-4 py-1.5 rounded-full text-xs font-semibold font-mono-tech border ${
                    inferenceResult.severity_code === 'RED'
                      ? 'bg-rose-950/60 text-rose-300 border-rose-800'
                      : inferenceResult.severity_code === 'ORANGE'
                      ? 'bg-amber-950/60 text-amber-300 border-amber-800'
                      : inferenceResult.severity_code === 'YELLOW'
                      ? 'bg-yellow-950/60 text-yellow-300 border-yellow-800'
                      : 'bg-emerald-950/60 text-emerald-300 border-emerald-800'
                  }`}>
                    {inferenceResult.severity}
                  </div>
                </div>

                {/* Risk Meter Bar */}
                <div>
                  <div className="flex justify-between text-xs text-[#8E8E93] mb-1.5 font-mono-tech">
                    <span>LOW DETERIORATION RISK</span>
                    <span>HIGH DETERIORATION RISK</span>
                  </div>
                  <div className="w-full h-3 rounded-full bg-[#262629] overflow-hidden border border-[#3A3A3D]">
                    <div
                      className={`h-full transition-all duration-700 ${
                        inferenceResult.mortality_risk_percentage > 50
                          ? 'bg-[#F2F2F2]'
                          : inferenceResult.mortality_risk_percentage > 25
                          ? 'bg-[#CCCCCC]'
                          : 'bg-[#888888]'
                      }`}
                      style={{ width: `${Math.max(5, inferenceResult.mortality_risk_percentage)}%` }}
                    />
                  </div>
                </div>

                {/* NEWS2 & Physician Recommendation */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2">
                  <div className="p-4 rounded-lg border border-[#2C2C2E] bg-[#161618]">
                    <div className="text-xs text-[#8E8E93] font-mono-tech uppercase">NEWS2 Clinical Score</div>
                    <div className="text-xl font-bold text-[#F2F2F2] mt-1">
                      Score: {inferenceResult.news_score} / 20
                    </div>
                    <div className="text-xs text-[#B3B3B3] mt-1">{inferenceResult.news_risk_level}</div>
                  </div>

                  <div className="p-4 rounded-lg border border-[#2C2C2E] bg-[#161618]">
                    <div className="text-xs text-[#8E8E93] font-mono-tech uppercase">Physician Protocol</div>
                    <div className="text-xs text-[#E6E2D8] mt-1 leading-relaxed">
                      {inferenceResult.recommendation}
                    </div>
                  </div>
                </div>

              </div>

              {/* Primary Clinical Risk Factor Drivers */}
              <div className="rounded-xl border border-[#2C2C2E] bg-[#161618] p-6 space-y-4">
                <h4 className="text-sm font-semibold text-[#F2F2F2] flex items-center gap-2">
                  <ShieldAlert className="h-4 w-4 text-[#B3B3B3]" />
                  <span>Key Physiological Risk Factor Deviations</span>
                </h4>

                <div className="space-y-3">
                  {inferenceResult.top_risk_factors.map((item, idx) => (
                    <div key={idx} className="p-3 rounded-lg border border-[#262628] bg-[#1A1A1D] flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                      <div>
                        <div className="text-xs font-semibold text-[#F2F2F2] flex items-center gap-2">
                          <span>{item.feature}</span>
                          <span className={`text-[10px] font-mono-tech px-2 py-0.5 rounded border ${
                            item.status === 'Elevated'
                              ? 'border-amber-800 text-amber-300 bg-amber-950/40'
                              : item.status === 'Depressed'
                              ? 'border-sky-800 text-sky-300 bg-sky-950/40'
                              : 'border-[#3C3C3E] text-[#B3B3B3] bg-[#242427]'
                          }`}>
                            {item.status}
                          </span>
                        </div>
                        <div className="text-xs text-[#8E8E93] mt-0.5">
                          Patient Value: <span className="font-mono-tech text-[#CCCCCC]">{item.patient_value}</span> • Normal Reference Baseline: <span className="font-mono-tech text-[#888888]">{item.baseline_mean}</span>
                        </div>
                      </div>

                      {/* Impact Bar */}
                      <div className="w-full sm:w-32">
                        <div className="flex justify-between text-[10px] font-mono-tech text-[#8E8E93] mb-1">
                          <span>Deviation</span>
                          <span>{(item.risk_impact * 100).toFixed(0)}%</span>
                        </div>
                        <div className="w-full h-1.5 rounded-full bg-[#2C2C2E] overflow-hidden">
                          <div
                            className="h-full bg-[#F2F2F2]"
                            style={{ width: `${Math.max(10, item.risk_impact * 100)}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Data Compliance & Security Verification Note */}
              <div className="rounded-xl border border-[#2C2C2E] bg-[#1A1A1D] p-4 text-xs font-mono-tech text-[#8E8E93] flex items-center gap-3">
                <ShieldCheck className="h-5 w-5 text-[#CCCCCC] shrink-0" />
                <div>
                  <span className="text-[#F2F2F2] font-semibold">HIPAA Data Security Compliant: </span>
                  Patient vital sign computations are performed locally using encrypted edge network protocol. No raw clinical records leave the hospital premises.
                </div>
              </div>

            </div>
          ) : (
            <div className="rounded-xl border border-dashed border-[#3A3A3C] p-12 text-center text-[#8E8E93]">
              <Stethoscope className="h-8 w-8 mx-auto text-[#555558]" />
              <p className="mt-3 text-sm">Select a patient profile or adjust vital signs to run clinical risk assessment.</p>
            </div>
          )}

        </div>

      </div>

    </div>
  );
};
