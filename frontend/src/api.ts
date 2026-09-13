import { SystemStatus, HospitalNode, PatientCase, PatientVitals, InferenceResult, TrainingRoundMetric } from './types';

const API_BASE = 'http://127.0.0.1:8000/api';

export async function fetchSystemStatus(): Promise<SystemStatus> {
  const res = await fetch(`${API_BASE}/status`);
  if (!res.ok) throw new Error('Failed to fetch system status');
  return res.json();
}

export async function fetchHospitalNodes(): Promise<HospitalNode[]> {
  const res = await fetch(`${API_BASE}/hospitals`);
  if (!res.ok) throw new Error('Failed to fetch hospital nodes');
  return res.json();
}

export async function fetchSamplePatients(): Promise<PatientCase[]> {
  const res = await fetch(`${API_BASE}/patients`);
  if (!res.ok) throw new Error('Failed to fetch sample patients');
  return res.json();
}

export async function fetchTrainingMetrics(): Promise<TrainingRoundMetric[]> {
  const res = await fetch(`${API_BASE}/metrics`);
  if (!res.ok) throw new Error('Failed to fetch training metrics');
  const data = await res.json();
  if (!Array.isArray(data)) return [];
  return data.map((item: any) => ({
    round: item.round ?? 0,
    accuracy: item.accuracy ?? item.global?.accuracy ?? 0,
    loss: item.loss ?? item.global?.loss ?? 0,
    pam_score: item.pam_score ?? item.pam ?? 0,
    eps_mean: item.eps_mean ?? item.privacy?.mean ?? 0,
    nan_clients: item.nan_clients ?? 0,
    time_sec: item.time_sec ?? 0,
  }));
}

export async function runPatientInference(vitals: PatientVitals): Promise<InferenceResult> {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(vitals),
  });
  if (!res.ok) throw new Error('Failed to run AI model inference');
  return res.json();
}

export async function triggerTraining(rounds: number = 5): Promise<{ message: string }> {
  const res = await fetch(`${API_BASE}/trigger-training?rounds=${rounds}`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error('Failed to trigger training');
  return res.json();
}
