export interface SystemStatus {
  status: string;
  framework: string;
  active_model_path: string;
  total_hospital_nodes: number;
  active_nodes_per_round: number;
  dataset: string;
  model_architecture: string;
  mps_compression_ratio: number;
  parameter_reduction_pct: number;
  total_mps_params: number;
  total_dense_params: number;
  latest_accuracy: number;
  latest_pam_score: number;
  latest_privacy_eps: number;
  privacy_budget_max: number;
}

export interface HospitalNode {
  id: number;
  name: string;
  location: string;
  capacity_beds: number;
  sample_size: number;
  privacy_budget_eps: number;
  privacy_limit: number;
  privacy_pct_used: number;
  local_accuracy: number;
  status: string;
  mps_compressed_size_kb: number;
  dense_size_kb: number;
}

export interface PatientVitals {
  age: number;
  gender: number;
  heart_rate_mean: number;
  sysbp_mean: number;
  diasbp_mean: number;
  meanbp_mean: number;
  resprate_mean: number;
  tempc_mean: number;
  spo2_mean: number;
  glucose_mean: number;
  sodium_mean: number;
  potassium_mean: number;
  creatinine_mean: number;
  bun_mean: number;
  wbc_mean: number;
  hgb_mean: number;
  icu_los_hours: number;
}

export interface PatientCase {
  id: string;
  name: string;
  age: number;
  gender: string;
  condition: string;
  vitals: PatientVitals;
}

export interface FeatureImpact {
  feature: string;
  key: string;
  patient_value: number;
  baseline_mean: number;
  risk_impact: number;
  status: string;
}

export interface InferenceResult {
  mortality_probability: number;
  mortality_risk_percentage: number;
  severity: string;
  severity_code: string;
  news_score: number;
  news_risk_level: string;
  recommendation: string;
  top_risk_factors: FeatureImpact[];
  all_feature_analysis: FeatureImpact[];
  model_provenance: {
    checkpoint_used: string;
    architecture: string;
    mps_compression: string;
    dp_privacy_eps: number;
    homomorphic_encrypted: boolean;
  };
}

export interface TrainingRoundMetric {
  round: number;
  accuracy: number;
  loss: number;
  pam_score: number;
  eps_mean: number;
  nan_clients: number;
  time_sec: number;
}
