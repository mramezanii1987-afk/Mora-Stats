export type VarType = "continuous" | "ordinal" | "nominal";
export type DesignType = "experimental" | "quasi_experimental" | "correlational";

export interface Variable {
  name: string;
  type: VarType;
  label: string | null;
  unit: string | null;
  lower_bound: number | null;
  upper_bound: number | null;
}

export interface VariableSummary {
  name: string;
  type: VarType;
  missing: number;
  distinct: number;
  mean?: number;
  sd?: number;
  min?: number;
  max?: number;
  sparkline?: string;
  top?: Record<string, number>;
}

export interface Dataset {
  source: string | null;
  n_rows: number;
  columns: string[];
  variables: Variable[];
  summary: VariableSummary[];
  preview: Record<string, unknown>[];
  design: DesignType;
  truncated: boolean;
}

export interface Suggestion {
  test: string;
  label: string;
  eligible: boolean;
  reason: string;
  recommended: boolean;
}

export interface Estimate {
  name: string;
  value: number;
  ci_low: number | null;
  ci_high: number | null;
  method: string;
  approximate: boolean;
}

export interface GroupDescriptives {
  label: string;
  n: number;
  mean: number | null;
  sd: number | null;
  median: number | null;
  iqr: number | null;
  missing: number;
}

export interface Threat {
  code: string;
  severity: "note" | "warn" | "serious";
  message: string;
  action: string;
  values: Record<string, unknown>;
}

export interface PowerSummary {
  alpha: number;
  target_power: number;
  mde: number | null;
  mde_label: string;
  mde_raw: number | null;
  observed_power: number | null;
  exaggeration: number | null;
  approximate: boolean;
  note: string;
}

export interface AnalysisResult {
  analysis: string;
  label: string;
  design: DesignType;
  statistic_name: string;
  statistic: number | null;
  df: number | null;
  df2: number | null;
  p: number | null;
  alpha: number;
  n: number;
  n_missing: number;
  effect: Estimate | null;
  raw_effect: Estimate | null;
  groups: GroupDescriptives[];
  variables: string[];
  power: PowerSummary;
  threats: Threat[];
  extra: Record<string, unknown>;
  inferential: boolean;
}

export interface TableColumn {
  key: string;
  label: string;
  align: "left" | "right";
}

export interface ResultTable {
  key: string;
  title: string;
  columns: TableColumn[];
  rows: Record<string, string | boolean>[];
  note: string;
  emphasis: string;
}

export interface Interpretation {
  result_rows: string[];
  tables: ResultTable[];
  apa: string;
  sentences: string[];
  threats: Threat[];
  multiplicity: string | null;
  provenance: Record<string, unknown>;
}

export interface ChartGroup {
  label: string;
  n: number;
  mean: number;
  ci_low: number;
  ci_high: number;
  median: number;
  q1: number;
  q3: number;
  whisker_low: number;
  whisker_high: number;
  values: number[];
  jitter: number[];
}

export interface BandPoint {
  x: number;
  y: number;
  low: number;
  high: number;
}

export interface MosaicCell {
  row: string;
  column: string;
  count: number;
  expected: number | null;
  residual: number;
  share: number;
}

export type ChartView = "intervals" | "distribution" | "points" | "fit" | "share" | "residual";

export interface ChartSpec {
  kind: "groups" | "scatter" | "mosaic" | "none";
  reason?: string;
  title?: string;
  unit?: string;
  x_label?: string;
  y_label?: string;
  views?: ChartView[];
  default_view?: ChartView;
  thinned?: boolean;
  groups?: ChartGroup[];
  points?: { x: number; y: number }[];
  fit?: { slope: number; intercept: number; x1: number; y1: number; x2: number; y2: number } | null;
  band?: BandPoint[];
  rows?: string[];
  columns?: string[];
  row_totals?: number[];
  cells?: MosaicCell[];
}

export interface LedgerState {
  project_count: number;
  session_count: number;
  family_wise_rate: number;
  holm: Record<string, number>;
  benjamini_hochberg: Record<string, number>;
}

export interface RunPayload {
  result: AnalysisResult;
  interpretation: Interpretation;
  chart: ChartSpec;
  ledger: LedgerState;
}

export interface EngineError {
  message: string;
  action: string;
  trace?: string;
}
