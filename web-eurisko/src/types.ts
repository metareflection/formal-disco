export interface Overview {
  total_tasks: number;
  total_objects: number;
  total_concepts: number;
  total_heuristics: number;
  proved_count: number;
  conjecture_count: number;
  definition_count: number;
  task_status: Record<string, Record<string, number>>;
  task_outcomes: Record<string, Record<string, number>>;
}

export interface Concept {
  path: string;
  name: string;
  kind: string;
  domain: string;
  description: string;
  lean_statement: string;
  lean_proof: string | null;
  lean_imports: string[];
  tags: string[];
  related_concepts: string[];
  origin_heuristic: string | null;
  proof_strategy: string | null;
  proof_attempts: number;
  interestingness: number;
  parents: string[];
}

export interface Heuristic {
  path: string;
  name: string;
  heuristic_kind: string;
  eurisclo_origin: string | null;
  input_concept_kinds: string[];
  attempts: number;
  successes: number;
  interestingness: number;
  born_from_reflection: boolean;
  concepts_created: number;
  theorems_proved: number;
  template: string;
}

export interface Theorem {
  name: string;
  description: string;
  lean_statement: string;
  lean_proof: string;
  proof_strategy: string;
  origin_heuristic: string | null;
  interestingness: number;
}

export interface GraphNode {
  id: string;
  name: string;
  kind: string;
  origin_heuristic: string | null;
  interestingness: number;
  proved: boolean;
}

export interface GraphEdge {
  source: string;
  target: string;
  type: string;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface TraceEvent {
  path: string;
  kind: string;
  worker: string;
  tick: number;
  seq: number;
  timestamp: string;
  payload: Record<string, unknown>;
}

export interface TraceSummary {
  total: number;
  by_kind: Record<string, number>;
  by_worker: Record<string, number>;
  by_kind_worker: Record<string, number>;
}
