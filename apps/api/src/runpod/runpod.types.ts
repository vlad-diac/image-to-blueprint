export type RunpodJobStatus =
  | 'IN_QUEUE'
  | 'IN_PROGRESS'
  | 'COMPLETED'
  | 'FAILED'
  | 'CANCELLED'
  | 'TIMED_OUT';

export interface RunpodHandlerOutput {
  /** Legacy single output (worker v1). */
  image_b64?: string;
  job_dir?: string;
  width?: number;
  height?: number;
  num_images?: number;
  images?: Array<{
    index: number;
    seed?: number;
    image_b64?: string;
  }>;
}

export interface RunpodStatusResponse {
  id?: string;
  status: RunpodJobStatus;
  delayTime?: number;
  executionTime?: number;
  output?: RunpodHandlerOutput | string;
  error?: string;
}

export interface RunpodSubmitResponse {
  id: string;
  status?: RunpodJobStatus;
}

export interface RunpodHandlerInput {
  image_b64: string;
  positive_prompt: string;
  negative_prompt?: string;
  steps?: number;
  cfg?: number;
  seed?: number;
  /** 1–4 variants; distinct seed per image on the worker. */
  num_images?: number;
}
