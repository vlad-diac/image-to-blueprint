import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import type {
  RunpodHandlerInput,
  RunpodStatusResponse,
  RunpodSubmitResponse,
} from './runpod.types';

@Injectable()
export class RunpodService {
  private readonly baseUrl: string;
  private readonly apiKey: string;

  constructor(private readonly config: ConfigService) {
    const id = this.config.getOrThrow<string>('RUNPOD_ENDPOINT_ID');
    this.apiKey = this.config.getOrThrow<string>('RUNPOD_API_KEY');
    this.baseUrl = `https://api.runpod.ai/v2/${id}`;
  }

  private headers(): Record<string, string> {
    return {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${this.apiKey}`,
    };
  }

  async submit(input: RunpodHandlerInput): Promise<RunpodSubmitResponse> {
    const res = await fetch(`${this.baseUrl}/run`, {
      method: 'POST',
      headers: this.headers(),
      body: JSON.stringify({ input }),
    });
    const text = await res.text();
    if (!res.ok) {
      throw new Error(`RunPod /run failed ${res.status}: ${text}`);
    }
    const json = JSON.parse(text) as Record<string, unknown>;
    const id = (json.id ?? json.jobId) as string | undefined;
    if (!id) {
      throw new Error(`RunPod /run unexpected body: ${text}`);
    }
    return { id, status: json.status as RunpodSubmitResponse['status'] };
  }

  async status(jobId: string): Promise<RunpodStatusResponse> {
    const res = await fetch(`${this.baseUrl}/status/${jobId}`, {
      method: 'GET',
      headers: { Authorization: `Bearer ${this.apiKey}` },
    });
    const text = await res.text();
    if (!res.ok) {
      throw new Error(`RunPod /status failed ${res.status}: ${text}`);
    }
    return JSON.parse(text) as RunpodStatusResponse;
  }

  async cancel(jobId: string): Promise<RunpodStatusResponse> {
    const res = await fetch(`${this.baseUrl}/cancel/${jobId}`, {
      method: 'POST',
      headers: this.headers(),
      body: JSON.stringify({}),
    });
    const text = await res.text();
    if (!res.ok) {
      throw new Error(`RunPod /cancel failed ${res.status}: ${text}`);
    }
    return JSON.parse(text) as RunpodStatusResponse;
  }
}
