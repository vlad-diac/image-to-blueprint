import { ConfigService } from '@nestjs/config';
import type { RunpodHandlerInput, RunpodStatusResponse, RunpodSubmitResponse } from './runpod.types';
export declare class RunpodService {
    private readonly config;
    private readonly baseUrl;
    private readonly apiKey;
    constructor(config: ConfigService);
    private headers;
    submit(input: RunpodHandlerInput): Promise<RunpodSubmitResponse>;
    status(jobId: string): Promise<RunpodStatusResponse>;
    cancel(jobId: string): Promise<RunpodStatusResponse>;
}
