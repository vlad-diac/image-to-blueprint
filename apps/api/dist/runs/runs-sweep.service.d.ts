import { RunsService } from './runs.service';
export declare class RunsSweepService {
    private readonly runs;
    constructor(runs: RunsService);
    sweep(): void;
}
