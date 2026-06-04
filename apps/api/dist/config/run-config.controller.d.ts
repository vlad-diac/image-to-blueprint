import { UpdateRunConfigDto } from './dto/update-run-config.dto';
import { RunConfigService } from './run-config.service';
export declare class RunConfigController {
    private readonly runConfig;
    constructor(runConfig: RunConfigService);
    get(): Promise<ReturnType<RunConfigService['getOrCreate']>>;
    put(dto: UpdateRunConfigDto): Promise<ReturnType<RunConfigService['upsert']>>;
}
