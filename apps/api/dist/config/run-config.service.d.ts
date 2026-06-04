import { PrismaService } from '../prisma/prisma.service';
import type { UpdateRunConfigDto } from './dto/update-run-config.dto';
import { type RunConfigurationSerializable } from './run-config.helpers';
export declare class RunConfigService {
    private readonly prisma;
    constructor(prisma: PrismaService);
    getOrCreate(): Promise<RunConfigurationSerializable>;
    upsert(dto: UpdateRunConfigDto): Promise<RunConfigurationSerializable>;
}
