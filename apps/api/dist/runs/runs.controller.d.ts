import { StreamableFile } from '@nestjs/common';
import { RunsService } from './runs.service';
import { CreateRunMultipartDto } from './dto/create-run-multipart.dto';
export declare class RunsController {
    private readonly runs;
    constructor(runs: RunsService);
    list(limitRaw?: string): Promise<import("./runs.helpers").RunSerializable[]>;
    create(file: Express.Multer.File, dto: CreateRunMultipartDto): Promise<import("./runs.helpers").RunSerializable>;
    cancel(id: string): Promise<import("./runs.helpers").RunSerializable>;
    input(id: string): Promise<StreamableFile>;
    output(id: string): Promise<StreamableFile>;
    getOne(id: string): Promise<import("./runs.helpers").RunSerializable>;
}
