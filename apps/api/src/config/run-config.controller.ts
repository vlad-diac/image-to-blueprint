import { Body, Controller, Get, Put } from '@nestjs/common';
import { UpdateRunConfigDto } from './dto/update-run-config.dto';
import { RunConfigService } from './run-config.service';

@Controller('config')
export class RunConfigController {
  constructor(private readonly runConfig: RunConfigService) {}

  @Get()
  async get(): Promise<ReturnType<RunConfigService['getOrCreate']>> {
    return this.runConfig.getOrCreate();
  }

  @Put()
  async put(
    @Body() dto: UpdateRunConfigDto,
  ): Promise<ReturnType<RunConfigService['upsert']>> {
    return this.runConfig.upsert(dto);
  }
}
