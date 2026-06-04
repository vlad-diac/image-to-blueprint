import { Module } from '@nestjs/common';
import { RunConfigController } from './run-config.controller';
import { RunConfigService } from './run-config.service';

@Module({
  controllers: [RunConfigController],
  providers: [RunConfigService],
})
export class RunConfigModule {}
