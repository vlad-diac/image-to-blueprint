import { Module } from '@nestjs/common';
import { RunpodService } from './runpod.service';

@Module({
  providers: [RunpodService],
  exports: [RunpodService],
})
export class RunpodModule {}
