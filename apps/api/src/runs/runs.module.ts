import { Module } from '@nestjs/common';
import { MulterModule } from '@nestjs/platform-express';
import { RunpodModule } from '../runpod/runpod.module';
import { RunsController } from './runs.controller';
import { RunsService } from './runs.service';
import { RunsSweepService } from './runs-sweep.service';

@Module({
  imports: [
    RunpodModule,
    MulterModule.register({
      limits: { fileSize: 40 * 1024 * 1024 },
    }),
  ],
  controllers: [RunsController],
  providers: [RunsService, RunsSweepService],
})
export class RunsModule {}
