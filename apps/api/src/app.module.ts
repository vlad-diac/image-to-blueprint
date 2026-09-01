import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { ScheduleModule } from '@nestjs/schedule';
import { AvailabilityModule } from './availability/availability.module';
import { RunConfigModule } from './config/run-config.module';
import { PrismaModule } from './prisma/prisma.module';
import { RunpodModule } from './runpod/runpod.module';
import { RunsModule } from './runs/runs.module';

@Module({
  imports: [
    ConfigModule.forRoot({ isGlobal: true }),
    ScheduleModule.forRoot(),
    PrismaModule,
    RunConfigModule,
    RunpodModule,
    RunsModule,
    AvailabilityModule,
  ],
})
export class AppModule {}
