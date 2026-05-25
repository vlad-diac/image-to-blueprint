import { Injectable } from '@nestjs/common';
import { Cron } from '@nestjs/schedule';
import { RunsService } from './runs.service';

@Injectable()
export class RunsSweepService {
  constructor(private readonly runs: RunsService) {}

  /** Backstop reconcile for in-flight rows (client may stop polling mid-run). */
  @Cron('*/3 * * * * *')
  sweep(): void {
    void this.runs.sweepStaleInFlight(12);
  }
}
