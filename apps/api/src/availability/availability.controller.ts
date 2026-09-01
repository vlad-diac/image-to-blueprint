import { Body, Controller, Get, Post } from '@nestjs/common';
import { AvailabilityService } from './availability.service';
import type { AvailabilityDto } from './availability.types';
import { CheckAvailabilityDto } from './dto/check-availability.dto';

@Controller('availability')
export class AvailabilityController {
  constructor(private readonly availability: AvailabilityService) {}

  @Get()
  get(): Promise<AvailabilityDto> {
    return this.availability.getAvailability();
  }

  /** Run the check script for the given pools, then return fresh data. */
  @Post('check')
  async check(@Body() dto: CheckAvailabilityDto): Promise<AvailabilityDto> {
    await this.availability.runCheck(dto.pools);
    return this.availability.getAvailability();
  }
}
