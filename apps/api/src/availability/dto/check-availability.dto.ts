import { ArrayNotEmpty, IsArray, IsString } from 'class-validator';

export class CheckAvailabilityDto {
  /** Pool ids to run `check_gpu_availability.py` for (e.g. ["ADA_80_PRO"]). */
  @IsArray()
  @ArrayNotEmpty()
  @IsString({ each: true })
  pools!: string[];
}
