import { Type } from 'class-transformer';
import {
  IsBoolean,
  IsNotEmpty,
  IsNumber,
  IsOptional,
  IsString,
  Max,
  Min,
} from 'class-validator';

export class UpdateRunConfigDto {
  @IsString()
  @IsNotEmpty()
  positivePrompt!: string;

  @IsOptional()
  @IsString()
  negativePrompt?: string;

  @Type(() => Number)
  @IsNumber()
  @Min(1)
  @Max(100)
  steps!: number;

  @Type(() => Number)
  @IsNumber()
  cfg!: number;

  @Type(() => Number)
  @IsNumber()
  @Min(1)
  @Max(4)
  numImages!: number;

  @IsBoolean()
  useLightningLora!: boolean;
}
