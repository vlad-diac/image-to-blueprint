import { Type } from 'class-transformer';
import {
  IsNotEmpty,
  IsNumber,
  IsOptional,
  IsString,
  Max,
  Min,
} from 'class-validator';

export class CreateRunMultipartDto {
  @IsString()
  @IsNotEmpty()
  positivePrompt!: string;

  @IsOptional()
  @IsString()
  negativePrompt?: string;

  @IsOptional()
  @Type(() => Number)
  @IsNumber()
  @Min(1)
  @Max(100)
  steps?: number;

  @IsOptional()
  @Type(() => Number)
  @IsNumber()
  cfg?: number;

  @IsOptional()
  @Type(() => Number)
  @IsNumber()
  seed?: number;
}
