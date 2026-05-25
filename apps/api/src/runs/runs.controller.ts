import {
  Body,
  Controller,
  Get,
  HttpCode,
  NotFoundException,
  Param,
  Post,
  Query,
  StreamableFile,
  UploadedFile,
  UseInterceptors,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import type { Express } from 'express';
import { RunsService } from './runs.service';
import { CreateRunMultipartDto } from './dto/create-run-multipart.dto';

@Controller('runs')
export class RunsController {
  constructor(private readonly runs: RunsService) {}

  @Get()
  async list(@Query('limit') limitRaw?: string) {
    const limit = limitRaw ? Number(limitRaw) : 20;
    return this.runs.listRecent(Number.isFinite(limit) ? limit : 20);
  }

  @Post()
  @UseInterceptors(FileInterceptor('image', { limits: { fileSize: 40 * 1024 * 1024 } }))
  async create(
    @UploadedFile() file: Express.Multer.File,
    @Body() dto: CreateRunMultipartDto,
  ) {
    return this.runs.createWithImage(file?.buffer, {
      positivePrompt: dto.positivePrompt,
      negativePrompt: dto.negativePrompt,
      steps: dto.steps,
      cfg: dto.cfg,
      seed: dto.seed,
    });
  }

  @Post(':id/cancel')
  @HttpCode(200)
  async cancel(@Param('id') id: string) {
    return this.runs.cancel(id);
  }

  @Get(':id/input.png')
  async input(@Param('id') id: string): Promise<StreamableFile> {
    const buf = await this.runs.getInputBytes(id);
    return new StreamableFile(buf, { type: 'image/png' });
  }

  @Get(':id/output.png')
  async output(@Param('id') id: string): Promise<StreamableFile> {
    const buf = await this.runs.getOutputBytes(id);
    if (!buf) throw new NotFoundException('Output not ready');
    return new StreamableFile(buf, { type: 'image/png' });
  }

  @Get(':id')
  async getOne(@Param('id') id: string) {
    return this.runs.findOne(id, true);
  }
}
