import {
  BadRequestException,
  Body,
  Controller,
  Get,
  HttpCode,
  NotFoundException,
  Param,
  Post,
  Query,
  Res,
  StreamableFile,
  UploadedFile,
  UseInterceptors,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import type { Express, Response } from 'express';
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
      numImages: dto.numImages,
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

  /** Redirect to index 0 for callers that still use the legacy URL. */
  @Get(':id/output.png')
  legacyOutput(@Param('id') _id: string, @Res() res: Response): void {
    res.redirect(302, `outputs/0.png`);
  }

  @Get(':id/outputs/:file')
  async outputAt(
    @Param('id') id: string,
    @Param('file') file: string,
  ): Promise<StreamableFile> {
    const m = /^(\d+)\.png$/i.exec(file);
    if (!m)
      throw new BadRequestException('Outputs path must look like outputs/<index>.png');
    const index = Number(m[1]);
    const buf = await this.runs.getOutputBytesAt(id, index);
    if (!buf) throw new NotFoundException('Output not ready');
    return new StreamableFile(buf, { type: 'image/png' });
  }

  @Get(':id')
  async getOne(@Param('id') id: string) {
    return this.runs.findOne(id, true);
  }
}
