import { Injectable } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import type { UpdateRunConfigDto } from './dto/update-run-config.dto';
import {
  serializeRunConfiguration,
  type RunConfigurationSerializable,
} from './run-config.helpers';

/** Matches apps/web seeded defaults / migration seed when DB is empty. */
const DEFAULT_POSITIVE =
  '<sks> front view elevated shot medium shot Using the reference vessel photo, generate a naval recognition-chart style top view of the ship.  Style: - simplified maritime blueprint - monochrome technical drawing - clean orthographic projection - white background - black ink linework  Preserve: - exact hull shape - superstructure placement - crane locations - mast and antenna positions - crow’s nest location - deck segmentation - large visible equipment  Simplify all details into readable geometric forms suitable for a ship recognition manual.  Do not include: - perspective - lighting - shadows - sea/waves - realistic textures - people - atmospheric effects - labels - decorative elements  The output should feel like an official naval silhouette reference sheet.';

const DEFAULT_ID = 'default' as const;

@Injectable()
export class RunConfigService {
  constructor(private readonly prisma: PrismaService) {}

  async getOrCreate(): Promise<RunConfigurationSerializable> {
    const existing = await this.prisma.runConfiguration.findUnique({
      where: { id: DEFAULT_ID },
    });
    if (existing) return serializeRunConfiguration(existing);

    const created = await this.prisma.runConfiguration.create({
      data: {
        id: DEFAULT_ID,
        positivePrompt: DEFAULT_POSITIVE,
        negativePrompt: '',
        steps: 4,
        cfg: 1.0,
        numImages: 3,
        useLightningLora: true,
      },
    });
    return serializeRunConfiguration(created);
  }

  async upsert(dto: UpdateRunConfigDto): Promise<RunConfigurationSerializable> {
    const row = await this.prisma.runConfiguration.upsert({
      where: { id: DEFAULT_ID },
      create: {
        id: DEFAULT_ID,
        positivePrompt: dto.positivePrompt.trim(),
        negativePrompt: (dto.negativePrompt ?? '').trim(),
        steps: dto.steps,
        cfg: dto.cfg,
        numImages: dto.numImages,
        useLightningLora: dto.useLightningLora,
      },
      update: {
        positivePrompt: dto.positivePrompt.trim(),
        negativePrompt: (dto.negativePrompt ?? '').trim(),
        steps: dto.steps,
        cfg: dto.cfg,
        numImages: dto.numImages,
        useLightningLora: dto.useLightningLora,
      },
    });
    return serializeRunConfiguration(row);
  }
}
