"use strict";
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
var __metadata = (this && this.__metadata) || function (k, v) {
    if (typeof Reflect === "object" && typeof Reflect.metadata === "function") return Reflect.metadata(k, v);
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.RunConfigService = void 0;
const common_1 = require("@nestjs/common");
const prisma_service_1 = require("../prisma/prisma.service");
const run_config_helpers_1 = require("./run-config.helpers");
const DEFAULT_POSITIVE = '<sks> front view elevated shot medium shot Using the reference vessel photo, generate a naval recognition-chart style top view of the ship.  Style: - simplified maritime blueprint - monochrome technical drawing - clean orthographic projection - white background - black ink linework  Preserve: - exact hull shape - superstructure placement - crane locations - mast and antenna positions - crow’s nest location - deck segmentation - large visible equipment  Simplify all details into readable geometric forms suitable for a ship recognition manual.  Do not include: - perspective - lighting - shadows - sea/waves - realistic textures - people - atmospheric effects - labels - decorative elements  The output should feel like an official naval silhouette reference sheet.';
const DEFAULT_ID = 'default';
let RunConfigService = class RunConfigService {
    constructor(prisma) {
        this.prisma = prisma;
    }
    async getOrCreate() {
        const existing = await this.prisma.runConfiguration.findUnique({
            where: { id: DEFAULT_ID },
        });
        if (existing)
            return (0, run_config_helpers_1.serializeRunConfiguration)(existing);
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
        return (0, run_config_helpers_1.serializeRunConfiguration)(created);
    }
    async upsert(dto) {
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
        return (0, run_config_helpers_1.serializeRunConfiguration)(row);
    }
};
exports.RunConfigService = RunConfigService;
exports.RunConfigService = RunConfigService = __decorate([
    (0, common_1.Injectable)(),
    __metadata("design:paramtypes", [prisma_service_1.PrismaService])
], RunConfigService);
//# sourceMappingURL=run-config.service.js.map