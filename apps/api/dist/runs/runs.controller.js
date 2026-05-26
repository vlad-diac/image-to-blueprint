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
var __param = (this && this.__param) || function (paramIndex, decorator) {
    return function (target, key) { decorator(target, key, paramIndex); }
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.RunsController = void 0;
const common_1 = require("@nestjs/common");
const platform_express_1 = require("@nestjs/platform-express");
const runs_service_1 = require("./runs.service");
const create_run_multipart_dto_1 = require("./dto/create-run-multipart.dto");
let RunsController = class RunsController {
    constructor(runs) {
        this.runs = runs;
    }
    async list(limitRaw) {
        const limit = limitRaw ? Number(limitRaw) : 20;
        return this.runs.listRecent(Number.isFinite(limit) ? limit : 20);
    }
    async create(file, dto) {
        return this.runs.createWithImage(file?.buffer, {
            positivePrompt: dto.positivePrompt,
            negativePrompt: dto.negativePrompt,
            steps: dto.steps,
            cfg: dto.cfg,
            seed: dto.seed,
        });
    }
    async cancel(id) {
        return this.runs.cancel(id);
    }
    async input(id) {
        const buf = await this.runs.getInputBytes(id);
        return new common_1.StreamableFile(buf, { type: 'image/png' });
    }
    async output(id) {
        const buf = await this.runs.getOutputBytes(id);
        if (!buf)
            throw new common_1.NotFoundException('Output not ready');
        return new common_1.StreamableFile(buf, { type: 'image/png' });
    }
    async getOne(id) {
        return this.runs.findOne(id, true);
    }
};
exports.RunsController = RunsController;
__decorate([
    (0, common_1.Get)(),
    __param(0, (0, common_1.Query)('limit')),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [String]),
    __metadata("design:returntype", Promise)
], RunsController.prototype, "list", null);
__decorate([
    (0, common_1.Post)(),
    (0, common_1.UseInterceptors)((0, platform_express_1.FileInterceptor)('image', { limits: { fileSize: 40 * 1024 * 1024 } })),
    __param(0, (0, common_1.UploadedFile)()),
    __param(1, (0, common_1.Body)()),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [Object, create_run_multipart_dto_1.CreateRunMultipartDto]),
    __metadata("design:returntype", Promise)
], RunsController.prototype, "create", null);
__decorate([
    (0, common_1.Post)(':id/cancel'),
    (0, common_1.HttpCode)(200),
    __param(0, (0, common_1.Param)('id')),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [String]),
    __metadata("design:returntype", Promise)
], RunsController.prototype, "cancel", null);
__decorate([
    (0, common_1.Get)(':id/input.png'),
    __param(0, (0, common_1.Param)('id')),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [String]),
    __metadata("design:returntype", Promise)
], RunsController.prototype, "input", null);
__decorate([
    (0, common_1.Get)(':id/output.png'),
    __param(0, (0, common_1.Param)('id')),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [String]),
    __metadata("design:returntype", Promise)
], RunsController.prototype, "output", null);
__decorate([
    (0, common_1.Get)(':id'),
    __param(0, (0, common_1.Param)('id')),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [String]),
    __metadata("design:returntype", Promise)
], RunsController.prototype, "getOne", null);
exports.RunsController = RunsController = __decorate([
    (0, common_1.Controller)('runs'),
    __metadata("design:paramtypes", [runs_service_1.RunsService])
], RunsController);
//# sourceMappingURL=runs.controller.js.map