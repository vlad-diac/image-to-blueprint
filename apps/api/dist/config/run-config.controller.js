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
exports.RunConfigController = void 0;
const common_1 = require("@nestjs/common");
const update_run_config_dto_1 = require("./dto/update-run-config.dto");
const run_config_service_1 = require("./run-config.service");
let RunConfigController = class RunConfigController {
    constructor(runConfig) {
        this.runConfig = runConfig;
    }
    async get() {
        return this.runConfig.getOrCreate();
    }
    async put(dto) {
        return this.runConfig.upsert(dto);
    }
};
exports.RunConfigController = RunConfigController;
__decorate([
    (0, common_1.Get)(),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", []),
    __metadata("design:returntype", Promise)
], RunConfigController.prototype, "get", null);
__decorate([
    (0, common_1.Put)(),
    __param(0, (0, common_1.Body)()),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [update_run_config_dto_1.UpdateRunConfigDto]),
    __metadata("design:returntype", Promise)
], RunConfigController.prototype, "put", null);
exports.RunConfigController = RunConfigController = __decorate([
    (0, common_1.Controller)('config'),
    __metadata("design:paramtypes", [run_config_service_1.RunConfigService])
], RunConfigController);
//# sourceMappingURL=run-config.controller.js.map