"use strict";
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.RunsModule = void 0;
const common_1 = require("@nestjs/common");
const platform_express_1 = require("@nestjs/platform-express");
const runpod_module_1 = require("../runpod/runpod.module");
const runs_controller_1 = require("./runs.controller");
const runs_service_1 = require("./runs.service");
const runs_sweep_service_1 = require("./runs-sweep.service");
let RunsModule = class RunsModule {
};
exports.RunsModule = RunsModule;
exports.RunsModule = RunsModule = __decorate([
    (0, common_1.Module)({
        imports: [
            runpod_module_1.RunpodModule,
            platform_express_1.MulterModule.register({
                limits: { fileSize: 40 * 1024 * 1024 },
            }),
        ],
        controllers: [runs_controller_1.RunsController],
        providers: [runs_service_1.RunsService, runs_sweep_service_1.RunsSweepService],
    })
], RunsModule);
//# sourceMappingURL=runs.module.js.map