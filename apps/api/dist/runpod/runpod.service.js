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
exports.RunpodService = void 0;
const common_1 = require("@nestjs/common");
const config_1 = require("@nestjs/config");
let RunpodService = class RunpodService {
    constructor(config) {
        this.config = config;
        const id = this.config.getOrThrow('RUNPOD_ENDPOINT_ID');
        this.apiKey = this.config.getOrThrow('RUNPOD_API_KEY');
        this.baseUrl = `https://api.runpod.ai/v2/${id}`;
    }
    headers() {
        return {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${this.apiKey}`,
        };
    }
    async submit(input) {
        const res = await fetch(`${this.baseUrl}/run`, {
            method: 'POST',
            headers: this.headers(),
            body: JSON.stringify({ input }),
        });
        const text = await res.text();
        if (!res.ok) {
            throw new Error(`RunPod /run failed ${res.status}: ${text}`);
        }
        const json = JSON.parse(text);
        const id = (json.id ?? json.jobId);
        if (!id) {
            throw new Error(`RunPod /run unexpected body: ${text}`);
        }
        return { id, status: json.status };
    }
    async status(jobId) {
        const res = await fetch(`${this.baseUrl}/status/${jobId}`, {
            method: 'GET',
            headers: { Authorization: `Bearer ${this.apiKey}` },
        });
        const text = await res.text();
        if (!res.ok) {
            throw new Error(`RunPod /status failed ${res.status}: ${text}`);
        }
        return JSON.parse(text);
    }
    async cancel(jobId) {
        const res = await fetch(`${this.baseUrl}/cancel/${jobId}`, {
            method: 'POST',
            headers: this.headers(),
            body: JSON.stringify({}),
        });
        const text = await res.text();
        if (!res.ok) {
            throw new Error(`RunPod /cancel failed ${res.status}: ${text}`);
        }
        return JSON.parse(text);
    }
};
exports.RunpodService = RunpodService;
exports.RunpodService = RunpodService = __decorate([
    (0, common_1.Injectable)(),
    __metadata("design:paramtypes", [config_1.ConfigService])
], RunpodService);
//# sourceMappingURL=runpod.service.js.map