"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const common_1 = require("@nestjs/common");
const core_1 = require("@nestjs/core");
const app_module_1 = require("./app.module");
async function bootstrap() {
    const app = await core_1.NestFactory.create(app_module_1.AppModule, { rawBody: false });
    app.enableCors({
        origin: process.env.WEB_ORIGIN?.split(',').filter(Boolean) ?? ['http://localhost:5173'],
        credentials: true,
    });
    app.useGlobalPipes(new common_1.ValidationPipe({
        whitelist: true,
        transform: true,
        forbidNonWhitelisted: false,
        transformOptions: { enableImplicitConversion: true },
    }));
    const port = Number(process.env.PORT ?? '3001');
    await app.listen(port);
    console.log(`API listening on http://localhost:${port}`);
}
void bootstrap();
//# sourceMappingURL=main.js.map