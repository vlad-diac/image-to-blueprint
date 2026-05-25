import { ValidationPipe } from '@nestjs/common';
import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';

async function bootstrap() {
  const app = await NestFactory.create(AppModule, { rawBody: false });
  app.enableCors({
    origin: process.env.WEB_ORIGIN?.split(',').filter(Boolean) ?? ['http://localhost:5173'],
    credentials: true,
  });
  app.useGlobalPipes(
    new ValidationPipe({
      whitelist: true,
      transform: true,
      forbidNonWhitelisted: false,
      transformOptions: { enableImplicitConversion: true },
    }),
  );

  const port = Number(process.env.PORT ?? '3001');
  await app.listen(port);
  console.log(`API listening on http://localhost:${port}`);
}

void bootstrap();
