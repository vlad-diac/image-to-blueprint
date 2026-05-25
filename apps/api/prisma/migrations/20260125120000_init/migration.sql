-- CreateEnum
CREATE TYPE "RunStatus" AS ENUM ('QUEUED', 'IN_QUEUE', 'IN_PROGRESS', 'SUCCEEDED', 'FAILED', 'CANCELLED', 'TIMED_OUT');

-- CreateTable
CREATE TABLE "Run" (
    "id" TEXT NOT NULL,
    "status" "RunStatus" NOT NULL DEFAULT 'QUEUED',
    "positivePrompt" TEXT NOT NULL,
    "negativePrompt" TEXT NOT NULL DEFAULT '',
    "steps" INTEGER NOT NULL DEFAULT 4,
    "cfg" DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    "seed" BIGINT,
    "inputImage" BYTEA NOT NULL,
    "outputImage" BYTEA,
    "runpodJobId" TEXT,
    "workerJobDir" TEXT,
    "durationMs" INTEGER,
    "delayMs" INTEGER,
    "executionMs" INTEGER,
    "errorMessage" TEXT,
    "rawStatus" JSONB,
    "startedAt" TIMESTAMP(3),
    "completedAt" TIMESTAMP(3),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "Run_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "Run_runpodJobId_key" ON "Run"("runpodJobId");

-- CreateIndex
CREATE INDEX "Run_status_updatedAt_idx" ON "Run"("status", "updatedAt");
