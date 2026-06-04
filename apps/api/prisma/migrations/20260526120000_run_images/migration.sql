-- CreateTable RunImage first (references Run).
CREATE TABLE "RunImage" (
    "id" TEXT NOT NULL,
    "runId" TEXT NOT NULL,
    "index" INTEGER NOT NULL,
    "bytes" BYTEA NOT NULL,
    "seed" BIGINT,
    "width" INTEGER,
    "height" INTEGER,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "RunImage_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX "RunImage_runId_index_key" ON "RunImage"("runId", "index");

CREATE INDEX "RunImage_runId_idx" ON "RunImage"("runId");

ALTER TABLE "RunImage"
ADD CONSTRAINT "RunImage_runId_fkey"
FOREIGN KEY ("runId") REFERENCES "Run"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- Migrate existing outputs into rows (preserve prior single-output runs).
ALTER TABLE "Run"
ADD COLUMN "numImages" INTEGER NOT NULL DEFAULT 3;

INSERT INTO "RunImage" ("id", "runId", "index", "bytes", "seed", "width", "height", "createdAt")
SELECT
    gen_random_uuid()::TEXT,
    "id",
    0,
    "outputImage",
    NULL,
    NULL,
    NULL,
    NOW()
FROM "Run"
WHERE "outputImage" IS NOT NULL;

ALTER TABLE "Run" DROP COLUMN "outputImage";
