-- CreateTable
CREATE TABLE "RunConfiguration" (
    "id" TEXT NOT NULL,
    "steps" INTEGER NOT NULL DEFAULT 4,
    "cfg" DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    "positivePrompt" TEXT NOT NULL,
    "negativePrompt" TEXT NOT NULL DEFAULT '',
    "numImages" INTEGER NOT NULL DEFAULT 3,
    "useLightningLora" BOOLEAN NOT NULL DEFAULT true,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "RunConfiguration_pkey" PRIMARY KEY ("id")
);

-- Seed default row (matches apps/web/src/lib/demoDefaults.ts DEMO_DEFAULTS).
INSERT INTO "RunConfiguration" (
    "id",
    "steps",
    "cfg",
    "positivePrompt",
    "negativePrompt",
    "numImages",
    "useLightningLora",
    "updatedAt"
)
VALUES (
    'default',
    4,
    1.0,
    $p$<sks> front view elevated shot medium shot Using the reference vessel photo, generate a naval recognition-chart style top view of the ship.  Style: - simplified maritime blueprint - monochrome technical drawing - clean orthographic projection - white background - black ink linework  Preserve: - exact hull shape - superstructure placement - crane locations - mast and antenna positions - crow’s nest location - deck segmentation - large visible equipment  Simplify all details into readable geometric forms suitable for a ship recognition manual.  Do not include: - perspective - lighting - shadows - sea/waves - realistic textures - people - atmospheric effects - labels - decorative elements  The output should feel like an official naval silhouette reference sheet.$p$,
    '',
    3,
    true,
    CURRENT_TIMESTAMP
)
ON CONFLICT ("id") DO NOTHING;
