/** Default run preset for Demo mode — matches the project's vessel → blueprint POC. */

export interface DemoPreset {
  positivePrompt: string;
  negativePrompt: string;
  steps: number;
  cfg: number;
  /** null ⇒ worker picks distinct random seeds per output */
  seed: number | null;
  numImages: number;
}

export const DEMO_DEFAULTS: DemoPreset = {
  positivePrompt:
    '<sks> front view elevated shot medium shot Using the reference vessel photo, generate a naval recognition-chart style top view of the ship.  Style: - simplified maritime blueprint - monochrome technical drawing - clean orthographic projection - white background - black ink linework  Preserve: - exact hull shape - superstructure placement - crane locations - mast and antenna positions - crow’s nest location - deck segmentation - large visible equipment  Simplify all details into readable geometric forms suitable for a ship recognition manual.  Do not include: - perspective - lighting - shadows - sea/waves - realistic textures - people - atmospheric effects - labels - decorative elements  The output should feel like an official naval silhouette reference sheet.',
  negativePrompt: '',
  steps: 4,
  cfg: 1,
  seed: null,
  numImages: 3,
};
