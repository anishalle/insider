import { cpSync, mkdirSync, rmSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import * as esbuild from "esbuild";

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = resolve(__dirname, "..");
const srcDir = join(rootDir, "src");
const distDir = join(rootDir, "dist");
const watch = process.argv.includes("--watch");

rmSync(distDir, { force: true, recursive: true });
mkdirSync(distDir, { recursive: true });

const sharedBuildOptions = {
  bundle: true,
  format: "esm",
  minify: !watch,
  sourcemap: watch,
  target: "chrome120",
  logLevel: "info",
};

const buildOnce = async () => {
  await esbuild.build({
    ...sharedBuildOptions,
    entryPoints: {
      background: join(srcDir, "background/index.ts"),
      content: join(srcDir, "content/index.ts"),
      sidepanel: join(srcDir, "sidepanel/index.ts"),
    },
    outdir: distDir,
  });
  cpSync(join(srcDir, "manifest.json"), join(distDir, "manifest.json"));
  cpSync(join(srcDir, "sidepanel/index.html"), join(distDir, "sidepanel.html"));
  cpSync(join(srcDir, "sidepanel/styles.css"), join(distDir, "sidepanel.css"));
};

const run = async () => {
  if (!watch) {
    await buildOnce();
    return;
  }

  const context = await esbuild.context({
    ...sharedBuildOptions,
    entryPoints: {
      background: join(srcDir, "background/index.ts"),
      content: join(srcDir, "content/index.ts"),
      sidepanel: join(srcDir, "sidepanel/index.ts"),
    },
    outdir: distDir,
  });
  await context.watch();
  await buildOnce();
  // Keep the process alive in watch mode.
  await new Promise(() => {});
};

run().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
