// import.meta.webpackContext is provided by the bundler (Rspack, via
// remotion.config.ts) and is what makes dynamic composition discovery work
// without Root edits.

interface WebpackContext<T = unknown> {
  keys(): string[];
  (id: string): T;
}

interface WebpackImportMeta {
  webpackContext<T = unknown>(
    directory: string,
    options?: { recursive?: boolean; regExp?: RegExp; mode?: string }
  ): WebpackContext<T>;
}

declare global {
  interface ImportMeta extends WebpackImportMeta {}
}

export {};
