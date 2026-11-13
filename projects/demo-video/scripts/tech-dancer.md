---
id: tech-dancer
title: BoomTick
---

# BoomTick demo

## One repo, two products

<!-- visual: title -->

BoomTick is two products in one repo: a content platform that ships to boomtick dot blog, and a developer toolkit that keeps it honest.

## A real content platform

<!-- visual: bullets -->

The platform is a Vite-powered React application covering West Coast Swing dance guides, gear reviews, research, and merch.

- West Coast Swing guides and research
- Curated gear reviews and merch
- Structured data and SEO baked in

## Designed for readers

<!-- visual: bullets -->

The home page pairs a hero with a featured guide, a topic grid, and the latest posts.

- Hero and featured guide panel
- Topic grid and latest posts
- Gear shelves connect content to the shop

## Built like a product

<!-- visual: split -->

Every build runs a typed pipeline: type checking, asset generation, sitemaps, and merchant feeds, all wired into one command.

```bash
$ pnpm build
✓ type-check
✓ asset generation
✓ sitemap + merchant feed
✓ dist ready
```

## Meet the td CLI

<!-- visual: terminal -->

The td CLI is the unified entry point for repository automation. One command checks environment health across node, pnpm, and python, and preflight verification runs before anything ships.

```bash
$ td doctor
✓ node runtime
✓ pnpm
✓ git safe directory
✓ GitHub app credentials

$ td preflight
✓ all checks passed
```

## Made for agents

<!-- visual: split -->

For agents, BoomTick exposes an MCP server with structured tools. The repo follows an MCP-first policy: prefer tools over raw shell commands, and self-correct when you drift.

```bash
$ npx @arii/boomtick-mcp
✓ structured tools exposed
✓ agents coordinated
```

## Guarded by quality gates

<!-- visual: terminal -->

Quality is enforced locally and in CI: linting with zero warnings, unit and end-to-end tests, static analysis, and lighthouse audits for performance.

```bash
$ pnpm ci:local
✓ lint: oxlint + eslint
✓ vitest + playwright
✓ semgrep + knip
✓ lighthouse
```

## BoomTick

<!-- visual: title -->

BoomTick: dance content on the front end, agent-grade automation behind it. One repository, one pipeline, from story to shipped.
