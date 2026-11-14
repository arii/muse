---
id: boomtick
title: Boomtick
---

# Boomtick: agent tooling, out of the box

## The tooling your agents wish they had

<!-- visual: title -->

Boomtick is a template repository that hands your autonomous agents the tooling real projects need. No glue code, no re-invented workflows, no key hunting.

Frontier SF — made by Ariel Anders using obvious.ai.

## Three pillars

<!-- visual: bullets -->

- A Model Context Protocol server for GitHub operations
- The T D command line for local automation
- An AI review framework with blast radius analysis

Boomtick ships three pillars. The MCP server exposes GitHub operations to any agent host. The T D command line is the local fallback tier. And the review framework measures blast radius before code lands.

## Wire it up in one command

<!-- visual: terminal -->

The MCP server gives agents structured access to pull requests, repository state, and CI logs. One command registers it with Claude Code, Cursor, and VS Code.

```bash
$ ./setup-agent.sh
$ td mcp install --execute
$ node mcp/dist/index.js
```

Integration is zero submodule: the server ships as an npm package, a docker image, and a composite action.

## Rescue pull requests, safely

<!-- visual: bullets -->

- Scouts conflicted pull requests and overlapping branches
- Applies minimal repairs in isolated git worktrees
- Proves the fix with test and Playwright suites

The rescue agent scouts conflicted pull requests, applies minimal repairs in isolated git worktrees, and proves the fix with real verification suites. Every mutating action sits behind explicit write guards, and every shell command runs through a token redacting allowlist. Your working directory is never touched.

## Reviews that see the blast radius

<!-- visual: split -->

The review framework pairs GitHub Models for low latency code review with Gemini for multi-modal visual interface analysis. It is routed through LangChain, with retrieval augmented triage built in.

```bash
$ pnpm run impact:analysis
$ pnpm run impact:gemini-review
$ pnpm run impact:visual-diff
```

## Clone it. Wire it. Ship it.

<!-- visual: title -->

Boomtick already powers boomtick dot blog. Clone the repo, run the setup script, and give your agents the tooling they deserve.
