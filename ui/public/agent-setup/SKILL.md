---
name: tavily
description: |
  Install, authenticate, and route Tavily for live web search, content extraction, site mapping and crawling, deep research, application integration, Agent Skills, Claude Code, or MCP clients. Use when the user needs Tavily onboarding, wants Tavily available inside a coding agent, needs current web data through the Tavily CLI, or wants to add Tavily to application code. Make setup autonomous: reuse existing credentials first, redeem a supplied one-time Tavily setup token when present, otherwise open browser OAuth automatically. Resolve local runtime prerequisites without changing the user's defaults, install Agent Skills globally for the actual target agent, and verify both Tavily and the agent integration before declaring setup complete.
---

# Tavily

Use Tavily for real-time web search, clean URL extraction, website mapping and crawling, and multi-source deep research.

## Skill Segments

Map Tavily usage to three different jobs:

| Segment      | Question it answers                                                    | Where the work runs                                      | Status          |
| ------------ | ---------------------------------------------------------------------- | -------------------------------------------------------- | --------------- |
| CLI skills   | "Which Tavily command should I run right now?"                         | In the agent's own terminal session                      | Available       |
| Build skills | "How do I add Tavily to this codebase?"                                | Inside the user's product code                           | Available       |
| Blueprints   | "What's the finished deliverable and how do I produce it with Tavily?" | In the agent's session, producing an outcome or artifact | TBD / N/A today |

Use CLI skills when the agent itself needs web data now. Use `tavily-best-practices` when adding Tavily to product code that will keep running after the agent session ends. Do not claim that a separate Tavily Blueprints bundle exists today; route cited research deliverables through `tavily-research` / `tvly research` until Blueprints exist.

## Autonomous Setup

Treat setup as one state machine: **preflight -> CLI -> authentication -> Agent Skills -> verification**. Do not make the user perform a step that can be completed from the terminal.

### 1. Preflight

Inspect the machine before installing anything:

1. Detect the current coding agent. If setup targets multiple agents, detect the requested installed agents and configure only those targets.
2. Check whether `tvly` already exists with `command -v tvly`.
3. If `tvly` exists, run `tvly auth --json` immediately. Reuse authenticated state; never replace a working credential unnecessarily.
4. Check for a supplied setup token before starting browser authentication. Accept it from the invoking instructions, `TAVILY_SETUP_TOKEN`, or the surrounding platform. Never print it.
5. Check Node.js before any path that uses `npx`, including both `tvly login` and Agent Skills installation.
6. If the active Node runtime is incompatible but a newer compatible runtime is already installed through `nvm`, `fnm`, `mise`, `volta`, `asdf`, or another local manager, use it for the Tavily setup command only. Do not change the user's default Node version.
7. Surface a manual prerequisite only when automation is genuinely blocked. State only the blocker and the minimum action required.

### Current Node compatibility note

The current Tavily CLI browser login delegates OAuth to:

```text
npx -y mcp-remote https://mcp.tavily.com/mcp
```

The CLI suppresses that subprocess's stdout and stderr while waiting for OAuth state. In some Node 18 environments, `mcp-remote` can fail before OAuth starts and the failure appears only as a Tavily login timeout. When an installed Node 20+ runtime is available, prefer it for `tvly login` and `npx skills` rather than troubleshooting Node 18. Do not change the user's global/default Node version just to complete Tavily setup.

### 2. Install the Tavily CLI

If `tvly` is missing, run:

```bash
curl -fsSL https://cli.tavily.com/install.sh | bash
```

Then verify that the current process can resolve it:

```bash
command -v tvly
```

If the installer succeeded but `tvly` is not on the current `PATH`, recover automatically when possible. For example, if `~/.local/bin/tvly` exists, prepend `~/.local/bin` to the current process `PATH` and check again. Do not edit shell startup files unless necessary.

### 3. Authenticate

Always run:

```bash
tvly auth --json
```

If it reports `authenticated: true`, reuse that credential and continue.

If it is not authenticated:

1. If a setup token is available, redeem it first and save the returned existing Tavily API key securely to Tavily CLI config.
2. If no setup token exists, or redemption fails, run `tvly login` **in the foreground** and let the normal Tavily browser authorization flow complete.
3. If the active Node runtime is older/incompatible and an installed Node 20+ runtime is available, run `tvly login` with that runtime prepended to the command's `PATH` only.
4. Wait for the human to approve Tavily in the browser. Waiting for human consent is not an error condition.
5. Run `tvly auth --json` again and do not continue to authenticated-only capabilities until it returns `authenticated: true`.

Never ask the user to find an API key or manually add `TAVILY_API_KEY` to `.env` for CLI onboarding.

Do not background `tvly login`. Do not inspect Tavily CLI source, reconstruct OAuth URLs, inject shell shims, or use browser automation while the normal browser flow is still available. If browser login times out, first retry once with a compatible installed Node 20+ runtime. If that still cannot open or complete the browser flow, report the authentication bridge as the blocker rather than falling back to manual API-key copy/paste.

Use the bundled helper when local script execution is available:

```bash
python scripts/tavily_auth_bootstrap.py
```

If a setup token is available:

```bash
python scripts/tavily_auth_bootstrap.py --setup-token "$TAVILY_SETUP_TOKEN"
```

Never print, echo, log, commit, or include the setup token or resulting API key in chat output.

### Setup-token bootstrap contract

Treat a setup token as a short-lived, one-time credential supplied by an authenticated Tavily surface such as the Playground. It is not a normal Tavily API key.

When a setup token is present and `tvly auth --json` says the machine is not authenticated:

1. `POST` it to `https://api.tavily.com/v1/setup-tickets/redeem` with JSON `{ "ticket": "..." }`. If Tavily deploys a different endpoint, use `TAVILY_SETUP_REDEEM_URL`.
2. Expect the user's existing Tavily API key as `api_key`.
3. Persist it to `~/.tavily/config.json` with owner-only permissions and the current Tavily CLI storage format. Never put the key in shell history or print it.
4. Discard the setup token and in-memory key as soon as they are no longer needed.
5. Verify with `tvly auth --json`.
6. If redemption fails because the token expired, was already used, or the endpoint is unavailable, fall back automatically to browser OAuth.

The bundled `scripts/tavily_auth_bootstrap.py` implements this authentication flow. Do not claim that the public `tvly` CLI has a `--setup-token` option unless that option has actually shipped.

### 4. Install Agent Skills

For coding-agent onboarding, install Tavily Agent Skills **globally** so Tavily works across projects instead of only in the repository that happened to be open during setup.

Target the actual agent or agents being configured. Do not use a project-local install unless the user explicitly asks for project scope.

Example for Claude Code:

```bash
npx skills add tavily-ai/skills \
  --skill '*' \
  --agent claude-code \
  --global \
  --yes
```

For multiple detected targets, enumerate only those targets:

```bash
npx skills add tavily-ai/skills \
  --skill '*' \
  --agent claude-code \
  --agent codex \
  --agent cursor \
  --global \
  --yes
```

If the active Node runtime cannot run the Skills CLI, use an already-installed compatible Node 20+ runtime for this command without changing the user's default Node version.

The official Tavily skills currently include:

- `tavily-cli` — route between Tavily CLI commands
- `tavily-search` — discover web sources and current information
- `tavily-extract` — extract clean content from known URLs
- `tavily-map` — discover URLs on a site
- `tavily-crawl` — extract many pages from a site section
- `tavily-research` — run cited multi-source research
- `tavily-dynamic-search` — use dynamic Tavily search routing when present in the installed skills source
- `tavily-best-practices` — build Tavily into application code

### 5. Verify every layer

Verify installation layers separately instead of treating one successful API call as proof that everything is ready.

CLI:

```bash
command -v tvly
tvly --version
```

Authentication:

```bash
tvly auth --json
```

Live Tavily request:

```bash
tvly search "Tavily Search API" --json
```

Agent Skills, for example Claude Code:

```bash
npx skills list --global --agent claude-code
```

Confirm that the expected Tavily skills are installed for the target agent. If the current agent session exposes a skill inventory, confirm that `tavily-search` is actually visible/active there too.

Only report **Tavily ready** when the applicable states are true:

```text
Tavily CLI installed
Authentication verified
Live Tavily Search verified
Agent Skills installed
Agent Skills active in current session
```

If skills are installed but the current client needs a restart/rescan, say **"installed; restart/rescan required"** rather than claiming they are already active.

## Install Surfaces

Choose the best surface for the user's goal. Avoid installing duplicate surfaces unless the user asks for them.

### Generic coding agent: CLI + Agent Skills

Use the Autonomous Setup above. This is the default when the user wants Tavily available to a coding agent across projects.

### Claude Code: native plugin

If the user explicitly wants Tavily's native Claude plugin and slash-command experience, install:

```bash
claude plugin install tavily@claude-plugins-official
```

The plugin expects Tavily credentials in Claude Code's environment/settings. If setup-token redemption produced a raw key, configure it without displaying the key. If only Tavily browser OAuth is available and no raw key is exposed, prefer the Tavily Remote MCP path rather than asking the user to copy a key manually.

Do not automatically install both the native plugin and generic Agent Skills into Claude Code unless the user wants both surfaces.

### MCP client: remote Tavily MCP

Use the hosted MCP server when the client should use Tavily as an MCP tool surface:

```text
https://mcp.tavily.com/mcp/
```

For Claude Code with OAuth:

```bash
claude mcp add tavily-remote-mcp --transport http https://mcp.tavily.com/mcp/
```

For keyless Search + Extract in Claude Code:

```bash
claude mcp add tavily-remote-mcp --transport http https://mcp.tavily.com/mcp/ \
  --header "X-Tavily-Access-Mode: keyless"
```

## Choose Your Path

- **Need web data during this session** -> Path A (CLI skills)
- **Need to add Tavily to application code** -> Path B (Build skills)
- **Need Tavily inside Claude Code, Cursor, Codex, or another client** -> Path C (agent/client integration)
- **Need a cited multi-source research report** -> Path D (Research; closest current equivalent to a Blueprint)
- **Need to authenticate** -> use the Autonomous Setup authentication state machine
- **Do not want to install the CLI** -> Path E (REST API or MCP directly)

---

## Path A: CLI Skills / Live Web Tools

Use the Tavily CLI when the agent itself needs current web data during the current session.

Default escalation pattern:

1. **Search** for discovery or current sources.
2. **Extract** when URLs are already known.
3. **Map** when the site is known but the right pages are not.
4. **Crawl** for content from many pages in a site or section.
5. **Research** for a cited synthesis across many sources.

Prefer structured output for agentic work:

```bash
tvly search "latest AI agent releases" --json
tvly extract "https://example.com/article" --json
tvly map "https://docs.example.com" --json
tvly crawl "https://docs.example.com" --output-dir ./docs/
tvly research "competitive landscape for AI coding agents" --json
```

Hand off to the matching installed skill when available: `tavily-search`, `tavily-extract`, `tavily-map`, `tavily-crawl`, or `tavily-research`.

Search and Extract can work keyless. Map, Crawl, and Research require authentication.

---

## Path B: Build Skills / Integrate Tavily Into an App

Use this when Tavily will run inside the user's product after the current agent session ends.

Do not use the CLI as the application's runtime integration. Inspect the project's language and conventions, then use the official SDK or REST API.

Python:

```bash
pip install tavily-python
```

JavaScript / TypeScript:

```bash
npm install @tavily/core
```

If the application needs a raw Tavily API key and one is available, configure it in the project's existing secret mechanism yourself rather than making the user discover `.env` setup. For a project that already uses a gitignored `.env` or `.env.local`, add `TAVILY_API_KEY` there; otherwise follow the project's existing secret-manager/runtime convention. Never commit or print the key.

Use `tavily-best-practices` for implementation guidance. Run one real request as a smoke test before reporting the integration works.

---

## Path C: Connect Tavily to an Agent or Client

Choose the client's best native surface instead of installing every possible integration.

### Claude Code

For the native Tavily plugin:

```bash
claude plugin install tavily@claude-plugins-official
```

For generic Tavily Agent Skills across all Claude Code projects:

```bash
npx skills add tavily-ai/skills --skill '*' --agent claude-code --global --yes
```

For browser OAuth through a remote tool surface:

```bash
claude mcp add tavily-remote-mcp --transport http https://mcp.tavily.com/mcp/
```

Choose one primary surface unless the user explicitly wants multiple.

### Cursor, Codex, Windsurf, Cline, and other coding agents

Use global Agent Skills when the client supports them. Target the real client explicitly with `--agent`; use MCP when the client is primarily an MCP host or centralized OAuth/tool configuration is preferable.

After setup, verify a real Tavily Search and verify the skills are installed for the target agent.

---

## Path D: Research / Blueprint Placeholder

There is currently no separate Tavily Blueprints bundle. For a finished cited research deliverable, use Tavily Research:

```bash
tvly research "Analyze the competitive landscape for AI coding assistants" --model pro
```

For machine-readable output:

```bash
tvly research "Analyze the competitive landscape for AI coding assistants" --json
```

For asynchronous work:

```bash
tvly research "topic" --no-wait --json
tvly research status <request_id> --json
tvly research poll <request_id> --json
```

Research requires authentication.

---

## Path E: Use Tavily Without Installing the CLI

### Direct REST API with an API key

Base URL:

```text
https://api.tavily.com
```

Primary endpoints are `/search`, `/extract`, `/crawl`, `/map`, and `/research`.

### Direct keyless Search / Extract

Send `X-Tavily-Access-Mode: keyless` and omit the API key. Keyless access is rate-limited. Crawl, Map, and Research still require authentication.

## Current CLI Surface

The current top-level Tavily CLI surface is:

```text
tvly
├── login
├── logout
├── auth
├── search
├── extract
├── crawl
├── map
└── research
    ├── run
    ├── status
    └── poll
```

Before using an unfamiliar option, run:

```bash
tvly --help
tvly <command> --help
```

## After Successful Setup: Turn Setup Into Useful Work

After Tavily is fully installed and verified, use the conversation, reliable prior knowledge, and local workspace to move the user into useful work. Do not mechanically show every user the same command demo.

Before asking a next-step question, inspect the active application for an explicit, implementation-ready current-web gap. A provider interface, documented missing web-search/research component, focused TODO, or intentionally unconfigured discovery service is stronger evidence than a merely non-empty working directory. When Tavily directly fits that gap, treat the active application as selected and application-integration intent as clear. Ask exactly one concise confirmation before modifying the project, such as: `This project has a clear current-web gap that Tavily fits. Should I add Tavily now?` After an affirmative response, continue immediately through Path B with `tavily-best-practices`, implement the provider through the project's native architecture, and test the real application surface. Do not present a menu or ask for another confirmation. Ask a focused scoping question before confirmation only when different implementations would materially change product behavior or the work would cross the active project's boundary.

First classify the context:

1. **Mid-session.** There is a concrete task or goal already in progress in the current conversation. Relate the handoff directly to that task. If it needed current web data, extraction, crawling, mapping, or research, continue it with Tavily. If it was about adding web capability to the application already being edited, continue in that application with `tavily-best-practices`. Do not replace the task with a menu, ask the user to restate it, or treat a non-empty working directory by itself as proof that a task is in progress.
2. **New session with reliable prior knowledge.** There is no active task in the current conversation, but the agent already has useful context from available memory, recent conversations supplied to the session, known projects, the current workspace, or accessible recent-project signals. Use the strongest relevant signals to personalize the next step. An explicit implementation-ready gap in the active application follows the immediate-integration rule above; weaker signals should only personalize the next choice. For example, recommend a known active project first, tailor a research suggestion to a known domain, or prioritize the most relevant start-from-scratch use case. Present weak inferences as recommendations and keep an alternative available; do not invent missing context, overstate confidence, or narrate private background unnecessarily.
3. **New session with no prior knowledge.** There is no active task and no reliable context about the user's projects or interests. Use the rough default selector below. Do not delay the handoff with extra discovery questions.

When a new session still needs the user to choose a direction, personalize this selector from reliable prior knowledge when possible. Otherwise use it exactly as the rough default:

```text
Tavily is ready. What would you like to do?

1. Use Tavily with me now — search, extract, crawl, map, or research.
2. Add Tavily to an existing project — show my recent repositories or let me enter a local folder.
3. Start something from scratch — show me a few ideas.

Reply with 1, 2, or 3.
```

Route `1` into the user's immediate web task, `2` into the local-project selector below, and `3` into the start-from-scratch use-case selector. If the user replies with only a number, follow that route immediately without restating the choices or asking for confirmation. Do not ask them to explain context that is already available.

### Select a local project

When application-integration intent is clear but the target project is not, offer the current project, recently active Git repositories, a custom local path, and a no-project route.

Build the list without installing an unrelated directory-tracking tool:

1. Prefer the current working directory or its Git root when it contains an application.
2. Search only a small, bounded set of conventional development locations already accessible to the agent, such as `~/Developer`, `~/Projects`, `~/src`, `~/Desktop`, and `~/Documents`. Do not recursively scan the entire home directory or request broader filesystem access solely to populate this menu.
3. For each discovered Git repository, resolve the worktree-safe HEAD reflog path:

   ```bash
   git -C "$repo" rev-parse --path-format=absolute --git-path logs/HEAD
   ```

4. Rank repositories by that file's modification time (`stat -f '%m'` on macOS or `stat -c '%Y'` on Linux). This approximates recent HEAD-changing Git activity such as commits, branch switches, merges, rebases, resets, and pulls that advanced HEAD. It does not prove recent editing and does not include status, diff, fetch, or an already-up-to-date pull.
5. Deduplicate Git roots, keep at most four recent repositories in addition to the current project, and omit entries whose activity cannot be resolved. Display short project names rather than full personal paths; add a parent label only when names collide.

Use a compact selector shaped like this:

```text
Tavily is ready. Where should I add it?

1. current-app — current project
2. recent-project — recently active
3. another-project — recently active
4. Another local repository or folder — paste its path
5. No — help me start from scratch

Reply with a number, or paste a local path.
```

Adjust the numbering to the available projects. Accept an existing local directory even when it is not a Git repository. If no recent repositories can be found, show only the current project when applicable, the custom-path option, and the no-project option.

### Integrate into the selected project

When the user selects a listed project, pastes a local repository or folder, or otherwise names a local project:

1. Resolve and validate the directory without printing secrets or unnecessarily exposing the full personal path.
2. Inspect the project's language, framework, architecture, existing web/data layer, package manager, testing conventions, and secret-management pattern.
3. Infer the intended Tavily capability from the conversation and codebase. If it is clear, ask only the single implementation confirmation below. If the product behavior is genuinely ambiguous and different choices would materially change the implementation, ask one focused question before confirmation.
4. Use `tavily-best-practices` as the implementation guide. Add Tavily through the official SDK or REST API in the application's native runtime; do not use the Tavily CLI as the product's runtime integration.
5. Put credentials in the project's existing secret mechanism, never commit or print them, and preserve unrelated local changes.
6. Run proportionate tests and one real Tavily request through the application's actual integration surface before claiming it works. Distinguish source inspection, local tests, and a verified live request.

Selecting or naming the project identifies the integration target. Ask exactly one concise confirmation before changing it; after an affirmative response, begin without a second menu or confirmation.

An active application with an explicit, implementation-ready current-web gap counts as already selected even in a fresh session. Ask the single implementation confirmation instead of asking the user to select that same application.

### Start from scratch: choose a use case

If the user chooses `3` from the direction selector or **No** from the local-project selector, ask exactly one multiple-choice question:

```text
What would you like to work on?

1. Academic research
2. People research
3. Product and competitor monitoring

Reply with 1, 2, or 3.
```

When the user selects a use case, immediately install only its mapped skill:

1. **Academic research** -> `academic-scientific-research`

   ```bash
   npx skills add https://github.com/tavily-ai/use-case-skills --skill academic-scientific-research
   ```

2. **People research** -> `sales-account-intelligence`

   ```bash
   npx skills add https://github.com/tavily-ai/use-case-skills --skill sales-account-intelligence
   ```

3. **Product and competitor monitoring** -> `product-competitor-intelligence`

   ```bash
   npx skills add https://github.com/tavily-ai/use-case-skills --skill product-competitor-intelligence
   ```

Run the selected command in the foreground. If the Skills CLI asks for the target agent or install scope, select the actual current agent and global scope so the skill remains available across projects. Do not install the other two use-case skills.

After installation:

1. Verify that the selected skill is installed for the target agent.
2. Check whether it is active in the current session. If the session loaded its skill inventory before installation, say **"installed; restart/rescan required"** rather than claiming it is active.
3. Give the user one concise first-use instruction matched to the installed skill:

   - Academic research: `Use academic-scientific-research to find, screen, and synthesize papers about <topic>.`
   - People research: `Use sales-account-intelligence to research <person> and their company, role, background, and relevant signals.`
   - Product and competitor monitoring: `Use product-competitor-intelligence to compare <product or company> with its competitors and surface important changes.`

Tell the user they can replace the angle-bracketed text with their subject. If the selected skill is already active, invite them to send that request now. Do not return to the generic Tavily menu or ask them to choose a second use case.

### Post-setup interaction rules

- Optimize for continuation and useful outcomes, not demonstration of every Tavily capability.
- Use conversation and workspace context naturally without narrating how much personal context is known.
- In a new session, personalize from reliable prior knowledge before using rough defaults; never fabricate prior context merely to avoid a generic choice.
- Choose Search, Extract, Map, Crawl, or Research for the user when the distinction does not change their decision.
- Ask at most one next-step question in each response.
- Do not repeat setup instructions after successful verification.
- Keep the handoff short.
