#!/usr/bin/env bun

const [type, subject, body] = process.argv.slice(2)

const TYPES = {
    feat: "A new feature",
    fix: "A bug fix",
    docs: "Documentation only changes",
    style: "Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)",
    refactor: "A code change that neither fixes a bug nor adds a feature",
    perf: "A code change that improves performance",
    test: "Adding missing tests or correcting existing tests",
    chore: "Changes to the build process or auxiliary tools and libraries such as documentation generation",
    ci: "Changes to our CI configuration files and scripts",
    build: "Changes that affect the build system or external dependencies",
    revert: "Reverts a previous commit",
} as const

type CommitType = keyof typeof TYPES

if (!type || !subject) {
    console.log("Available commit types:")
    Object.entries(TYPES).forEach(([key, desc]) => {
        console.log(`  ${key}: ${desc}`)
    })
    console.log(
        "\nUsage: bun run .husky/commit-agent.ts <type> <subject> [body]",
    )
    console.log(
        'Example: bun run .husky/commit-agent.ts feat "add onClick prop to component"',
    )
    process.exit(1)
}

if (!TYPES[type as CommitType]) {
    console.log(`Error: Invalid commit type "${type}"`)
    console.log("Available types:", Object.keys(TYPES).join(", "))
    process.exit(1)
}

const commitMessage = body
    ? `${type}: ${subject}\n\n${body}`
    : `${type}: ${subject}`

console.log(`Committing with message:\n${commitMessage}`)

import { $ } from "bun"

try {
    await $`git commit -m ${commitMessage}`.env({ ...process.env, COMMIT_AGENT: "1" })
} catch (error) {
    console.error("Commit failed. Make sure you have staged files first.")
    process.exit(1)
}
