---
id: 004
title: Generate Actionable Chapter Tasks
stage: tasks
date: 2025-12-22
surface: agent
model: claude-opus-4-5-20251101
feature: 002-book-structure
branch: 002-book-structure
user: A R COMPUTER
command: /sp.tasks
labels: ["tasks", "chapters", "deployment", "docusaurus"]
links:
  spec: specs/002-book-structure/spec.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/002-book-structure/tasks.md
tests:
  - N/A
---

## Prompt

Generate actionable tasks.

For each chapter:
- Write markdown content
- Add diagrams placeholders
- Add code snippets
- Add real-world examples
- Add chapter summary

Global Tasks:
- Configure Docusaurus sidebar
- Add images folder
- Setup GitHub Actions for deployment
- Final review and polish

## Response snapshot

Generated comprehensive tasks.md with 345 total tasks organized in 10 phases:

**Phase Breakdown:**
| Phase | Tasks | Purpose |
|-------|-------|---------|
| 1: Setup | 12 | Docusaurus initialization |
| 2: Foundational | 15 | Module directories |
| 3: US1 | 37 | Module 0 (Ch 1-3) |
| 4: US2 | 55 | Module 1 (Ch 4-7) |
| 5: US3 | 37 | Module 2 (Ch 8-10) |
| 6: US4 | 49 | Module 3 (Ch 11-14) |
| 7: US5 | 50 | Module 4 (Ch 15-18) |
| 8: US6 | 50 | Module 5 (Ch 19-22) |
| 9: US7 | 26 | Module 6 (Ch 23-24) |
| 10: Deploy | 14 | GitHub Pages + Vercel |

**Per-Chapter Task Pattern:**
1. Front matter (title, description, sidebar_position, tags)
2. Learning objectives (3-5 measurable)
3. Prerequisites section
4. Main content sections
5. Code snippets with language specified
6. Humanoid example (specific robot)
7. Diagram placeholders (<!-- DIAGRAM: -->)
8. Chapter summary (3-5 takeaways)
9. Key Concepts glossary (if 5+ terms)

**Global Tasks Included:**
- Docusaurus sidebar configuration (T005)
- Images folder structure (T006-T007)
- GitHub Actions deployment (T332)
- Vercel deployment (T333)
- Final validation (T335-T345)

**Parallel Opportunities:** ~40% of tasks marked [P]

**Suggested MVP:** Phases 1-4 (119 tasks) for Modules 0-1

## Outcome

- Impact: Complete 345-task execution plan for 24-chapter book
- Tests: N/A (task generation)
- Files: tasks.md created
- Next prompts: /sp.implement to begin execution, or manual task execution
- Reflection: Tasks granular enough for LLM execution, includes all constitution requirements

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): Task format validation - PASS (all use checklist format)
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A
