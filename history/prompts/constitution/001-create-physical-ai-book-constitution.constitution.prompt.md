---
id: 001
title: Create Physical AI Book Constitution
stage: constitution
date: 2025-12-22
surface: agent
model: claude-opus-4-5-20251101
feature: none
branch: 001-physical-ai-book
user: A R COMPUTER
command: /sp.constitution
labels: ["constitution", "physical-ai", "humanoid-robotics", "docusaurus"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - .specify/memory/constitution.md
tests:
  - N/A
---

## Prompt

Create a constitution for a Docusaurus-based technical book titled
"Physical AI & Humanoid Robotics".

Goals:
- Teach Physical AI and Embodied Intelligence
- Bridge digital AI and physical humanoid robots
- Focus on ROS 2, Gazebo, Unity, NVIDIA Isaac, and Vision-Language-Action

Rules:
- Content must be beginner-to-advanced
- Use clear explanations with real-world humanoid examples
- Use Markdown compatible with Docusaurus
- Include diagrams suggestions and summaries per chapter
- Avoid unnecessary theory; stay practical and applied

## Response snapshot

Created constitution v1.0.0 for the Physical AI & Humanoid Robotics book with 6 core principles:

1. **Practical-First Learning** - Hands-on over theory, real humanoid examples required
2. **Progressive Complexity** - Beginner-to-advanced progression, explicit prerequisites
3. **Toolchain Integration** - ROS 2, Gazebo, Unity, NVIDIA Isaac, VLA models with version specs
4. **Docusaurus Compatibility** - Markdown/MDX format, proper front matter, admonitions
5. **Visual & Structural Clarity** - Summaries, diagram placeholders, glossaries, Mermaid diagrams
6. **Example-Driven Humanoid Focus** - Concrete humanoid examples (Atlas, Optimus, Digit, etc.)

Also included: Content Standards, Development Workflow, and Governance sections.

## Outcome

- Impact: Initial constitution established for book project
- Tests: N/A (documentation artifact)
- Files: .specify/memory/constitution.md (created)
- Next prompts: Create book outline spec, set up Docusaurus project structure
- Reflection: Constitution covers all user requirements - practical focus, toolchain standards, Docusaurus compatibility, and humanoid-specific examples

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A
