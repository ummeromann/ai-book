---
id: 002
title: Clarify Book Scope and Depth
stage: spec
date: 2025-12-22
surface: agent
model: claude-opus-4-5-20251101
feature: 002-book-structure
branch: 002-book-structure
user: A R COMPUTER
command: /sp.clarify
labels: ["clarification", "scope", "simulation-first", "tool-boundaries"]
links:
  spec: specs/002-book-structure/spec.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/002-book-structure/spec.md
  - specs/002-book-structure/checklists/requirements.md
tests:
  - N/A
---

## Prompt

Clarify scope and depth.

- Simulation-first approach; real hardware is optional
- Gazebo and Isaac Sim are mandatory
- Unity is used for visualization, not control
- ROS 2 Humble/Iron assumed
- NVIDIA Isaac used for perception, navigation, and training
- GPT models used only for planning, not motor control
- Capstone runs fully in simulation with optional Jetson deployment

## Response snapshot

Updated spec with 7 key clarifications:

**New Scope & Boundaries Section**:
- Added In Scope / Out of Scope tables
- Added Tool Responsibilities ASCII diagram showing stack layers

**Updated Sections**:
1. User Story 3 (Simulation Engineer): Unity now explicitly "visualization only"
2. User Story 5 (VLA Researcher): LLMs for "planning, not motor control"
3. User Story 7 (Capstone): "fully in simulation, Jetson optional"
4. Module 2 description: Gazebo/Isaac MANDATORY, Unity VISUALIZATION ONLY
5. Module 4 description: "LLMs for planning (NOT motor control)"
6. Module 6 description: "RUNS FULLY IN SIMULATION", "HARDWARE OPTIONAL"
7. Assumptions: Added simulation-first, ROS 2 version, hardware optional

**New Functional Requirements** (FR-013 to FR-016):
- FR-013: All examples runnable in simulation without hardware
- FR-014: Gazebo/Isaac mandatory; Unity visualization only
- FR-015: LLM content focuses on planning, NOT motor control
- FR-016: Capstone runs in simulation; Jetson optional

**Checklist Updated**:
- Added Clarification Log table tracking all 7 clarifications
- All 16 checklist items pass

## Outcome

- Impact: Spec scope and tool boundaries now explicit and unambiguous
- Tests: N/A (documentation clarification)
- Files: spec.md (updated), checklists/requirements.md (updated with clarification log)
- Next prompts: /sp.plan to create implementation plan
- Reflection: Clarifications prevent scope creep and set clear expectations for readers about simulation-first approach

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): Quality checklist - PASS (16/16 items)
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A
