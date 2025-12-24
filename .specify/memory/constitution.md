<!--
Sync Impact Report
==================
Version change: 0.0.0 → 1.0.0 (MAJOR - initial constitution creation)

Modified principles: N/A (new document)

Added sections:
- Core Principles (6 principles for Physical AI book)
- Content Standards (Docusaurus-compatible technical writing)
- Development Workflow (chapter lifecycle and review)
- Governance (amendment and compliance rules)

Removed sections: N/A

Templates requiring updates:
- .specify/templates/plan-template.md - ✅ No changes needed (Constitution Check section is generic)
- .specify/templates/spec-template.md - ✅ No changes needed (user story format is compatible)
- .specify/templates/tasks-template.md - ✅ No changes needed (task format is compatible)

Follow-up TODOs: None
-->

# Physical AI & Humanoid Robotics Constitution

## Core Principles

### I. Practical-First Learning

All content MUST prioritize hands-on, applied learning over abstract theory. Every concept introduced MUST be tied to a real-world humanoid robotics use case (e.g., Boston Dynamics, Tesla Optimus, Figure 01, Unitree). Theoretical foundations are permitted only when directly enabling practical implementation. Chapters MUST include runnable code examples, simulation exercises, or hardware interaction guides.

**Rationale**: Readers learn robotics by doing. Excessive theory without application leads to poor retention and inability to apply knowledge to real systems.

### II. Progressive Complexity

Content MUST follow a beginner-to-advanced progression within each chapter and across the book. Each chapter MUST begin with foundational concepts accessible to newcomers before introducing advanced topics. Prerequisites MUST be explicitly stated at the start of each chapter. Learning objectives MUST be measurable and testable.

**Rationale**: Physical AI spans multiple disciplines (ML, control systems, perception, hardware). A clear progression prevents overwhelming beginners while ensuring advanced readers find depth.

### III. Toolchain Integration

All tutorials and examples MUST use industry-standard tools: ROS 2 (Humble or later), Gazebo (Ignition/Fortress+), Unity with ML-Agents or Robotics packages, NVIDIA Isaac Sim/Gym, and Vision-Language-Action (VLA) models. Code examples MUST specify exact versions and dependencies. Docker/containerization MUST be provided for reproducibility. Platform-specific instructions MUST cover Linux (primary), with Windows/WSL2 guidance where applicable.

**Rationale**: Readers expect to learn tools they will use professionally. Standardizing on production toolchains ensures skills transfer directly to industry.

### IV. Docusaurus Compatibility

All content MUST be authored in Markdown fully compatible with Docusaurus 3.x. Files MUST use `.md` or `.mdx` extensions. Front matter MUST include: title, description, sidebar_position, and tags. Admonitions (:::note, :::tip, :::warning, :::danger) MUST be used for callouts. Code blocks MUST specify language for syntax highlighting. Internal links MUST use relative paths. Images MUST be stored in `/static/img/` with descriptive alt text.

**Rationale**: Docusaurus is the publishing platform. Non-compliant content breaks the build pipeline and degrades reader experience.

### V. Visual & Structural Clarity

Each chapter MUST include: (1) a chapter summary at the end (3-5 key takeaways), (2) diagram suggestions clearly marked with `<!-- DIAGRAM: description -->` comments, (3) a "Key Concepts" glossary section if introducing 5+ new terms. Tables MUST be used for comparing options (e.g., simulation platforms, robot specifications). Mermaid diagrams SHOULD be used for architecture and flow visualizations.

**Rationale**: Technical robotics content is dense. Visual aids and structured summaries dramatically improve comprehension and serve as reference material.

### VI. Example-Driven Humanoid Focus

Every chapter MUST include at least one concrete humanoid robotics example (e.g., bipedal walking, manipulation, human-robot interaction). Examples MUST reference specific robot platforms where applicable (Digit, Atlas, Optimus, H1, NAO, Pepper). Generic robotic arm or mobile base examples MUST explicitly connect back to humanoid applications. VLA and embodied AI concepts MUST be demonstrated with humanoid-specific scenarios.

**Rationale**: The book's unique value proposition is humanoid robotics. Generic robotics content without humanoid application fails to deliver on the book's promise.

## Content Standards

### Writing Style

- Use active voice and second person ("you will learn", "you can configure")
- Explain acronyms on first use in each chapter
- Limit sentences to 25 words maximum for technical instructions
- Use numbered steps for procedures, bullets for concepts
- Provide expected output/results after code examples

### Code Standards

- All code MUST be tested and runnable
- Python code MUST follow PEP 8 with type hints
- C++ code MUST follow ROS 2 style guidelines
- Include comments explaining non-obvious logic
- Provide GitHub repository links for complete implementations

### Multimedia Guidelines

- Diagrams: System architecture, data flow, robot kinematics
- Screenshots: IDE setup, simulation environments, visualization tools
- GIFs/Videos: Motion sequences, real-time demonstrations (linked, not embedded)
- All media MUST have descriptive captions

## Development Workflow

### Chapter Lifecycle

1. **Outline**: Define learning objectives, prerequisites, and section structure
2. **Draft**: Write content following all constitution principles
3. **Technical Review**: Verify code examples, tool versions, and accuracy
4. **Editorial Review**: Check clarity, progression, and Docusaurus compatibility
5. **Integration**: Add to book structure, verify cross-references and navigation

### Quality Gates

- All code examples MUST execute without errors
- Docusaurus build MUST succeed with no warnings
- At least one diagram suggestion per major section
- Chapter summary MUST exist before merge
- Prerequisites MUST link to relevant earlier chapters

## Governance

This constitution is the authoritative source for all content decisions in the Physical AI & Humanoid Robotics book project. All contributors, chapters, and reviews MUST comply with these principles.

### Amendment Process

1. Propose amendment via pull request with rationale
2. Amendments require explicit approval from project maintainer(s)
3. Breaking changes (principle removal/redefinition) require migration plan for affected content
4. All amendments MUST update the version and Last Amended date

### Versioning Policy

- **MAJOR**: Backward-incompatible principle changes or removals
- **MINOR**: New principles added or existing principles materially expanded
- **PATCH**: Clarifications, typo fixes, non-semantic refinements

### Compliance Review

- Every chapter PR MUST include constitution compliance checklist
- Reviewers MUST verify adherence to all 6 core principles
- Non-compliant content MUST be revised before merge

**Version**: 1.0.0 | **Ratified**: 2025-12-22 | **Last Amended**: 2025-12-22
