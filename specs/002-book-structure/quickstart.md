# Quickstart: Contributing to Physical AI Book

**Feature**: 002-book-structure
**Date**: 2025-12-22
**Audience**: Book contributors and reviewers

## Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| Node.js | 18+ | Docusaurus runtime |
| npm or yarn | Latest | Package management |
| Git | 2.30+ | Version control |
| VS Code | Latest | Recommended editor |

### Optional (for code examples)

| Software | Version | Purpose |
|----------|---------|---------|
| Docker | 24+ | Isolated environments |
| ROS 2 | Humble | Robot middleware |
| Python | 3.10+ | Code examples |

## Quick Setup

### 1. Clone Repository

```bash
git clone https://github.com/{org}/ai-book.git
cd ai-book
```

### 2. Install Dependencies

```bash
npm install
```

### 3. Start Development Server

```bash
npm run start
```

Open http://localhost:3000 to see the book.

### 4. Build for Production

```bash
npm run build
```

## Writing a Chapter

### Step 1: Create Chapter File

```bash
# Copy template to appropriate module
cp specs/002-book-structure/contracts/chapter-template.md \
   docs/module-{N}-{slug}/ch{NN}-{slug}.md
```

### Step 2: Fill Front Matter

```yaml
---
id: ch04-ros2-architecture
title: "ROS 2 Architecture & Core Concepts"
sidebar_position: 2
description: "Learn ROS 2 architecture including DDS, QoS, packages, and workspaces for humanoid robot development."
tags: [ros2, dds, architecture, middleware]
---
```

### Step 3: Write Content Following Constitution

Checklist for each chapter:

- [ ] **Practical-First**: Includes runnable code/simulation example
- [ ] **Progressive Complexity**: Begins with basics, advances to complex
- [ ] **Toolchain Integration**: Specifies exact versions
- [ ] **Docusaurus Compatible**: Valid front matter, admonitions, code blocks
- [ ] **Visual Clarity**: Has diagram placeholders, chapter summary
- [ ] **Humanoid Focus**: Includes specific humanoid robot example

### Step 4: Add Diagram Placeholders

```markdown
<!-- DIAGRAM: id="ros2-node-architecture" type="architecture" format="mermaid"
     description="ROS 2 node communication showing publishers and subscribers" -->
```

### Step 5: Create Code Examples

```bash
# Create directory for chapter examples
mkdir -p code-examples/module-1-ros2/ch04-architecture

# Add example code
cat > code-examples/module-1-ros2/ch04-architecture/minimal_publisher.py << 'EOF'
#!/usr/bin/env python3
"""Minimal ROS 2 publisher example."""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        # ... rest of implementation
EOF
```

### Step 6: Validate Build

```bash
# Check for build errors
npm run build

# Check for broken links
npm run serve
# Then manually test links
```

## File Structure Reference

```
ai-book/
├── docs/
│   ├── module-0-foundations/
│   │   ├── _category_.json      # Module metadata
│   │   ├── index.md             # Module introduction
│   │   ├── ch01-intro-physical-ai.md
│   │   ├── ch02-digital-to-physical.md
│   │   └── ch03-humanoid-landscape.md
│   └── ...
├── code-examples/
│   └── module-1-ros2/
│       └── ch04-architecture/
│           └── minimal_publisher.py
├── static/
│   └── img/                     # Images and diagrams
├── docusaurus.config.ts
├── sidebars.ts
└── package.json
```

## Constitution Compliance Checklist

Before submitting a chapter, verify:

### I. Practical-First Learning
- [ ] Includes at least one runnable code example
- [ ] Tied to real humanoid robotics use case
- [ ] Minimal theory, maximum application

### II. Progressive Complexity
- [ ] Starts with beginner-friendly introduction
- [ ] Prerequisites explicitly stated
- [ ] Learning objectives are measurable

### III. Toolchain Integration
- [ ] Specifies exact tool versions
- [ ] Works with ROS 2 Humble
- [ ] Docker environment provided (if applicable)

### IV. Docusaurus Compatibility
- [ ] Valid YAML front matter
- [ ] Uses admonitions (:::note, :::tip, etc.)
- [ ] Code blocks specify language
- [ ] Relative internal links

### V. Visual & Structural Clarity
- [ ] Chapter summary (3-5 takeaways)
- [ ] At least one diagram placeholder
- [ ] Key Concepts section if 5+ new terms
- [ ] Tables for comparisons

### VI. Example-Driven Humanoid Focus
- [ ] Specific humanoid robot example (Atlas, Optimus, Digit, H1, etc.)
- [ ] Connects to humanoid applications

## Common Tasks

### Add a New Module

1. Create module directory:
   ```bash
   mkdir docs/module-{N}-{slug}
   ```

2. Add `_category_.json`:
   ```json
   {
     "label": "Module {N}: {Title}",
     "position": {N+1},
     "link": {
       "type": "doc",
       "id": "module-{N}-{slug}/index"
     }
   }
   ```

3. Create `index.md` from template

4. Update `sidebars.ts`

### Add Images

1. Place images in `static/img/{module}/`
2. Reference in Markdown:
   ```markdown
   ![Alt text](/img/module-1-ros2/ros2-architecture.png)
   ```

### Create Mermaid Diagrams

```markdown
```mermaid
graph TD
    A[Node A] -->|publishes| B[Topic]
    B -->|subscribes| C[Node B]
```​
```

### Use Admonitions

```markdown
:::note
Standard note for additional information.
:::

:::tip Helpful Tip
Pro tip for readers.
:::

:::warning
Something to be careful about.
:::

:::danger
Critical warning - mistakes here cause problems.
:::
```

## Deployment

### GitHub Pages (Automatic)

Pushes to `main` trigger automatic deployment via GitHub Actions.

### Vercel Preview

Each PR gets a preview deployment URL from Vercel.

### Manual Build

```bash
npm run build
npm run serve  # Local preview of production build
```

## Getting Help

- Check existing chapters for examples
- Review constitution at `.specify/memory/constitution.md`
- Open an issue for questions
- Join Discord for real-time help

## Review Process

1. Create branch: `git checkout -b chapter/ch{NN}-{slug}`
2. Write chapter following this guide
3. Run `npm run build` to validate
4. Open PR with constitution checklist
5. Address reviewer feedback
6. Merge when approved
