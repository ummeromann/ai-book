---
id: ch{NN}-{slug}
title: "{Chapter Title}"
sidebar_position: {N}
description: "{SEO description 150-160 characters describing what the reader will learn}"
tags: [{tag1}, {tag2}, {tag3}]
---

# {Chapter Title}

<!-- DIAGRAM: id="{chapter-overview}" type="architecture" format="mermaid"
     description="High-level overview of {topic} showing key components and relationships" -->

## Learning Objectives

By the end of this chapter, you will be able to:

1. {Measurable objective 1 - use action verbs: explain, implement, configure, demonstrate}
2. {Measurable objective 2}
3. {Measurable objective 3}
4. {Optional objective 4}
5. {Optional objective 5}

## Prerequisites

Before starting this chapter, ensure you have:

- Completed [Chapter N: {Title}](../module-{X}-{slug}/ch{NN}-{slug}.md)
- {Specific knowledge or skill required}
- {Tool or environment setup required}

## Introduction

{2-3 paragraphs introducing the chapter topic}

{Explain why this topic matters for humanoid robotics}

{Connect to the reader's learning journey - what they learned before and what's coming}

:::note Simulation-First
All examples in this chapter run in simulation. No physical hardware required.
:::

## {Section 1 Title}

{Section content with beginner-friendly introduction}

### {Subsection 1.1}

{Detailed content}

```python
# Example code with language specified
# Include comments explaining non-obvious logic
def example_function():
    """Docstring explaining purpose."""
    pass
```

**Expected Output**:
```
{What the reader should see when running the code}
```

:::tip {Helpful Tip Title}
{Practical advice or shortcut}
:::

### {Subsection 1.2}

{More detailed content building on previous subsection}

<!-- DIAGRAM: id="{section-specific-diagram}" type="flow" format="mermaid"
     description="{What this diagram shows}" -->

## {Section 2 Title}

{Progress to intermediate concepts}

### Humanoid Example: {Specific Robot or Application}

{Concrete example using a real humanoid robot: Atlas, Optimus, Digit, H1, NAO, or Pepper}

```python
# Code specific to humanoid application
# Reference specific robot platform
```

:::warning Common Pitfall
{Something readers often get wrong and how to avoid it}
:::

## {Section 3 Title}

{Advanced concepts for readers who want to go deeper}

| Option A | Option B | Option C |
|----------|----------|----------|
| {Pro 1}  | {Pro 1}  | {Pro 1}  |
| {Con 1}  | {Con 1}  | {Con 1}  |

## Hands-On Exercise

{Guided exercise that reinforces the chapter's learning objectives}

1. {Step 1 with specific instructions}
2. {Step 2}
3. {Step 3}

**Challenge**: {Optional advanced exercise for ambitious readers}

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| {Common error 1} | {Why it happens} | {How to fix it} |
| {Common error 2} | {Why it happens} | {How to fix it} |

## Key Concepts

{Include this section only if chapter introduces 5+ new terms}

| Term | Definition |
|------|------------|
| **{Term 1}** | {Clear, concise definition} |
| **{Term 2}** | {Clear, concise definition} |
| **{Term 3}** | {Clear, concise definition} |

## Chapter Summary

In this chapter, you learned:

1. **{Key takeaway 1}**: {Brief explanation}
2. **{Key takeaway 2}**: {Brief explanation}
3. **{Key takeaway 3}**: {Brief explanation}
4. {Optional takeaway 4}
5. {Optional takeaway 5}

## What's Next

In the next chapter, [{Next Chapter Title}](./ch{NN+1}-{slug}.md), you will {preview of next topic and how it builds on this chapter}.

## Further Reading

- [{Resource 1 title}]({url}) - {Brief description}
- [{Resource 2 title}]({url}) - {Brief description}
- [Code Examples for This Chapter](https://github.com/{repo}/tree/main/code-examples/module-{N}-{slug})

---

<!-- Metadata for validation -->
<!--
constitution_check:
  - practical_first: true (humanoid example included)
  - progressive_complexity: true (beginner → advanced sections)
  - toolchain_integration: true (versions specified)
  - docusaurus_compatible: true (front matter, admonitions)
  - visual_clarity: true (diagrams, summary, tables)
  - humanoid_focus: true (specific robot example)
-->
