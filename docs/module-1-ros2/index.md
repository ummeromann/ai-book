---
id: index
title: "Module 1: ROS 2 Middleware"
sidebar_position: 1
description: "Master ROS 2, the industry-standard middleware for robotics development, from installation to advanced communication patterns."
tags: [ros2, middleware, robotics, communication]
---

# Module 1: ROS 2 Middleware

**Focus**: Robot Operating System 2 fundamentals and communication patterns

<!-- DIAGRAM: id="module-1-overview" type="architecture" format="mermaid"
     description="Overview of Module 1 showing chapter relationships and learning flow" -->

```mermaid
graph LR
    A[Chapter 4: ROS 2 Setup] --> B[Chapter 5: Nodes & Topics]
    B --> C[Chapter 6: Services & Actions]
    C --> D[Chapter 7: Launch & TF2]

    style A fill:#e1f5fe
    style D fill:#c8e6c9
```

## Module Overview

ROS 2 (Robot Operating System 2) is the de facto standard middleware for robotics development. It provides the communication infrastructure, tools, and libraries that allow you to build complex robotic systems from modular components.

In this module, you'll go from installation to building complete ROS 2 applications capable of controlling simulated humanoid robots.

**Key Technologies Covered**: ROS 2 Humble/Iron, DDS, Python & C++ APIs, TF2

## Why ROS 2?

| Feature | Benefit |
|---------|---------|
| **Modularity** | Build complex systems from simple, reusable components |
| **Communication** | Standardized message passing between processes |
| **Tools** | Visualization, debugging, and analysis utilities |
| **Ecosystem** | Thousands of packages for perception, navigation, manipulation |
| **Real-time** | Designed for time-critical robotic applications |
| **Cross-platform** | Works on Linux, Windows, and macOS |

## Learning Path

This module builds your ROS 2 foundation:

1. **Installation and setup** of ROS 2 development environment
2. **Core concepts**: nodes, topics, publishers, subscribers
3. **Advanced patterns**: services, actions, parameters
4. **System architecture**: launch files, coordinate transforms

## Chapters in This Module

### [Chapter 4: ROS 2 Installation, Workspaces & First Nodes](./ch04-ros2-setup)

Set up your ROS 2 development environment and create your first nodes. Learn the workspace structure and build system.

**Learning Objectives**:
- Install ROS 2 Humble on Ubuntu 22.04 or via Docker
- Create and build a ROS 2 workspace
- Write your first publisher and subscriber nodes
- Understand the ROS 2 build system (colcon)

---

### [Chapter 5: Nodes, Topics, Publishers & Subscribers](./ch05-nodes-topics)

Deep dive into ROS 2's publish-subscribe communication model. Learn to build modular, communicating robotic systems.

**Learning Objectives**:
- Understand the ROS 2 computation graph
- Create complex multi-node systems
- Work with standard and custom message types
- Use Quality of Service (QoS) settings effectively

---

### [Chapter 6: Services, Actions & Parameters](./ch06-services-actions)

Master request-response patterns and long-running tasks with services and actions. Configure nodes dynamically with parameters.

**Learning Objectives**:
- Implement synchronous service calls
- Build action servers and clients for long-running tasks
- Use parameters for runtime configuration
- Choose the right communication pattern for each use case

---

### [Chapter 7: Launch Files & TF2 Coordinate Frames](./ch07-launch-tf2)

Learn to orchestrate complex robotic systems with launch files and manage coordinate transformations with TF2.

**Learning Objectives**:
- Write Python launch files to start multiple nodes
- Understand coordinate frames in robotics
- Use TF2 for coordinate transformations
- Visualize robot state in RViz2

## Prerequisites

Before starting this module, you should have:

- Completed Module 0: Foundations of Physical AI
- Ubuntu 22.04 LTS (native or WSL2) or Docker
- Python programming experience
- Basic command-line familiarity

## Development Environment

This module uses:

| Tool | Version | Purpose |
|------|---------|---------|
| **ROS 2** | Humble Hawksbill | Primary middleware |
| **Python** | 3.10+ | Node development |
| **colcon** | Latest | Build system |
| **Docker** | 24.0+ | Optional containerization |

:::tip Docker Option
If you can't install ROS 2 natively, use the provided Docker containers:

```bash
cd docker
docker-compose up ros2-humble
```
:::

## What You'll Build

By the end of this module, you will have:

1. A working ROS 2 development environment
2. Custom nodes that publish and subscribe to topics
3. Service and action servers for robotic tasks
4. Launch files that orchestrate multi-node systems
5. Understanding of coordinate transforms for robotics

## Module Timeline

| Chapter | Estimated Time | Difficulty |
|---------|---------------|------------|
| Chapter 4: ROS 2 Setup | 3 hours | Beginner |
| Chapter 5: Nodes & Topics | 3 hours | Beginner |
| Chapter 6: Services & Actions | 3 hours | Intermediate |
| Chapter 7: Launch & TF2 | 3 hours | Intermediate |

**Total Module Time**: ~12 hours

## Code Repository

All code examples for this module are in:

```bash
code-examples/module-1-ros2/
├── ch04_first_nodes/
├── ch05_pubsub/
├── ch06_services_actions/
└── ch07_launch_tf2/
```

:::tip Getting Started
Start with Chapter 4 to set up your environment. Each chapter builds on the previous one.
:::
