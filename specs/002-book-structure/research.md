# Research: Physical AI Book Structure

**Feature**: 002-book-structure
**Date**: 2025-12-22
**Status**: Complete

## Research Areas

### 1. Docusaurus 3.x Best Practices

**Decision**: Use Docusaurus 3.x with TypeScript configuration

**Rationale**:
- Docusaurus 3.x is the latest stable version with React 18 support
- TypeScript config provides better IDE support and type checking
- Native MDX 2 support for interactive content
- Built-in versioning for future book editions

**Alternatives Considered**:
| Alternative | Why Rejected |
|-------------|--------------|
| Docusaurus 2.x | End of life, missing React 18 features |
| MkDocs | Less customization, no React components |
| GitBook | Commercial, less control over styling |
| VuePress | Constitution specifies Docusaurus |

**Implementation Notes**:
- Use `@docusaurus/preset-classic` for standard features
- Enable `docs-only` mode (no blog needed)
- Configure Algolia DocSearch for search functionality
- Use Mermaid plugin for diagrams

### 2. ROS 2 Version Selection

**Decision**: ROS 2 Humble as baseline, Iron compatibility notes

**Rationale**:
- Humble is LTS (support until 2027)
- Most Isaac ROS packages target Humble
- Iron is latest stable but not LTS
- Jazzy released 2024 but ecosystem still maturing

**Alternatives Considered**:
| Alternative | Why Rejected |
|-------------|--------------|
| Iron only | Not LTS, shorter support window |
| Foxy | EOL May 2023 |
| Galactic | EOL December 2022 |

**Implementation Notes**:
- All code examples test on Humble
- Note Iron-specific syntax differences where applicable
- Docker images based on `ros:humble-desktop`

### 3. Gazebo Version Selection

**Decision**: Gazebo Fortress (Ignition) as primary, Harmonic notes

**Rationale**:
- Fortress is LTS (support until 2026)
- Native ROS 2 Humble integration via `ros_gz`
- Harmonic is newer but Fortress more stable for education

**Alternatives Considered**:
| Alternative | Why Rejected |
|-------------|--------------|
| Gazebo Classic | Deprecated, no longer maintained |
| Harmonic only | Newer, less documentation available |
| Webots | Different paradigm, not industry standard |

**Implementation Notes**:
- Use `ros_gz_bridge` for ROS 2 integration
- Provide humanoid URDF that works in both versions
- Document migration path to Harmonic

### 4. NVIDIA Isaac Version Selection

**Decision**: Isaac Sim 2023.1.1+ and Isaac ROS 2.0+

**Rationale**:
- 2023.1.1 has stable ROS 2 Humble support
- Isaac ROS 2.0 provides Nav2 integration
- Omniverse cloud options for readers without local GPU

**Alternatives Considered**:
| Alternative | Why Rejected |
|-------------|--------------|
| Isaac SDK (original) | Deprecated in favor of Isaac ROS |
| Isaac Sim 2022.x | Missing key humanoid features |

**Implementation Notes**:
- Document GPU requirements (RTX 3070+ recommended)
- Provide cloud GPU alternatives (Omniverse Cloud, AWS)
- Isaac Gym for RL training examples

### 5. Unity Configuration for Visualization

**Decision**: Unity 2022.3 LTS with ROS-TCP-Connector

**Rationale**:
- 2022.3 is current LTS (support until 2025)
- ROS-TCP-Connector maintained by Unity Robotics
- Clear separation: visualization only, NOT control

**Alternatives Considered**:
| Alternative | Why Rejected |
|-------------|--------------|
| Unity 6 | Too new, less stable |
| Unreal Engine | Less ROS integration support |
| Custom WebGL | More development effort |

**Implementation Notes**:
- Use Unity Robotics Hub packages
- ROS-TCP-Endpoint node for ROS 2 bridge
- Emphasize visualization role in chapter 10

### 6. LLM Integration Approach

**Decision**: LLMs for cognitive planning only, via API (OpenAI, local models)

**Rationale**:
- LLMs excel at task decomposition and reasoning
- Direct motor control is unsafe and unreliable
- API approach allows model flexibility
- Local models (Llama 3) for offline/privacy scenarios

**Alternatives Considered**:
| Alternative | Why Rejected |
|-------------|--------------|
| LLM direct control | Unsafe, unpredictable latency |
| No LLM integration | Misses VLA opportunity |
| Embedded-only models | Too limited for planning |

**Implementation Notes**:
- Chapter 17 covers planning architecture
- Clear boundary: LLM outputs high-level plans
- Traditional controllers execute motor commands
- Example: "Pick up red cube" → LLM → plan → motion primitives

### 7. Deployment Strategy

**Decision**: GitHub Pages (primary) + Vercel (preview/backup)

**Rationale**:
- GitHub Pages free for public repos
- Vercel provides preview deployments for PRs
- Both support static Docusaurus sites
- CDN delivery for global performance

**Alternatives Considered**:
| Alternative | Why Rejected |
|-------------|--------------|
| Netlify | Similar to Vercel, one is sufficient |
| Self-hosted | More maintenance overhead |
| AWS S3/CloudFront | Overkill for static docs |

**Implementation Notes**:
- GitHub Actions for automated deployment
- Vercel for branch previews
- Custom domain setup optional

### 8. Code Example Organization

**Decision**: Separate `code-examples/` directory with module subdirectories

**Rationale**:
- Clean separation of prose and code
- Each module has self-contained examples
- Docker compose for multi-container setups
- Easy to clone/run independently

**Alternatives Considered**:
| Alternative | Why Rejected |
|-------------|--------------|
| Inline code only | Not runnable, can't test |
| Separate repo | Harder to keep in sync |
| Embedded in docs/ | Clutters documentation |

**Implementation Notes**:
- Each module has `README.md` with setup instructions
- Docker images tagged by module
- CI validates all examples build/run

## Technology Stack Summary

| Component | Version | Purpose |
|-----------|---------|---------|
| Docusaurus | 3.x | Documentation platform |
| Node.js | 18+ | Build environment |
| ROS 2 | Humble (Iron compat) | Robot middleware |
| Gazebo | Fortress (Harmonic compat) | Physics simulation |
| Isaac Sim | 2023.1.1+ | Advanced simulation |
| Isaac ROS | 2.0+ | Perception/navigation |
| Unity | 2022.3 LTS | Visualization only |
| Python | 3.10+ | Code examples |
| Docker | Latest | Environment isolation |

## Open Questions Resolved

| Question | Resolution |
|----------|------------|
| Which ROS 2 version? | Humble (LTS) with Iron notes |
| Which Gazebo? | Fortress (LTS) with Harmonic notes |
| Isaac version? | 2023.1.1+ and Isaac ROS 2.0+ |
| Unity role? | Visualization only, not control |
| LLM scope? | Planning only, not motor control |
| Deployment? | GitHub Pages + Vercel |

## Research Complete

All technical decisions resolved. Ready for Phase 1 design artifacts.
