# Tasks: Physical AI Book Structure

**Input**: Design documents from `/specs/002-book-structure/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/

**Tests**: Tests are NOT included (not requested in feature specification).

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Initialize Docusaurus project and configure shared infrastructure

- [ ] T001 Initialize Docusaurus 3.x project with TypeScript in repository root
- [ ] T002 Configure package.json with required dependencies (docusaurus, mermaid plugin, search)
- [ ] T003 [P] Create tsconfig.json for TypeScript configuration
- [ ] T004 [P] Create docusaurus.config.ts with site metadata and theme settings
- [ ] T005 Configure sidebars.ts with 7-module structure per plan.md
- [ ] T006 [P] Create static/img/ directory for images and diagrams
- [ ] T007 [P] Create static/img/module-0/ through static/img/module-6/ subdirectories
- [ ] T008 [P] Create src/css/custom.css for custom styling
- [ ] T009 [P] Create src/components/ directory for React components
- [ ] T010 Create docs/intro.md as book introduction page
- [ ] T011 [P] Create code-examples/ directory structure per plan.md
- [ ] T012 [P] Create docker/ directory with docker-compose.yml scaffold

**Checkpoint**: Docusaurus project runs with `npm run start`

---

## Phase 2: Foundational (Module Category Setup)

**Purpose**: Create all module directory structures and category metadata - MUST complete before chapter writing

**CRITICAL**: No chapter content can begin until this phase is complete

- [ ] T013 Create docs/module-0-foundations/ directory
- [ ] T014 [P] Create docs/module-0-foundations/_category_.json with label "Module 0: Foundations of Physical AI" position 1
- [ ] T015 Create docs/module-1-ros2/ directory
- [ ] T016 [P] Create docs/module-1-ros2/_category_.json with label "Module 1: The Robotic Nervous System (ROS 2)" position 2
- [ ] T017 Create docs/module-2-simulation/ directory
- [ ] T018 [P] Create docs/module-2-simulation/_category_.json with label "Module 2: The Digital Twin (Gazebo & Unity)" position 3
- [ ] T019 Create docs/module-3-isaac/ directory
- [ ] T020 [P] Create docs/module-3-isaac/_category_.json with label "Module 3: The AI-Robot Brain (NVIDIA Isaac)" position 4
- [ ] T021 Create docs/module-4-vla/ directory
- [ ] T022 [P] Create docs/module-4-vla/_category_.json with label "Module 4: Vision-Language-Action (VLA)" position 5
- [ ] T023 Create docs/module-5-humanoid/ directory
- [ ] T024 [P] Create docs/module-5-humanoid/_category_.json with label "Module 5: Humanoid Intelligence & Interaction" position 6
- [ ] T025 Create docs/module-6-capstone/ directory
- [ ] T026 [P] Create docs/module-6-capstone/_category_.json with label "Module 6: Capstone – The Autonomous Humanoid" position 7
- [ ] T027 Validate Docusaurus build passes with empty module structure

**Checkpoint**: Foundation ready - chapter writing can now begin

---

## Phase 3: User Story 1 - Foundation Reader Journey (Priority: P1)

**Goal**: Write Module 0 chapters (1-3) for beginner readers learning Physical AI foundations

**Independent Test**: Reader completes Module 0 and can explain Physical AI, identify humanoid platforms, describe sensors

### Module 0 Introduction

- [ ] T028 [US1] Write docs/module-0-foundations/index.md with module overview, learning path, and chapter summaries

### Chapter 1: Introduction to Physical AI & Embodied Intelligence

- [ ] T029 [US1] Write front matter for docs/module-0-foundations/ch01-intro-physical-ai.md (title, description, sidebar_position: 2, tags)
- [ ] T030 [P] [US1] Write learning objectives section (3-5 measurable outcomes) in ch01-intro-physical-ai.md
- [ ] T031 [P] [US1] Write prerequisites section (Python background required) in ch01-intro-physical-ai.md
- [ ] T032 [US1] Write introduction section defining Physical AI and embodied intelligence in ch01-intro-physical-ai.md
- [ ] T033 [US1] Write section on history and evolution of Physical AI in ch01-intro-physical-ai.md
- [ ] T034 [US1] Write section on real-world applications (manufacturing, healthcare, service) in ch01-intro-physical-ai.md
- [ ] T035 [US1] Add humanoid example: Tesla Optimus use case in ch01-intro-physical-ai.md
- [ ] T036 [P] [US1] Add diagram placeholder for Physical AI ecosystem overview in ch01-intro-physical-ai.md
- [ ] T037 [P] [US1] Add diagram placeholder for embodiment hypothesis illustration in ch01-intro-physical-ai.md
- [ ] T038 [US1] Write chapter summary (3-5 key takeaways) in ch01-intro-physical-ai.md
- [ ] T039 [P] [US1] Add Key Concepts glossary (Physical AI, Embodied Intelligence, Embodiment Hypothesis) in ch01-intro-physical-ai.md

### Chapter 2: From Digital AI to Robots that Understand Physical Laws

- [ ] T040 [US1] Write front matter for docs/module-0-foundations/ch02-digital-to-physical.md
- [ ] T041 [P] [US1] Write learning objectives section in ch02-digital-to-physical.md
- [ ] T042 [P] [US1] Write prerequisites section (Chapter 1 required) in ch02-digital-to-physical.md
- [ ] T043 [US1] Write introduction comparing digital AI vs physical AI in ch02-digital-to-physical.md
- [ ] T044 [US1] Write section on physics-informed machine learning in ch02-digital-to-physical.md
- [ ] T045 [US1] Write section on world models and common sense reasoning in ch02-digital-to-physical.md
- [ ] T046 [US1] Write section on simulation-to-reality challenges in ch02-digital-to-physical.md
- [ ] T047 [US1] Add humanoid example: Boston Dynamics Atlas physics understanding in ch02-digital-to-physical.md
- [ ] T048 [P] [US1] Add diagram placeholder for digital vs physical AI comparison in ch02-digital-to-physical.md
- [ ] T049 [P] [US1] Add diagram placeholder for world model architecture in ch02-digital-to-physical.md
- [ ] T050 [US1] Write chapter summary in ch02-digital-to-physical.md
- [ ] T051 [P] [US1] Add Key Concepts glossary (World Model, Sim-to-Real, Physics-Informed ML) in ch02-digital-to-physical.md

### Chapter 3: Humanoid Robotics Landscape & Sensor Systems

- [ ] T052 [US1] Write front matter for docs/module-0-foundations/ch03-humanoid-landscape.md
- [ ] T053 [P] [US1] Write learning objectives section in ch03-humanoid-landscape.md
- [ ] T054 [P] [US1] Write prerequisites section in ch03-humanoid-landscape.md
- [ ] T055 [US1] Write introduction to humanoid robotics landscape in ch03-humanoid-landscape.md
- [ ] T056 [US1] Write section on major humanoid platforms (Atlas, Optimus, Digit, H1, NAO, Pepper) with comparison table in ch03-humanoid-landscape.md
- [ ] T057 [US1] Write section on vision sensors (cameras, depth, stereo) in ch03-humanoid-landscape.md
- [ ] T058 [US1] Write section on proprioceptive sensors (IMU, encoders, force/torque) in ch03-humanoid-landscape.md
- [ ] T059 [US1] Write section on exteroceptive sensors (LiDAR, ultrasonic, tactile) in ch03-humanoid-landscape.md
- [ ] T060 [US1] Add humanoid example: Unitree H1 sensor configuration in ch03-humanoid-landscape.md
- [ ] T061 [P] [US1] Add diagram placeholder for humanoid platform comparison chart in ch03-humanoid-landscape.md
- [ ] T062 [P] [US1] Add diagram placeholder for sensor placement on humanoid body in ch03-humanoid-landscape.md
- [ ] T063 [US1] Write chapter summary in ch03-humanoid-landscape.md
- [ ] T064 [P] [US1] Add Key Concepts glossary (IMU, Proprioception, LiDAR, Force/Torque) in ch03-humanoid-landscape.md

**Checkpoint**: Module 0 complete - readers can understand Physical AI foundations

---

## Phase 4: User Story 2 - ROS 2 Practitioner Journey (Priority: P1)

**Goal**: Write Module 1 chapters (4-7) with Python rclpy examples for ROS 2 development

**Independent Test**: Reader creates ROS 2 package with pub/sub, writes URDF for humanoid arm

### Module 1 Introduction

- [ ] T065 [US2] Write docs/module-1-ros2/index.md with module overview and ROS 2 learning path

### Chapter 4: ROS 2 Architecture & Core Concepts

- [ ] T066 [US2] Write front matter for docs/module-1-ros2/ch04-ros2-architecture.md
- [ ] T067 [P] [US2] Write learning objectives section in ch04-ros2-architecture.md
- [ ] T068 [P] [US2] Write prerequisites section (Module 0, Ubuntu/WSL2) in ch04-ros2-architecture.md
- [ ] T069 [US2] Write introduction to ROS 2 and DDS middleware in ch04-ros2-architecture.md
- [ ] T070 [US2] Write section on ROS 2 installation (Humble) with code snippets in ch04-ros2-architecture.md
- [ ] T071 [US2] Write section on packages, workspaces, and colcon build in ch04-ros2-architecture.md
- [ ] T072 [US2] Write section on QoS (Quality of Service) profiles in ch04-ros2-architecture.md
- [ ] T073 [US2] Add Python code snippet: creating a ROS 2 workspace in ch04-ros2-architecture.md
- [ ] T074 [US2] Add humanoid example: ROS 2 architecture for Digit robot in ch04-ros2-architecture.md
- [ ] T075 [P] [US2] Add diagram placeholder for ROS 2 architecture layers in ch04-ros2-architecture.md
- [ ] T076 [P] [US2] Add diagram placeholder for DDS communication flow in ch04-ros2-architecture.md
- [ ] T077 [US2] Write chapter summary in ch04-ros2-architecture.md
- [ ] T078 [P] [US2] Add Key Concepts glossary (DDS, QoS, colcon, ament) in ch04-ros2-architecture.md

### Chapter 5: Nodes, Topics, Services, and Actions

- [ ] T079 [US2] Write front matter for docs/module-1-ros2/ch05-nodes-topics-services.md
- [ ] T080 [P] [US2] Write learning objectives section in ch05-nodes-topics-services.md
- [ ] T081 [P] [US2] Write prerequisites section (Chapter 4) in ch05-nodes-topics-services.md
- [ ] T082 [US2] Write introduction to ROS 2 communication patterns in ch05-nodes-topics-services.md
- [ ] T083 [US2] Write section on nodes and executors in ch05-nodes-topics-services.md
- [ ] T084 [US2] Write section on topics (pub/sub) with Python code example in ch05-nodes-topics-services.md
- [ ] T085 [US2] Write section on services (request/response) with Python code example in ch05-nodes-topics-services.md
- [ ] T086 [US2] Write section on actions (long-running tasks) with Python code example in ch05-nodes-topics-services.md
- [ ] T087 [US2] Add humanoid example: joint state publisher/subscriber for humanoid arm in ch05-nodes-topics-services.md
- [ ] T088 [P] [US2] Add diagram placeholder for node communication patterns in ch05-nodes-topics-services.md
- [ ] T089 [P] [US2] Add diagram placeholder for action server/client flow in ch05-nodes-topics-services.md
- [ ] T090 [US2] Write chapter summary in ch05-nodes-topics-services.md

### Chapter 6: Bridging Python Agents to ROS Controllers (rclpy)

- [ ] T091 [US2] Write front matter for docs/module-1-ros2/ch06-rclpy-python.md
- [ ] T092 [P] [US2] Write learning objectives section in ch06-rclpy-python.md
- [ ] T093 [P] [US2] Write prerequisites section in ch06-rclpy-python.md
- [ ] T094 [US2] Write introduction to rclpy client library in ch06-rclpy-python.md
- [ ] T095 [US2] Write section on async programming with rclpy in ch06-rclpy-python.md
- [ ] T096 [US2] Write section on parameter handling in Python nodes in ch06-rclpy-python.md
- [ ] T097 [US2] Write section on lifecycle nodes in ch06-rclpy-python.md
- [ ] T098 [US2] Add Python code snippet: complete rclpy node with callbacks in ch06-rclpy-python.md
- [ ] T099 [US2] Add humanoid example: Python agent controlling humanoid head movement in ch06-rclpy-python.md
- [ ] T100 [P] [US2] Add diagram placeholder for rclpy node lifecycle in ch06-rclpy-python.md
- [ ] T101 [US2] Write chapter summary in ch06-rclpy-python.md

### Chapter 7: URDF – Unified Robot Description Format for Humanoids

- [ ] T102 [US2] Write front matter for docs/module-1-ros2/ch07-urdf-humanoids.md
- [ ] T103 [P] [US2] Write learning objectives section in ch07-urdf-humanoids.md
- [ ] T104 [P] [US2] Write prerequisites section in ch07-urdf-humanoids.md
- [ ] T105 [US2] Write introduction to URDF and robot description in ch07-urdf-humanoids.md
- [ ] T106 [US2] Write section on links, joints, and visual/collision geometry in ch07-urdf-humanoids.md
- [ ] T107 [US2] Write section on xacro macros for modular URDF in ch07-urdf-humanoids.md
- [ ] T108 [US2] Write section on URDF visualization tools (rviz2, joint_state_publisher) in ch07-urdf-humanoids.md
- [ ] T109 [US2] Add XML code snippet: complete humanoid arm URDF in ch07-urdf-humanoids.md
- [ ] T110 [US2] Add humanoid example: URDF for simplified humanoid torso in ch07-urdf-humanoids.md
- [ ] T111 [P] [US2] Add diagram placeholder for URDF link/joint hierarchy in ch07-urdf-humanoids.md
- [ ] T112 [P] [US2] Add diagram placeholder for humanoid kinematic chain in ch07-urdf-humanoids.md
- [ ] T113 [US2] Write chapter summary in ch07-urdf-humanoids.md
- [ ] T114 [P] [US2] Add Key Concepts glossary (URDF, xacro, joint types, TF) in ch07-urdf-humanoids.md

### Code Examples for Module 1

- [ ] T115 [P] [US2] Create code-examples/module-1-ros2/README.md with setup instructions
- [ ] T116 [P] [US2] Create code-examples/module-1-ros2/minimal_publisher/ Python package
- [ ] T117 [P] [US2] Create code-examples/module-1-ros2/minimal_subscriber/ Python package
- [ ] T118 [P] [US2] Create code-examples/module-1-ros2/humanoid_arm_urdf/ with URDF files
- [ ] T119 [US2] Create docker/Dockerfile.ros2 with ROS 2 Humble environment

**Checkpoint**: Module 1 complete - readers can build ROS 2 packages and write URDFs

---

## Phase 5: User Story 3 - Simulation Engineer Journey (Priority: P2)

**Goal**: Write Module 2 chapters (8-10) for Gazebo simulation and Unity visualization

**Independent Test**: Reader sets up Gazebo sim with humanoid, creates Unity visualization scene

### Module 2 Introduction

- [ ] T120 [US3] Write docs/module-2-simulation/index.md with module overview

### Chapter 8: Gazebo Simulation Environment Setup

- [ ] T121 [US3] Write front matter for docs/module-2-simulation/ch08-gazebo-setup.md
- [ ] T122 [P] [US3] Write learning objectives section in ch08-gazebo-setup.md
- [ ] T123 [P] [US3] Write prerequisites section (Module 1) in ch08-gazebo-setup.md
- [ ] T124 [US3] Write introduction to Gazebo (Fortress/Harmonic) in ch08-gazebo-setup.md
- [ ] T125 [US3] Write section on Gazebo installation and ros_gz bridge in ch08-gazebo-setup.md
- [ ] T126 [US3] Write section on world files and SDF format in ch08-gazebo-setup.md
- [ ] T127 [US3] Write section on model spawning and robot integration in ch08-gazebo-setup.md
- [ ] T128 [US3] Add bash/XML code snippets for Gazebo world creation in ch08-gazebo-setup.md
- [ ] T129 [US3] Add humanoid example: spawning humanoid in Gazebo world in ch08-gazebo-setup.md
- [ ] T130 [P] [US3] Add diagram placeholder for Gazebo architecture in ch08-gazebo-setup.md
- [ ] T131 [US3] Write chapter summary in ch08-gazebo-setup.md

### Chapter 9: Physics Simulation, Sensors, and Collisions

- [ ] T132 [US3] Write front matter for docs/module-2-simulation/ch09-physics-sensors.md
- [ ] T133 [P] [US3] Write learning objectives section in ch09-physics-sensors.md
- [ ] T134 [P] [US3] Write prerequisites section in ch09-physics-sensors.md
- [ ] T135 [US3] Write introduction to physics engines (ODE, Bullet, DART) in ch09-physics-sensors.md
- [ ] T136 [US3] Write section on sensor plugins (camera, LiDAR, IMU, contact) in ch09-physics-sensors.md
- [ ] T137 [US3] Write section on collision detection and contact forces in ch09-physics-sensors.md
- [ ] T138 [US3] Write section on ROS 2 sensor message bridging in ch09-physics-sensors.md
- [ ] T139 [US3] Add XML code snippets for sensor plugin configuration in ch09-physics-sensors.md
- [ ] T140 [US3] Add humanoid example: simulating humanoid foot contact detection in ch09-physics-sensors.md
- [ ] T141 [P] [US3] Add diagram placeholder for sensor data flow in ch09-physics-sensors.md
- [ ] T142 [US3] Write chapter summary in ch09-physics-sensors.md

### Chapter 10: Unity for High-Fidelity Visualization & HRI

- [ ] T143 [US3] Write front matter for docs/module-2-simulation/ch10-unity-visualization.md
- [ ] T144 [P] [US3] Write learning objectives section in ch10-unity-visualization.md
- [ ] T145 [P] [US3] Write prerequisites section in ch10-unity-visualization.md
- [ ] T146 [US3] Write introduction clarifying Unity is VISUALIZATION ONLY (not control) in ch10-unity-visualization.md
- [ ] T147 [US3] Write section on Unity Robotics Hub and ROS-TCP-Connector setup in ch10-unity-visualization.md
- [ ] T148 [US3] Write section on importing robot models and visualization in ch10-unity-visualization.md
- [ ] T149 [US3] Write section on HRI scene design for demonstrations in ch10-unity-visualization.md
- [ ] T150 [US3] Add C#/Unity code snippets for ROS message subscription in ch10-unity-visualization.md
- [ ] T151 [US3] Add humanoid example: Unity HRI scene with NAO/Pepper visualization in ch10-unity-visualization.md
- [ ] T152 [P] [US3] Add diagram placeholder for Unity-ROS 2 bridge architecture in ch10-unity-visualization.md
- [ ] T153 [US3] Write chapter summary emphasizing visualization-only role in ch10-unity-visualization.md

### Code Examples for Module 2

- [ ] T154 [P] [US3] Create code-examples/module-2-simulation/README.md
- [ ] T155 [P] [US3] Create code-examples/module-2-simulation/gazebo_worlds/ with SDF files
- [ ] T156 [P] [US3] Create code-examples/module-2-simulation/unity_hri_scene/ Unity project scaffold

**Checkpoint**: Module 2 complete - readers can run Gazebo sims and Unity visualizations

---

## Phase 6: User Story 4 - AI/ML Engineer Journey (Priority: P2)

**Goal**: Write Module 3 chapters (11-14) for NVIDIA Isaac perception and RL

**Independent Test**: Reader generates synthetic data, runs VSLAM, trains RL policy

### Module 3 Introduction

- [ ] T157 [US4] Write docs/module-3-isaac/index.md with module overview

### Chapter 11: NVIDIA Isaac Platform Overview

- [ ] T158 [US4] Write front matter for docs/module-3-isaac/ch11-isaac-overview.md
- [ ] T159 [P] [US4] Write learning objectives section in ch11-isaac-overview.md
- [ ] T160 [P] [US4] Write prerequisites section (Module 2, NVIDIA GPU) in ch11-isaac-overview.md
- [ ] T161 [US4] Write introduction to NVIDIA Isaac ecosystem in ch11-isaac-overview.md
- [ ] T162 [US4] Write section on Isaac SDK, Sim, ROS, Gym components in ch11-isaac-overview.md
- [ ] T163 [US4] Write section on GPU requirements and cloud alternatives in ch11-isaac-overview.md
- [ ] T164 [US4] Write section on Isaac installation and setup in ch11-isaac-overview.md
- [ ] T165 [US4] Add humanoid example: Isaac Sim humanoid assets in ch11-isaac-overview.md
- [ ] T166 [P] [US4] Add diagram placeholder for Isaac platform architecture in ch11-isaac-overview.md
- [ ] T167 [US4] Write chapter summary in ch11-isaac-overview.md

### Chapter 12: Isaac Sim & Synthetic Data Generation

- [ ] T168 [US4] Write front matter for docs/module-3-isaac/ch12-isaac-sim-synthetic.md
- [ ] T169 [P] [US4] Write learning objectives section in ch12-isaac-sim-synthetic.md
- [ ] T170 [P] [US4] Write prerequisites section in ch12-isaac-sim-synthetic.md
- [ ] T171 [US4] Write introduction to Isaac Sim and Omniverse in ch12-isaac-sim-synthetic.md
- [ ] T172 [US4] Write section on domain randomization for sim-to-real in ch12-isaac-sim-synthetic.md
- [ ] T173 [US4] Write section on synthetic data generation and labeling in ch12-isaac-sim-synthetic.md
- [ ] T174 [US4] Write section on dataset export formats (COCO, KITTI) in ch12-isaac-sim-synthetic.md
- [ ] T175 [US4] Add Python code snippets for Replicator data generation in ch12-isaac-sim-synthetic.md
- [ ] T176 [US4] Add humanoid example: generating training data for humanoid detection in ch12-isaac-sim-synthetic.md
- [ ] T177 [P] [US4] Add diagram placeholder for synthetic data pipeline in ch12-isaac-sim-synthetic.md
- [ ] T178 [US4] Write chapter summary in ch12-isaac-sim-synthetic.md

### Chapter 13: Isaac ROS, VSLAM & Nav2 Navigation

- [ ] T179 [US4] Write front matter for docs/module-3-isaac/ch13-isaac-ros-nav2.md
- [ ] T180 [P] [US4] Write learning objectives section in ch13-isaac-ros-nav2.md
- [ ] T181 [P] [US4] Write prerequisites section in ch13-isaac-ros-nav2.md
- [ ] T182 [US4] Write introduction to Isaac ROS packages in ch13-isaac-ros-nav2.md
- [ ] T183 [US4] Write section on Visual SLAM (cuVSLAM) in ch13-isaac-ros-nav2.md
- [ ] T184 [US4] Write section on Nav2 integration with costmaps in ch13-isaac-ros-nav2.md
- [ ] T185 [US4] Write section on path planning and localization in ch13-isaac-ros-nav2.md
- [ ] T186 [US4] Add Python/launch code snippets for VSLAM setup in ch13-isaac-ros-nav2.md
- [ ] T187 [US4] Add humanoid example: humanoid navigation in cluttered environment in ch13-isaac-ros-nav2.md
- [ ] T188 [P] [US4] Add diagram placeholder for VSLAM pipeline in ch13-isaac-ros-nav2.md
- [ ] T189 [US4] Write chapter summary in ch13-isaac-ros-nav2.md

### Chapter 14: Reinforcement Learning & Sim-to-Real Transfer

- [ ] T190 [US4] Write front matter for docs/module-3-isaac/ch14-rl-sim-to-real.md
- [ ] T191 [P] [US4] Write learning objectives section in ch14-rl-sim-to-real.md
- [ ] T192 [P] [US4] Write prerequisites section in ch14-rl-sim-to-real.md
- [ ] T193 [US4] Write introduction to RL for robotics in ch14-rl-sim-to-real.md
- [ ] T194 [US4] Write section on Isaac Gym and parallel simulation in ch14-rl-sim-to-real.md
- [ ] T195 [US4] Write section on PPO and policy training in ch14-rl-sim-to-real.md
- [ ] T196 [US4] Write section on domain adaptation and reality gap in ch14-rl-sim-to-real.md
- [ ] T197 [US4] Add Python code snippets for Isaac Gym training loop in ch14-rl-sim-to-real.md
- [ ] T198 [US4] Add humanoid example: training humanoid standing balance policy in ch14-rl-sim-to-real.md
- [ ] T199 [P] [US4] Add diagram placeholder for RL training loop in ch14-rl-sim-to-real.md
- [ ] T200 [P] [US4] Add diagram placeholder for sim-to-real transfer process in ch14-rl-sim-to-real.md
- [ ] T201 [US4] Write chapter summary in ch14-rl-sim-to-real.md

### Code Examples for Module 3

- [ ] T202 [P] [US4] Create code-examples/module-3-isaac/README.md
- [ ] T203 [P] [US4] Create code-examples/module-3-isaac/synthetic_data/ scripts
- [ ] T204 [P] [US4] Create code-examples/module-3-isaac/vslam_nav2/ launch files
- [ ] T205 [US4] Create docker/Dockerfile.isaac with Isaac Sim environment

**Checkpoint**: Module 3 complete - readers can use Isaac for perception and RL

---

## Phase 7: User Story 5 - VLA Researcher Journey (Priority: P2)

**Goal**: Write Module 4 chapters (15-18) for Vision-Language-Action pipelines (LLM for planning only)

**Independent Test**: Reader builds VLA pipeline where LLM plans and ROS 2 executes

### Module 4 Introduction

- [ ] T206 [US5] Write docs/module-4-vla/index.md with module overview emphasizing LLM=planning only

### Chapter 15: Vision Systems for Humanoid Robots

- [ ] T207 [US5] Write front matter for docs/module-4-vla/ch15-vision-systems.md
- [ ] T208 [P] [US5] Write learning objectives section in ch15-vision-systems.md
- [ ] T209 [P] [US5] Write prerequisites section (Module 3) in ch15-vision-systems.md
- [ ] T210 [US5] Write introduction to humanoid vision requirements in ch15-vision-systems.md
- [ ] T211 [US5] Write section on depth cameras and stereo vision in ch15-vision-systems.md
- [ ] T212 [US5] Write section on object detection and recognition in ch15-vision-systems.md
- [ ] T213 [US5] Write section on pose estimation for humans and objects in ch15-vision-systems.md
- [ ] T214 [US5] Add Python code snippets for vision pipeline in ch15-vision-systems.md
- [ ] T215 [US5] Add humanoid example: vision system for humanoid object manipulation in ch15-vision-systems.md
- [ ] T216 [P] [US5] Add diagram placeholder for humanoid vision pipeline in ch15-vision-systems.md
- [ ] T217 [US5] Write chapter summary in ch15-vision-systems.md

### Chapter 16: Voice-to-Action using OpenAI Whisper

- [ ] T218 [US5] Write front matter for docs/module-4-vla/ch16-whisper-voice.md
- [ ] T219 [P] [US5] Write learning objectives section in ch16-whisper-voice.md
- [ ] T220 [P] [US5] Write prerequisites section in ch16-whisper-voice.md
- [ ] T221 [US5] Write introduction to voice interfaces for robots in ch16-whisper-voice.md
- [ ] T222 [US5] Write section on Whisper ASR integration in ch16-whisper-voice.md
- [ ] T223 [US5] Write section on command parsing and intent extraction in ch16-whisper-voice.md
- [ ] T224 [US5] Write section on mapping voice to HIGH-LEVEL action plans in ch16-whisper-voice.md
- [ ] T225 [US5] Add Python code snippets for Whisper-to-ROS bridge in ch16-whisper-voice.md
- [ ] T226 [US5] Add humanoid example: voice-controlled humanoid for household tasks in ch16-whisper-voice.md
- [ ] T227 [P] [US5] Add diagram placeholder for voice-to-action pipeline in ch16-whisper-voice.md
- [ ] T228 [US5] Write chapter summary in ch16-whisper-voice.md

### Chapter 17: LLM-Based Cognitive Planning

- [ ] T229 [US5] Write front matter for docs/module-4-vla/ch17-llm-planning.md
- [ ] T230 [P] [US5] Write learning objectives section in ch17-llm-planning.md
- [ ] T231 [P] [US5] Write prerequisites section in ch17-llm-planning.md
- [ ] T232 [US5] Write introduction clarifying LLMs for PLANNING ONLY (not motor control) in ch17-llm-planning.md
- [ ] T233 [US5] Write section on task decomposition with LLMs in ch17-llm-planning.md
- [ ] T234 [US5] Write section on chain-of-thought reasoning for robotics in ch17-llm-planning.md
- [ ] T235 [US5] Write section on grounding LLM outputs in robot capabilities in ch17-llm-planning.md
- [ ] T236 [US5] Write section on safety constraints and validation in ch17-llm-planning.md
- [ ] T237 [US5] Add Python code snippets for LLM-based task planner in ch17-llm-planning.md
- [ ] T238 [US5] Add humanoid example: LLM planning for "make coffee" task decomposition in ch17-llm-planning.md
- [ ] T239 [P] [US5] Add diagram placeholder for LLM planning architecture in ch17-llm-planning.md
- [ ] T240 [US5] Write chapter summary emphasizing planning-only role in ch17-llm-planning.md

### Chapter 18: Vision-Language-Action Pipelines in ROS 2

- [ ] T241 [US5] Write front matter for docs/module-4-vla/ch18-vla-pipelines.md
- [ ] T242 [P] [US5] Write learning objectives section in ch18-vla-pipelines.md
- [ ] T243 [P] [US5] Write prerequisites section in ch18-vla-pipelines.md
- [ ] T244 [US5] Write introduction to end-to-end VLA (LLM plans → controllers execute) in ch18-vla-pipelines.md
- [ ] T245 [US5] Write section on RT-2 and VLA model concepts in ch18-vla-pipelines.md
- [ ] T246 [US5] Write section on multimodal integration in ROS 2 in ch18-vla-pipelines.md
- [ ] T247 [US5] Write section on action execution via traditional controllers in ch18-vla-pipelines.md
- [ ] T248 [US5] Add Python code snippets for complete VLA pipeline in ch18-vla-pipelines.md
- [ ] T249 [US5] Add humanoid example: VLA pipeline for humanoid pick-and-place in ch18-vla-pipelines.md
- [ ] T250 [P] [US5] Add diagram placeholder for end-to-end VLA architecture in ch18-vla-pipelines.md
- [ ] T251 [US5] Write chapter summary in ch18-vla-pipelines.md

### Code Examples for Module 4

- [ ] T252 [P] [US5] Create code-examples/module-4-vla/README.md
- [ ] T253 [P] [US5] Create code-examples/module-4-vla/vision_pipeline/ scripts
- [ ] T254 [P] [US5] Create code-examples/module-4-vla/whisper_ros/ bridge package
- [ ] T255 [P] [US5] Create code-examples/module-4-vla/llm_planner/ planning service

**Checkpoint**: Module 4 complete - readers can build VLA pipelines

---

## Phase 8: User Story 6 - Humanoid Specialist Journey (Priority: P3)

**Goal**: Write Module 5 chapters (19-22) for humanoid kinematics, locomotion, and HRI

**Independent Test**: Reader implements IK, understands balance control, designs HRI flow

### Module 5 Introduction

- [ ] T256 [US6] Write docs/module-5-humanoid/index.md with module overview

### Chapter 19: Humanoid Kinematics & Dynamics

- [ ] T257 [US6] Write front matter for docs/module-5-humanoid/ch19-kinematics-dynamics.md
- [ ] T258 [P] [US6] Write learning objectives section in ch19-kinematics-dynamics.md
- [ ] T259 [P] [US6] Write prerequisites section (Module 1-4 recommended) in ch19-kinematics-dynamics.md
- [ ] T260 [US6] Write introduction to humanoid kinematics in ch19-kinematics-dynamics.md
- [ ] T261 [US6] Write section on forward kinematics and DH parameters in ch19-kinematics-dynamics.md
- [ ] T262 [US6] Write section on inverse kinematics for humanoid arms in ch19-kinematics-dynamics.md
- [ ] T263 [US6] Write section on Jacobians and velocity kinematics in ch19-kinematics-dynamics.md
- [ ] T264 [US6] Write section on dynamics equations (Newton-Euler, Lagrangian) in ch19-kinematics-dynamics.md
- [ ] T265 [US6] Add Python code snippets for IK solver in ch19-kinematics-dynamics.md
- [ ] T266 [US6] Add humanoid example: IK for Atlas arm reaching in ch19-kinematics-dynamics.md
- [ ] T267 [P] [US6] Add diagram placeholder for humanoid kinematic chain in ch19-kinematics-dynamics.md
- [ ] T268 [US6] Write chapter summary in ch19-kinematics-dynamics.md

### Chapter 20: Bipedal Locomotion & Balance Control

- [ ] T269 [US6] Write front matter for docs/module-5-humanoid/ch20-bipedal-locomotion.md
- [ ] T270 [P] [US6] Write learning objectives section in ch20-bipedal-locomotion.md
- [ ] T271 [P] [US6] Write prerequisites section in ch20-bipedal-locomotion.md
- [ ] T272 [US6] Write introduction to bipedal walking challenges in ch20-bipedal-locomotion.md
- [ ] T273 [US6] Write section on Zero Moment Point (ZMP) theory in ch20-bipedal-locomotion.md
- [ ] T274 [US6] Write section on inverted pendulum models in ch20-bipedal-locomotion.md
- [ ] T275 [US6] Write section on gait generation and walking patterns in ch20-bipedal-locomotion.md
- [ ] T276 [US6] Write section on push recovery and balance control in ch20-bipedal-locomotion.md
- [ ] T277 [US6] Add humanoid example: Digit bipedal walking analysis in ch20-bipedal-locomotion.md
- [ ] T278 [P] [US6] Add diagram placeholder for ZMP and support polygon in ch20-bipedal-locomotion.md
- [ ] T279 [P] [US6] Add diagram placeholder for gait phases in ch20-bipedal-locomotion.md
- [ ] T280 [US6] Write chapter summary in ch20-bipedal-locomotion.md

### Chapter 21: Manipulation & Grasping with Humanoid Hands

- [ ] T281 [US6] Write front matter for docs/module-5-humanoid/ch21-manipulation-grasping.md
- [ ] T282 [P] [US6] Write learning objectives section in ch21-manipulation-grasping.md
- [ ] T283 [P] [US6] Write prerequisites section in ch21-manipulation-grasping.md
- [ ] T284 [US6] Write introduction to humanoid manipulation in ch21-manipulation-grasping.md
- [ ] T285 [US6] Write section on grasp planning and contact models in ch21-manipulation-grasping.md
- [ ] T286 [US6] Write section on dexterous manipulation with multi-finger hands in ch21-manipulation-grasping.md
- [ ] T287 [US6] Write section on force control and compliance in ch21-manipulation-grasping.md
- [ ] T288 [US6] Add Python code snippets for grasp pose generation in ch21-manipulation-grasping.md
- [ ] T289 [US6] Add humanoid example: Optimus hand grasping objects in ch21-manipulation-grasping.md
- [ ] T290 [P] [US6] Add diagram placeholder for grasp taxonomy in ch21-manipulation-grasping.md
- [ ] T291 [US6] Write chapter summary in ch21-manipulation-grasping.md

### Chapter 22: Natural Human-Robot Interaction Design

- [ ] T292 [US6] Write front matter for docs/module-5-humanoid/ch22-hri-design.md
- [ ] T293 [P] [US6] Write learning objectives section in ch22-hri-design.md
- [ ] T294 [P] [US6] Write prerequisites section in ch22-hri-design.md
- [ ] T295 [US6] Write introduction to HRI for humanoid robots in ch22-hri-design.md
- [ ] T296 [US6] Write section on social robotics principles in ch22-hri-design.md
- [ ] T297 [US6] Write section on gesture recognition and generation in ch22-hri-design.md
- [ ] T298 [US6] Write section on safety in human-robot collaboration in ch22-hri-design.md
- [ ] T299 [US6] Write section on ethical considerations in HRI in ch22-hri-design.md
- [ ] T300 [US6] Add humanoid example: Pepper social interaction scenario in ch22-hri-design.md
- [ ] T301 [P] [US6] Add diagram placeholder for HRI interaction flow in ch22-hri-design.md
- [ ] T302 [US6] Write chapter summary in ch22-hri-design.md

### Code Examples for Module 5

- [ ] T303 [P] [US6] Create code-examples/module-5-humanoid/README.md
- [ ] T304 [P] [US6] Create code-examples/module-5-humanoid/ik_solver/ Python package
- [ ] T305 [P] [US6] Create code-examples/module-5-humanoid/gait_generator/ scripts

**Checkpoint**: Module 5 complete - readers understand humanoid-specific challenges

---

## Phase 9: User Story 7 - Capstone Integrator Journey (Priority: P3)

**Goal**: Write Module 6 chapters (23-24) and deploy book to GitHub Pages + Vercel

**Independent Test**: Reader designs complete system in simulation, evaluates performance

### Module 6 Introduction

- [ ] T306 [US7] Write docs/module-6-capstone/index.md with module overview (simulation-first capstone)

### Chapter 23: Capstone Architecture & System Design

- [ ] T307 [US7] Write front matter for docs/module-6-capstone/ch23-capstone-architecture.md
- [ ] T308 [P] [US7] Write learning objectives section in ch23-capstone-architecture.md
- [ ] T309 [P] [US7] Write prerequisites section (all prior modules) in ch23-capstone-architecture.md
- [ ] T310 [US7] Write introduction to capstone (RUNS FULLY IN SIMULATION) in ch23-capstone-architecture.md
- [ ] T311 [US7] Write section on system integration patterns in ch23-capstone-architecture.md
- [ ] T312 [US7] Write section on component selection from prior modules in ch23-capstone-architecture.md
- [ ] T313 [US7] Write section on architecture design for autonomous humanoid in ch23-capstone-architecture.md
- [ ] T314 [US7] Write section on simulation environment configuration in ch23-capstone-architecture.md
- [ ] T315 [US7] Add humanoid example: complete system architecture diagram in ch23-capstone-architecture.md
- [ ] T316 [P] [US7] Add diagram placeholder for capstone system architecture in ch23-capstone-architecture.md
- [ ] T317 [P] [US7] Add diagram placeholder for data flow between modules in ch23-capstone-architecture.md
- [ ] T318 [US7] Write chapter summary in ch23-capstone-architecture.md

### Chapter 24: End-to-End Deployment, Evaluation & Future of Physical AI

- [ ] T319 [US7] Write front matter for docs/module-6-capstone/ch24-deployment-future.md
- [ ] T320 [P] [US7] Write learning objectives section in ch24-deployment-future.md
- [ ] T321 [P] [US7] Write prerequisites section in ch24-deployment-future.md
- [ ] T322 [US7] Write introduction to deployment and evaluation in ch24-deployment-future.md
- [ ] T323 [US7] Write section on testing strategies in simulation in ch24-deployment-future.md
- [ ] T324 [US7] Write section on evaluation metrics for humanoid systems in ch24-deployment-future.md
- [ ] T325 [US7] Write section on OPTIONAL Jetson edge deployment path in ch24-deployment-future.md
- [ ] T326 [US7] Write section on future of Physical AI and emerging trends in ch24-deployment-future.md
- [ ] T327 [US7] Add humanoid example: capstone evaluation results in ch24-deployment-future.md
- [ ] T328 [P] [US7] Add diagram placeholder for deployment options in ch24-deployment-future.md
- [ ] T329 [US7] Write chapter summary and book conclusion in ch24-deployment-future.md

### Code Examples for Module 6

- [ ] T330 [P] [US7] Create code-examples/module-6-capstone/README.md
- [ ] T331 [P] [US7] Create code-examples/module-6-capstone/full_system/ integrated launch files

**Checkpoint**: All chapters complete

---

## Phase 10: Polish & Deployment

**Purpose**: Final review, CI/CD setup, and deployment to GitHub Pages and Vercel

### Deployment Configuration

- [ ] T332 Create .github/workflows/deploy.yml for GitHub Pages deployment
- [ ] T333 [P] Create vercel.json for Vercel deployment configuration
- [ ] T334 [P] Create .github/workflows/preview.yml for PR preview deployments

### Final Validation

- [ ] T335 Run npm run build and fix any errors across all chapters
- [ ] T336 [P] Validate all internal links between chapters work correctly
- [ ] T337 [P] Verify all diagram placeholders are present and described
- [ ] T338 [P] Verify all code blocks have language specified
- [ ] T339 Validate all chapters have front matter (title, description, sidebar_position, tags)
- [ ] T340 [P] Verify all chapters have chapter summaries
- [ ] T341 [P] Verify all chapters have humanoid examples
- [ ] T342 Review sidebars.ts matches actual chapter structure

### Deployment

- [ ] T343 Deploy to GitHub Pages via Actions workflow
- [ ] T344 [P] Deploy to Vercel and verify preview
- [ ] T345 Validate live site navigation and content

**Checkpoint**: Book deployed and accessible

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies - start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 completion - BLOCKS all content writing
- **Phases 3-9 (User Stories)**: All depend on Phase 2 completion
  - US1 (Module 0): Independent - can start first
  - US2 (Module 1): Can parallel with US1 after Phase 2
  - US3 (Module 2): Best after US2 (builds on ROS 2)
  - US4 (Module 3): After US3 (builds on simulation)
  - US5 (Module 4): After US4 (builds on Isaac)
  - US6 (Module 5): After US3-5 (needs all prior knowledge)
  - US7 (Module 6): After US1-6 (capstone integrates all)
- **Phase 10 (Deployment)**: Depends on all user stories complete

### Within Each User Story

1. Module index before chapters
2. Chapter front matter before content
3. Learning objectives and prerequisites before main content
4. Main content before diagrams/summary
5. Code examples can parallel chapter writing

### Parallel Opportunities

Within each module, tasks marked [P] can run in parallel:
- Front matter and learning objectives
- Different diagram placeholders
- Different code example directories
- Glossary and summary sections

---

## Implementation Strategy

### MVP First (Module 0 + Module 1)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1 (Module 0)
4. Complete Phase 4: User Story 2 (Module 1)
5. **STOP and VALIDATE**: Deploy MVP with Modules 0-1
6. Incremental: Add modules 2-6

### Incremental Delivery

| Increment | Modules | Chapters | User Stories |
|-----------|---------|----------|--------------|
| MVP | 0-1 | 1-7 | US1, US2 |
| +Simulation | 2 | 8-10 | US3 |
| +Isaac | 3 | 11-14 | US4 |
| +VLA | 4 | 15-18 | US5 |
| +Humanoid | 5 | 19-22 | US6 |
| Complete | 6 | 23-24 | US7 |

---

## Summary

| Phase | Task Count | Story |
|-------|------------|-------|
| Phase 1: Setup | 12 | - |
| Phase 2: Foundational | 15 | - |
| Phase 3: US1 (Module 0) | 37 | Foundation Reader |
| Phase 4: US2 (Module 1) | 55 | ROS 2 Practitioner |
| Phase 5: US3 (Module 2) | 37 | Simulation Engineer |
| Phase 6: US4 (Module 3) | 49 | AI/ML Engineer |
| Phase 7: US5 (Module 4) | 50 | VLA Researcher |
| Phase 8: US6 (Module 5) | 50 | Humanoid Specialist |
| Phase 9: US7 (Module 6) | 26 | Capstone Integrator |
| Phase 10: Deployment | 14 | - |
| **Total** | **345** | |

**Parallel Opportunities**: ~40% of tasks within each phase can run in parallel

**Suggested MVP Scope**: Phases 1-4 (Setup + Module 0 + Module 1 = 119 tasks)
