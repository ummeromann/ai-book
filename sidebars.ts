import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  bookSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Module 0: Foundations of Physical AI',
      link: {
        type: 'doc',
        id: 'module-0-foundations/index',
      },
      items: [
        'module-0-foundations/ch01-intro-physical-ai',
        'module-0-foundations/ch02-digital-to-physical',
        'module-0-foundations/ch03-humanoid-landscape',
      ],
    },
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2)',
      link: {
        type: 'doc',
        id: 'module-1-ros2/index',
      },
      items: [
        'module-1-ros2/ch04-ros2-setup',
        'module-1-ros2/ch05-nodes-topics',
        'module-1-ros2/ch06-services-actions',
        'module-1-ros2/ch07-launch-tf2',
      ],
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin (Gazebo & Unity)',
      link: {
        type: 'doc',
        id: 'module-2-simulation/index',
      },
      items: [
        'module-2-simulation/ch08-gazebo-basics',
        'module-2-simulation/ch09-robot-simulation',
        'module-2-simulation/ch10-unity-visualization',
      ],
    },
    {
      type: 'category',
      label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac)',
      link: {
        type: 'doc',
        id: 'module-3-isaac/index',
      },
      items: [
        'module-3-isaac/ch11-isaac-setup',
        'module-3-isaac/ch12-perception',
        'module-3-isaac/ch13-navigation',
        'module-3-isaac/ch14-reinforcement-learning',
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      link: {
        type: 'doc',
        id: 'module-4-vla/index',
      },
      items: [
        'module-4-vla/ch15-vla-foundations',
        'module-4-vla/ch16-llm-integration',
        'module-4-vla/ch17-vla-pipelines',
        'module-4-vla/ch18-language-to-action',
      ],
    },
    {
      type: 'category',
      label: 'Module 5: Humanoid Intelligence & Interaction',
      link: {
        type: 'doc',
        id: 'module-5-humanoid/index',
      },
      items: [
        'module-5-humanoid/ch19-kinematics-dynamics',
        'module-5-humanoid/ch20-bipedal-locomotion',
        'module-5-humanoid/ch21-manipulation-grasping',
        'module-5-humanoid/ch22-hri-design',
      ],
    },
    {
      type: 'category',
      label: 'Module 6: Capstone – The Autonomous Humanoid',
      link: {
        type: 'doc',
        id: 'module-6-capstone/index',
      },
      items: [
        'module-6-capstone/ch23-capstone-architecture',
        'module-6-capstone/ch24-deployment-future',
      ],
    },
  ],
};

export default sidebars;
