# AI Book - Physical AI & Humanoid Robotics

Comprehensive educational resource covering Physical AI, Humanoid Robotics, and Embodied Intelligence.

## ✨ New Features

### 🔐 User Authentication & Personalized Learning

The book now includes **user authentication and personalized learning** powered by your profile!

#### Features:
- 👤 **User Accounts**: Sign up and sign in with email/password
- 📊 **Learning Profile**: Tell us about your background (software skills, AI experience, hardware)
- 🎯 **Personalized Responses**: The chatbot adapts answers to your skill level and goals
- 💾 **Saved Progress**: Your profile and chat history are stored securely

#### How It Works:
1. **Sign up** using the "Sign Up" button (top-right corner)
2. **Fill out your profile** with:
   - Software level (Beginner/Intermediate/Advanced)
   - Programming languages you know (Python, JavaScript, ROS, etc.)
   - AI/ML experience level
   - Hardware available (GPU, Jetson, etc.)
   - Learning goals
3. **Get personalized answers** from the chatbot based on your profile
4. **Update your profile anytime** by clicking your email in the top-right

### 💬 RAG-Powered Chatbot

The book includes an intelligent **RAG (Retrieval-Augmented Generation) chatbot** that can answer questions about the book content!

#### Features:
- 💬 **Full-Book Q&A**: Ask questions and get answers from the entire book
- 📝 **Selected-Text Q&A**: Select specific text and ask questions about just that selection
- 🔍 **Source Citations**: Every answer includes references to the relevant book sections
- 💾 **Conversation History**: Your chat sessions are saved for later reference
- 🎯 **Personalized Explanations**: Answers adapt to your skill level (when signed in)

#### How It Works:
1. **Click the floating chat button** (bottom-right corner)
2. **Ask general questions** about any topic in the book, OR
3. **Select text** in the book and click "Ask Selected" to focus on that specific content
4. The AI assistant answers strictly from the book content - no hallucinations!
5. **Signed-in users** get responses tailored to their background and learning goals

## 🏗️ Project Structure

```
ai-book/
├── frontend/          # Docusaurus documentation site
│   ├── docs/         # Documentation content
│   ├── src/          # Custom components (includes ChatBot)
│   ├── static/       # Static assets
│   └── README.md     # Frontend-specific documentation
├── backend/          # FastAPI RAG service
│   ├── app/          # Application code
│   │   ├── main.py           # FastAPI endpoints
│   │   ├── rag_service.py    # RAG logic
│   │   ├── database.py       # Database models
│   │   └── config.py         # Configuration
│   ├── requirements.txt      # Python dependencies
│   └── README.md     # Backend documentation
├── .github/          # GitHub Actions workflows
├── .claude/          # Claude Code configurations
├── .specify/         # Project templates and scripts
├── specs/            # Feature specifications
├── history/          # Prompt history records
├── code-examples/    # Code examples
└── docker/           # Docker configurations
```

## 🚀 Quick Start

### Frontend (Documentation Site)

```bash
cd frontend
npm install
npm start
```

Visit http://localhost:3000 to view the site.

### Backend (RAG Chatbot API)

1. **Install dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   # Create .env file in backend/ directory with:

   # OpenAI Configuration
   OPENAI_API_KEY=your-openai-api-key

   # Qdrant Configuration
   QDRANT_URL=your-qdrant-url
   QDRANT_API_KEY=your-qdrant-api-key
   QDRANT_COLLECTION_NAME=physical_ai_book

   # Neon Postgres Configuration
   DATABASE_URL=postgresql+psycopg://user:pass@host/db

   # Authentication Settings (IMPORTANT: Change JWT_SECRET_KEY in production!)
   JWT_SECRET_KEY=your-super-secret-jwt-key-min-32-characters
   JWT_ALGORITHM=HS256
   ACCESS_TOKEN_EXPIRE_DAYS=7

   # CORS Settings
   ALLOWED_ORIGINS=http://localhost:3000,https://your-production-url.com
   ```

3. **Start the backend:**
   ```bash
   python run.py
   ```

4. **Ingest book content:**
   ```bash
   curl -X POST http://localhost:8000/ingest \
     -H "Content-Type: application/json" \
     -d '{"content_path": "../frontend/docs", "force_reingest": false}'
   ```

The API will be available at http://localhost:8000 (docs at `/docs`)

### Architecture

```
┌─────────────┐     ┌──────────────────────┐     ┌─────────────┐
│  Frontend   │────▶│  FastAPI Backend     │────▶│   OpenAI    │
│ (Docusaurus)│     │  - RAG Service       │     │  (GPT-4)    │
│             │     │  - Authentication    │     └─────────────┘
│             │     │  - Personalization   │
└─────────────┘     └──────────────────────┘
                           │
                           ├──────▶ Qdrant Cloud (Vectors)
                           │
                           └──────▶ Neon Postgres (Users, Profiles, Metadata)
```

**Technologies:**
- **Frontend**: React, TypeScript, Docusaurus
- **Backend**: FastAPI, Python
- **Vector Search**: Qdrant Cloud (free tier)
- **Database**: Neon Serverless Postgres
- **Authentication**: JWT tokens with bcrypt password hashing
- **LLM**: OpenAI GPT-4 (answer generation with personalization)
- **Embeddings**: OpenAI text-embedding-3-small

## 📚 Content Modules

- **Module 0**: Foundations of Physical AI
- **Module 1**: ROS 2 Fundamentals
- **Module 2**: Robot Simulation (Gazebo, Unity)
- **Module 3**: NVIDIA Isaac Platform
- **Module 4**: Vision-Language-Action Models
- **Module 5**: Humanoid Robotics
- **Module 6**: Capstone Projects

## 🌐 Deployment

- **Production**: https://ai-book-nine-mocha.vercel.app
- **GitHub**: https://github.com/ummeromann/ai-book

### Automatic Deployment

- **Vercel**: Automatically deploys on push to main
- **GitHub Pages**: Configured via GitHub Actions (optional)

## 🛠️ Development

### Frontend Development

```bash
cd frontend
npm start          # Start dev server
npm run build      # Build for production
npm run serve      # Preview production build
```

### Project Management

This project uses Spec-Driven Development (SDD) workflow:
- Specifications in `/specs`
- Planning artifacts tracked
- Prompt history in `/history`

## 📖 Documentation

- Frontend documentation: [frontend/README.md](./frontend/README.md)
- Backend documentation: [backend/README.md](./backend/README.md)
- Project constitution: [.specify/memory/constitution.md](./.specify/memory/constitution.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

Educational resource for learning Physical AI and Humanoid Robotics.

---

Built with ❤️ using Docusaurus, React, and TypeScript
