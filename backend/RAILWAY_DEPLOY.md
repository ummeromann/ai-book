# Deploy Backend to Railway

## Quick Deploy Steps

### 1. Push your code to GitHub (if not already done)
```bash
git add .
git commit -m "Add Railway configuration"
git push
```

### 2. Deploy on Railway Dashboard

1. Go to [railway.app](https://railway.app) and login
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your `ai-book` repository
5. Select the `backend` folder as the root directory
6. Railway will auto-detect it as a Python project

### 3. Add Environment Variables

In Railway project settings, add these environment variables:

```env
# Copy these from your backend/.env file
OPENAI_API_KEY=your_openai_api_key_here

OPENROUTER_API_KEY=your_openrouter_api_key_here
USE_FALLBACK=true
OPENROUTER_CHAT_MODEL=openai/gpt-3.5-turbo
OPENROUTER_EMBEDDING_MODEL=text-embedding-3-small

QDRANT_URL=your_qdrant_url_here
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_COLLECTION_NAME=physical_ai_book

DATABASE_URL=sqlite:///./chat_history.db

EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o-mini
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_RETRIEVAL_RESULTS=5

ALLOWED_ORIGINS=http://localhost:3000,https://ai-book-nine-mocha.vercel.app
```

**Note:** Replace the placeholder values with your actual API keys from `backend/.env`

### 4. Get your Railway URL

After deployment, Railway will provide a URL like:
`https://your-app-name.up.railway.app`

Copy this URL!

### 5. Update Vercel Environment Variables

1. Go to your Vercel project dashboard
2. Go to Settings → Environment Variables
3. Add a new variable:
   - Name: `REACT_APP_API_URL`
   - Value: `https://your-railway-url.up.railway.app` (your Railway URL from step 4)
4. Redeploy your Vercel site

## Alternative: Deploy via CLI

If you prefer using the CLI and can open a browser:

```bash
cd backend
railway login
railway init
railway up
railway variables set OPENAI_API_KEY=your_key
# ... add all other env vars
railway domain
```

## Troubleshooting

- If deployment fails, check Railway logs in the dashboard
- Make sure all environment variables are set
- The `sentence-transformers` package may take time to download on first deployment
- SQLite might not persist on Railway - consider upgrading to Neon Postgres if you need chat history
