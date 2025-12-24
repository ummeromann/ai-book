# Vercel Deployment Setup

## Required Configuration

Since the project has been restructured with a `frontend/` directory, you need to configure Vercel to use the correct root directory.

### Steps to Configure Vercel:

1. **Go to your Vercel Dashboard:**
   - Navigate to: https://vercel.com/dashboard
   - Select the `ai-book` project

2. **Update Project Settings:**
   - Go to **Settings** → **General**
   - Find **Root Directory** setting
   - Set it to: `frontend`
   - Click **Save**

3. **Redeploy:**
   - Go to **Deployments** tab
   - Click the three dots (•••) on the latest deployment
   - Select **Redeploy**

### Alternative: Configure via vercel.json in frontend/

The `frontend/vercel.json` file is already configured with:
- Framework: Docusaurus
- Build command: `npm run build`
- Output directory: `build`

Once you set the Root Directory to `frontend` in Vercel dashboard, this configuration will be used automatically.

### Expected Result:

After configuration:
- ✅ Vercel builds from `frontend/` directory
- ✅ Runs `npm install` in `frontend/`
- ✅ Runs `npm run build` in `frontend/`
- ✅ Deploys `frontend/build/` directory
- ✅ Site accessible at: https://ai-book-nine-mocha.vercel.app

### Troubleshooting:

If the site still shows errors after redeployment:
1. Check build logs for errors
2. Verify `frontend/package.json` exists
3. Ensure `frontend/docusaurus.config.ts` has correct baseUrl: `/`
4. Clear Vercel build cache and redeploy
