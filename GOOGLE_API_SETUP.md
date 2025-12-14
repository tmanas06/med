# How to Use Google Gemini (Free API) Instead of OpenAI

Google Gemini offers a **free tier** with generous limits, making it a great alternative to OpenAI.

## Step 1: Get Your Free Google API Key

1. **Go to Google AI Studio:**
   - Visit: https://aistudio.google.com/app/apikey
   - Sign in with your Google account

2. **Create API Key:**
   - Click "Create API Key"
   - Select "Create API key in new project" (or choose existing project)
   - Copy your API key immediately (you won't see it again)

3. **Free Tier Limits:**
   - **60 requests per minute**
   - **1,500 requests per day**
   - **No credit card required**
   - **Completely free** for these limits

## Step 2: Configure Your .env File

1. **Open your `.env` file** in the project root (create it if it doesn't exist)

2. **Add these lines:**
   ```
   LLM_PROVIDER=google
   GOOGLE_API_KEY=your_google_api_key_here
   ```

3. **Example `.env` file:**
   ```
   # Use Google Gemini (free) instead of OpenAI
   LLM_PROVIDER=google
   GOOGLE_API_KEY=AIzaSy...your_key_here
   
   # Optional: Keep OpenAI key if you want to switch back
   # OPENAI_API_KEY=sk-...your_openai_key
   ```

## Step 3: Restart Your Backend Server

1. **Stop the current server** (Ctrl+C in terminal)
2. **Start it again:**
   ```bash
   python backend/main.py
   ```

3. **Look for this message:**
   ```
   ✓ Google Gemini API configured (FREE tier available)
   ```

## Step 4: Test It

1. Upload a PDF label
2. Click "Generate FAQs"
3. It should work with Google Gemini now!

## Switching Back to OpenAI

If you want to use OpenAI instead:

1. **Update `.env`:**
   ```
   LLM_PROVIDER=openai
   OPENAI_API_KEY=sk-your_openai_key
   ```

2. **Restart the server**

## Troubleshooting

**Error: "Google Gemini API not configured"**
- Make sure `GOOGLE_API_KEY` is set in `.env`
- Make sure `LLM_PROVIDER=google` is set
- Restart the server after changing `.env`

**Error: "Quota exceeded"**
- Free tier: 60 requests/minute, 1,500/day
- Wait a minute and try again
- Check your usage at https://aistudio.google.com/app/apikey

**Error: "Invalid API key"**
- Double-check your API key in `.env`
- Make sure there are no extra spaces
- Get a new key from https://aistudio.google.com/app/apikey

## Model Information

The app uses **`gemini-1.5-flash`** which is:
- ✅ Fast and efficient
- ✅ Available on free tier
- ✅ Great for FAQ generation
- ✅ Supports long context windows

If you need more advanced capabilities, you can change the model in `backend/main.py` to `gemini-1.5-pro` (may have different free tier limits).

## Benefits of Google Gemini Free Tier

✅ **No credit card required**  
✅ **60 requests per minute** (plenty for testing)  
✅ **1,500 requests per day** (generous free limit)  
✅ **No cost tracking needed** (it's free!)  
✅ **Same quality responses** as paid options

