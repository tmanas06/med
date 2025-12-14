# How to Set Up Groq API (Fast Inference) as Fallback for Gemini Rate Limits

Groq API provides fast, low-cost inference and can be used as an automatic fallback when Google Gemini hits rate limits. This ensures your application continues working even when Gemini's free tier limits are reached.

## Step 1: Get Your Groq API Key

1. **Go to Groq Console:**
   - Visit: https://console.groq.com/
   - Sign in with your account (or create one)

2. **Create API Key:**
   - Navigate to API Keys section
   - Click "Create API Key"
   - Copy your API key immediately

3. **Pricing:**
   - Groq offers fast inference at low cost
   - Check current pricing at: https://console.groq.com/docs/pricing
   - Pay-as-you-go pricing model

## Step 2: Configure Your .env File

1. **Open your `.env` file** in the project root

2. **Add the Groq API key:**
   ```
   LLM_PROVIDER=google
   GOOGLE_API_KEY=your_google_api_key_here
   GROQ_API_KEY=your_groq_api_key_here
   ```

3. **Example `.env` file:**
   ```
   # Primary LLM provider
   LLM_PROVIDER=google
   GOOGLE_API_KEY=AIzaSy...your_google_key
   
   # Fallback when Gemini hits rate limits (fast inference)
   GROQ_API_KEY=gsk_...your_groq_key
   ```

## Step 3: How It Works

When you set `GROQ_API_KEY`:

1. **Normal Operation:** The system uses Google Gemini as the primary LLM
2. **Rate Limit Hit:** When Gemini returns a 429 (rate limit) error, the system automatically:
   - Detects the rate limit error
   - Falls back to Groq API (fast inference)
   - Returns the answer from Groq
   - Logs: "🔄 Gemini rate limit hit, falling back to Groq API..."
3. **Seamless Experience:** Users don't notice the switch - answers continue to work with fast inference

## Step 4: Restart Your Backend Server

1. **Stop the current server** (Ctrl+C in terminal)
2. **Start it again:**
   ```bash
   uvicorn backend.main:app --reload
   ```

3. **Look for this message:**
   ```
   ✓ Groq API (fast inference) configured as fallback for Gemini rate limits
   ```

## Step 5: Available Groq Models

The system uses `llama-3.1-70b-versatile` by default, which is:
- Fast inference (Groq's specialty)
- High quality responses
- Cost-effective

Other available models you can use:
- `mixtral-8x7b-32768` - Mixtral model
- `llama-3.1-8b-instant` - Faster, smaller model
- `llama-3.1-70b-versatile` - Default (balanced)

To change the model, edit `backend/main.py` and update the `model` parameter in `call_groq_llm_raw()` and `_call_groq_once()` functions.

## Testing the Fallback

1. Upload a PDF label
2. Generate FAQs (uses Gemini)
3. If Gemini hits rate limit, you'll see in the console:
   ```
   🔄 Gemini rate limit hit, falling back to Groq API...
   ✅ Groq fallback succeeded.
   ```
4. The FAQs will still be generated using Groq's fast inference

## Troubleshooting

**Error: "Groq API not configured"**
- Make sure `GROQ_API_KEY` is set in `.env`
- Restart the server after changing `.env`

**Error: "Invalid Groq API key"**
- Verify your API key is correct
- Check that your Groq account has API access enabled
- Ensure the API key starts with `gsk_`

**Fallback not working:**
- Ensure `LLM_PROVIDER=google` is set
- Check that Groq API key is valid
- Look for console messages indicating fallback attempts

## Benefits

- **No Interruptions:** Users never see rate limit errors when Groq is configured
- **Automatic:** No code changes needed - just set the API key
- **Fast Inference:** Groq specializes in high-speed inference
- **Cost-Effective:** Only uses Groq when Gemini is unavailable
- **Seamless:** Same answer format and quality

## Additional Resources

- **Groq Console:** https://console.groq.com/
- **Groq Documentation:** https://console.groq.com/docs
- **Groq Models:** https://console.groq.com/docs/models
- **Groq Pricing:** https://console.groq.com/docs/pricing

