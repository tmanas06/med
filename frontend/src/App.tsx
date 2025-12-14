import { useState } from 'react'
import './App.css'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

interface FAQ {
  question: string
  answer: string
}

function App() {
  const [labelId, setLabelId] = useState<string | null>(null)
  const [uploadStatus, setUploadStatus] = useState<string>('')
  const [uploadError, setUploadError] = useState<string>('')
  const [isUploading, setIsUploading] = useState(false)
  
  const [faqs, setFaqs] = useState<FAQ[]>([])
  const [isGeneratingFAQs, setIsGeneratingFAQs] = useState(false)
  const [faqError, setFaqError] = useState<string>('')
  const [faqButtonDisabledUntil, setFaqButtonDisabledUntil] = useState<number | null>(null)
  
  const [question, setQuestion] = useState<string>('')
  const [answer, setAnswer] = useState<string>('')
  const [isAsking, setIsAsking] = useState(false)
  const [askError, setAskError] = useState<string>('')
  const [askDebounceTimer, setAskDebounceTimer] = useState<NodeJS.Timeout | null>(null)

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    if (!file.name.endsWith('.pdf')) {
      setUploadError('Please upload a PDF file')
      return
    }

    setIsUploading(true)
    setUploadError('')
    setUploadStatus('Uploading...')

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Upload failed')
      }

      const data = await response.json()
      setLabelId(data.label_id)
      setUploadStatus(`Upload successful! Processed ${data.num_pages} pages, created ${data.num_chunks} chunks.`)
      setUploadError('')
      // Reset FAQs and answer when new label is uploaded
      setFaqs([])
      setAnswer('')
    } catch (error: any) {
      setUploadError(error.message || 'Failed to upload PDF')
      setUploadStatus('')
      setLabelId(null)
    } finally {
      setIsUploading(false)
    }
  }

  const handleGenerateFAQs = async () => {
    if (!labelId) {
      setFaqError('Please upload a label first')
      return
    }

    setIsGeneratingFAQs(true)
    setFaqError('')

    try {
      const response = await fetch(`${API_BASE_URL}/generate_faqs`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ label_id: labelId }),
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Failed to generate FAQs')
      }

      const data = await response.json()
      setFaqs(data.faqs)
      // Disable button for 30 seconds after successful generation
      setFaqButtonDisabledUntil(Date.now() + 30000)
    } catch (error: any) {
      setFaqError(error.message || 'Failed to generate FAQs')
      // If rate limit error, disable for 60 seconds
      if (error.message?.includes('rate limit') || error.message?.includes('429')) {
        setFaqButtonDisabledUntil(Date.now() + 60000)
      }
    } finally {
      setIsGeneratingFAQs(false)
    }
  }

  const handleAsk = async () => {
    if (!labelId) {
      setAskError('Please upload a label first')
      return
    }

    if (!question.trim()) {
      setAskError('Please enter a question')
      return
    }

    // Debounce: clear any pending request
    if (askDebounceTimer) {
      clearTimeout(askDebounceTimer)
    }

    // If already asking, ignore
    if (isAsking) {
      return
    }

    setIsAsking(true)
    setAskError('')

    try {
      const response = await fetch(`${API_BASE_URL}/ask`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          label_id: labelId,
          question: question.trim(),
        }),
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Failed to get answer')
      }

      const data = await response.json()
      setAnswer(data.answer)
    } catch (error: any) {
      setAskError(error.message || 'Failed to get answer')
      setAnswer('')
      // If rate limit, disable input temporarily
      if (error.message?.includes('rate limit') || error.message?.includes('429')) {
        // Input will be disabled via isAsking state for a moment
      }
    } finally {
      setIsAsking(false)
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Pharma LBL FAQ Generator & Q&A Tool</h1>
      </header>

      <main className="app-main">
        {/* Section A: Upload Label */}
        <section className="section">
          <h2>Section A: Upload Label</h2>
          <div className="upload-section">
            <input
              type="file"
              accept=".pdf"
              onChange={handleFileUpload}
              disabled={isUploading}
              className="file-input"
            />
            {isUploading && <p className="status">Uploading...</p>}
            {uploadStatus && <p className="status success">{uploadStatus}</p>}
            {uploadError && <p className="error">{uploadError}</p>}
            {labelId && (
              <p className="label-id">Label ID: <code>{labelId}</code></p>
            )}
          </div>
        </section>

        {/* Section B: Auto-generated FAQs */}
        <section className="section">
          <h2>Section B: Auto-generated FAQs</h2>
          <div className="faq-section">
            <button
              onClick={handleGenerateFAQs}
              disabled={!labelId || isGeneratingFAQs || (faqButtonDisabledUntil !== null && Date.now() < faqButtonDisabledUntil)}
              className="button primary"
            >
              {isGeneratingFAQs ? 'Generating FAQs...' : 
               (faqButtonDisabledUntil !== null && Date.now() < faqButtonDisabledUntil) 
                 ? `Please wait ${Math.ceil((faqButtonDisabledUntil - Date.now()) / 1000)}s...` 
                 : 'Generate FAQs'}
            </button>
            {faqError && <p className="error">{faqError}</p>}
            {faqs.length > 0 && (
              <div className="faq-list">
                {faqs.map((faq, index) => (
                  <div key={index} className="faq-item">
                    <h3 className="faq-question">{faq.question}</h3>
                    <div className="faq-answer">{faq.answer}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </section>

        {/* Section C: Ask a Question */}
        <section className="section">
          <h2>Section C: Ask a Question</h2>
          <div className="ask-section">
            <div className="input-group">
              <input
                type="text"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Enter your question about the label..."
                className="question-input"
                disabled={!labelId || isAsking}
                onKeyPress={(e) => {
                  if (e.key === 'Enter' && !isAsking) {
                    handleAsk()
                  }
                }}
              />
              <button
                onClick={handleAsk}
                disabled={!labelId || !question.trim() || isAsking}
                className="button primary"
              >
                {isAsking ? 'Asking...' : 'Ask'}
              </button>
            </div>
            {askError && <p className="error">{askError}</p>}
            {answer && (
              <div className="answer-box">
                <h3>Answer:</h3>
                <div className="answer-text">{answer}</div>
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  )
}

export default App
