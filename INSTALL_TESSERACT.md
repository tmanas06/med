# Installing Tesseract OCR on Windows

Since your PDF is image-based, you need Tesseract OCR to extract text from it.

## Option 1: Install Tesseract OCR (Recommended)

### Method A: Direct Download (No Admin Required for User Install)
1. Download Tesseract installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Download the **Windows installer** (e.g., `tesseract-ocr-w64-setup-5.x.x.exe`)
3. **Important**: During installation, choose "Install for current user only" (not "All users") - this doesn't require admin rights
4. Note the installation path (usually `C:\Users\YourUsername\AppData\Local\Programs\Tesseract-OCR` or `C:\Program Files\Tesseract-OCR`)
5. Add Tesseract to PATH:
   - Open PowerShell
   - Run: `$env:Path += ";C:\Users\YourUsername\AppData\Local\Programs\Tesseract-OCR"` (replace with your actual path)
   - Or add it permanently via System Environment Variables

### Method B: Using Chocolatey (Requires Admin)
If you have admin rights:
```powershell
# Run PowerShell as Administrator
choco install tesseract
```

## Option 2: Manual PATH Configuration

If Tesseract is installed but not found:
1. Find where Tesseract is installed (usually `C:\Program Files\Tesseract-OCR` or `C:\Users\YourUsername\AppData\Local\Programs\Tesseract-OCR`)
2. Add the `tesseract.exe` path to your Python code or environment

You can also set it in your `.env` file or configure it in the code.

## Verify Installation

After installing, restart your backend server and check the terminal output. You should see:
```
Tesseract OCR is available for image-based PDF processing.
```

If you see a warning, Tesseract is not in your PATH. You may need to restart your terminal/IDE after installation.

