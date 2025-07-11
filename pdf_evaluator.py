import fitz  # PyMuPDF
from pdf2image import convert_from_path
import ollama
import os

# Step 1: Extract PDF Text
def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()

# Step 2: Extract PDF Images (Optional)
def extract_images(pdf_path, output_dir="pdf_images", poppler_path=r"C:/Users/sharath.chandran/Tools/poppler-24.08.0/Library/bin"):
    os.makedirs(output_dir, exist_ok=True)
    images = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
    image_paths = []
    for i, img in enumerate(images):
        path = os.path.join(output_dir, f"page_{i+1}.png")
        img.save(path)
        image_paths.append(path)
    return image_paths

# Step 3: Send to Ollama (LLaMA 3.2) for Analysis
def generate_explanation(text, model_name="llama3"):
    prompt = (
        "You are a document analyst. Read the following document content and explain its purpose, key points, and structure in plain English. Avoid assumptions.\n\n"
        f"{text}"
    )

    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response["message"]["content"]

# Step 4: Complete PDF Analysis
def analyze_pdf(pdf_path, model_name="llama3"):
    print(f"Analyzing PDF: {pdf_path}")
    text = extract_pdf_text(pdf_path)
    images = extract_images(pdf_path)

    print(f"Extracted {len(images)} image(s) from the PDF.")
    explanation = generate_explanation(text, model_name=model_name)

    print("\n--- Document Explanation ---\n")
    print(explanation)

    print("\n--- Image Files (for further review or vision analysis) ---")
    for img in images:
        print(f"  -> {img}")

# Example usage
if __name__ == "__main__":
    analyze_pdf("C:/Users/Public/Python/SuperAGI-main/samples_programs/vision-models/credit-memo-sample.pdf", model_name="llama3.2")
