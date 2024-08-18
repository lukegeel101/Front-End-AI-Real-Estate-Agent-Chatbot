import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)
    
    # Initialize an empty string to store the text
    text = ""
    
    # Iterate over each page
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    
    # Close the PDF file
    pdf_document.close()
    
    return text

# Example usage:
pdf_path = 'your_pdf_file.pdf'  # Replace with the path to your PDF file
text = extract_text_from_pdf(pdf_path)
print(text)
