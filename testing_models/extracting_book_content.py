import pdfplumber
import os

def extract_text_from_pdf(pdf_path, output_txt_path):
    """
    Extracts all text from a PDF file and saves it to a plain text file.

    Args:
        pdf_path (str): The path to the input PDF file.
        output_txt_path (str): The path where the extracted text will be saved.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at '{pdf_path}'")
        return

    try:
        # Open the PDF file
        with pdfplumber.open(pdf_path) as pdf:
            all_text = []
            # Iterate through each page in the PDF
            for page in pdf.pages:
                # Extract text from the current page
                text = page.extract_text()
                if text: # Ensure text was actually extracted
                    all_text.append(text)
                    # Add a page separator for clarity in the output text file
                    all_text.append("\n--- PAGE END ---\n\n")

        # Join all extracted text and save it to the output text file
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write("".join(all_text))

        print(f"Text successfully extracted from '{pdf_path}' to '{output_txt_path}'")

    except Exception as e:
        print(f"An error occurred during text extraction: {e}")

# --- Example Usage ---
if __name__ == "__main__":
    # Define the path to your input PDF file
    # Make sure to replace 'your_document.pdf' with the actual name/path of your PDF
    # For example, if your PDF is in the same directory as this script, just use its name.
    # If it's in a subfolder, use 'subfolder/your_document.pdf'.
    input_pdf_file = "book3.pdf"

    # Define the path for the output text file
    output_text_file = "extracted_book3_text.txt"

    # Call the function to perform the extraction
    extract_text_from_pdf(input_pdf_file, output_text_file)

