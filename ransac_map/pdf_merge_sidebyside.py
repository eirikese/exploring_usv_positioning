from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO

def merge_pdfs_side_by_side(pdf1_path, pdf2_path, output_path):
    # Load the PDF files
    pdf1 = PdfReader(pdf1_path)
    pdf2 = PdfReader(pdf2_path)

    # Create a PDF writer for the output file
    pdf_writer = PdfWriter()

    # Determine the maximum number of pages
    max_pages = max(len(pdf1.pages), len(pdf2.pages))

    # Iterate through all pages
    for page_num in range(max_pages):
        # Create a new PDF page with double width
        packet = BytesIO()
        can = canvas.Canvas(packet, pagesize=letter)
        width, height = letter
        can.setPageSize((width * 2, height))
        can.showPage()
        can.save()

        # Move the buffer to the beginning
        packet.seek(0)
        new_page = PdfReader(packet).pages[0]

        # Merge pages from both PDFs if available
        if page_num < len(pdf1.pages):
            new_page.merge_page(pdf1.pages[page_num], (0, 0))

        if page_num < len(pdf2.pages):
            new_page.merge_page(pdf2.pages[page_num], (width, 0))

        pdf_writer.add_page(new_page)

    # Write the merged PDF to a file
    with open(output_path, 'wb') as output_file:
        pdf_writer.write(output_file)

# Example Usage
merge_pdfs_side_by_side(r'C:\Users\eirik\OneDrive - NTNU\General - o365_Prosjektoppgave (Sensor fusion med Maritime Robotics)\Project Report Figures\ransac_fit\ransac_angle_failed.pdf', r"C:\Users\eirik\OneDrive - NTNU\General - o365_Prosjektoppgave (Sensor fusion med Maritime Robotics)\Project Report Figures\ransac_fit\ransac_angle_success.pdf", 'output.pdf')
