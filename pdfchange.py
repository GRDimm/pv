import fitz  # PyMuPDF

# Open your PDF
pdf = fitz.open("doc.pdf")

# Open a specific page
page = pdf.load_page(0)  # 0-based index

# Insert image
rect = fitz.Rect(0, 0, 100, 100)  # Coordinates for image placement
page.insert_image(rect, filename="img.jpg")

# Save the modified PDF
pdf.save("output.pdf")
pdf.close()