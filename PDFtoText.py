import pdfplumber
with pdfplumber.open("/home/amirhossein/Documents/GitHub/Semantic-Annotation/files"
                     "/Papers_Semantic_Gateway_as_a_Service_architecture_for_IoT_1___1_.pdf") as pdf:
    for page in pdf.pages:
        print(page.extract_text(x_tolerance=0.15, y_tolerance = 1).lower())


import PyPDF2

pdfFileObject = open("/home/amirhossein/Documents/GitHub/Semantic-Annotation/files"
                     "/Papers_Semantic_Gateway_as_a_Service_architecture_for_IoT_1___1_.pdf", 'rb')

pdfReader = PyPDF2.PdfFileReader(pdfFileObject)

print(" No. Of Pages :", pdfReader.numPages)

pageObject = pdfReader.getPage(0)

print(pageObject.extractText())

pdfFileObject.close()