from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader('IEIT.pdf')

docs = loader.load()

print(len(docs))
print(docs[0].page_content)
print(docs[1].metadata)
