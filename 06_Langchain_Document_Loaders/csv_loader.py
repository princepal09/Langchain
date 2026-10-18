from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='ecommerce_sales_data.csv')

docs = loader.lazy_load()

print(len(docs))
print(docs[0])