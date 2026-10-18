from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """Artificial intelligence has rapidly transformed the way we interact with technology. From voice assistants that understand natural language to recommendation systems that predict our preferences, AI is becoming an invisible part of everyday life. Machine learning models are trained on vast amounts of data, allowing them to recognize patterns and make decisions with remarkable accuracy.

In the field of natural language processing, large language models have shown impressive capabilities in generating coherent text, answering questions, and even writing code. However, these models still face challenges such as hallucinations, bias, and limited context windows. Researchers continue to explore new architectures and training techniques to overcome these limitations.
Retrieval-Augmented Generation, or RAG, is one promising approach. By combining the generative power of language models with external knowledge sources, RAG systems can produce more accurate and up-to-date responses. The process typically involves splitting documents into smaller chunks, embedding them into a vector space, and retrieving the most relevant pieces when a query arrives.
Text splitting plays a crucial role in this pipeline. Choosing the right chunk size and overlap can significantly affect retrieval quality. Too small, and the model loses important context. Too large, and irrelevant information may dilute the signal. Finding the right balance often requires experimentation with different strategies and evaluation on real-world queries.
As AI systems become more integrated into critical applications, questions of reliability, transparency, and ethical use grow increasingly important. Developers and researchers must carefully consider how these technologies impact society while continuing to push the boundaries of what is possible."""

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0

)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks)


