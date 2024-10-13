from transformers import pipeline

# Initialize the Hugging Face pipeline for question-answering
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad") #"distilbert-base-uncased-distilled-squad","bert-large-uncased-whole-word-masking-finetuned-squad"

# Example context (input paragraph)
context = """
Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems. 
These processes include learning, reasoning, and self-correction. AI is used in expert systems, NLP, speech recognition, etc.
"""

# Example question based on the paragraph
question = "What is AI?"

# Get the answer
result = qa_pipeline(question=question, context=context)
print(f"Answer: {result['answer']}")
