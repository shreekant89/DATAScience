# from flask import Flask, request, jsonify
# from transformers import pipeline

# app = Flask(__name__)

# # Load the question-answering model
# qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

# @app.route('/ask', methods=['POST'])
# def ask_question():
#     data = request.json
#     context = data.get("context", "")
#     question = data.get("question", "")
    
#     # Get the answer from the model
#     answer = qa_pipeline(question=question, context=context)
    
#     return jsonify({'answer': answer['answer']})

# if __name__ == "__main__":
#     app.run(debug=True)


#######################################################################
from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

app = Flask(__name__)

# Load the RoBERTa large model and tokenizer
model_name = "deepset/roberta-large-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Initialize the question-answering pipeline with the model and tokenizer
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

@app.route('/ask', methods=['POST'])
def ask_question():
    # Get data from the request
    data = request.json
    context = data.get("context", "")
    question = data.get("question", "")
    
    if not context or not question:
        return jsonify({'error': 'Context or question is missing'}), 400
    
    try:
        # Get the answer from the model
        answer = qa_pipeline(question=question, context=context)
        return jsonify({'answer': answer['answer']})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
