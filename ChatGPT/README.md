1.mkdir my_ai_project
2.cd my_ai_project
3.Create a virtual environment:-
	python -m venv venv
4.Activate the virtual environment (important to ensure package installation is isolated):-
	.\venv\Scripts\activate
5.In the terminal, with the virtual environment activated, run:-
	pip install transformers torch flask
	add qa_model.py and then code as it is given in file 

6.run below coammand to run it will give answer to given question for model like Answer: the simulation of human intelligence processes by machines
	python qa_model.py 
 
7.Thats it it have run but we must want model to be dynamic take text and then analyze  so we will use Flask API
	Create a new Python file in your project folder named app.py
	Write the Flask API code that will use the BERT model for answering questions:
8.Run the Flask API on Windows
	In the VS Code terminal (with the virtual environment activated), run:
	python app.py	or in debug start python devbugger:flask
9.Open Postman or use curl in the terminal to send a request to the API:
	curl -X POST http://127.0.0.1:5000/ask -H "Content-Type: application/json" -d "{\"context\": \"AI is a branch of computer science.\", \"question\": \"What is AI?\"}"			
	curl -X POST http://127.0.0.1:5000/ask -H "Content-Type: application/json" -d "{\"context\": \"shreekant running faster than sneha,sneha running faster than shreenika, shreenika running faster than raju .\", \"question\": \"who fastest person among all?\"}"
10. or else run directly launchjson in vscode and do above curl command 