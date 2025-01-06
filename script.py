import click
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from typing import List, Optional, TypedDict
from langgraph.graph import START, StateGraph
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
import uuid

def main():
    load_dotenv()
    
    interview()

def setup_prompt_templates():
    template_next_question = PromptTemplate.from_template("""
        You are an interviewer for the following role: {role}.
        This is the candidate's resume: {resume}.
        This is the interview so far: {history}.
        Ask your next question, based on a different entry of the candidate's resume.
        Don't repeat your questions.
        Output just the question and no extra text.
    """)

    template_followup_question = PromptTemplate.from_template("""
        You are an interviewer for the following role: {role}.
        This is the candidate's resume: {resume}.
        This is the interview so far: {history}.
        Ask a follow-up question based on the recent history around the current subject, which is the following: {question_history}.
        Don't repeat your questions.
        Output just the question and no extra text.
    """)

    template_judgement = PromptTemplate.from_template("""
        You are an interviewer for the following role: {role}.
        This is the interview so far: {history}.
        Based on the candidate's answers, would you recommend them for the role? Why or why not?
    """)
    
    return template_next_question, template_followup_question, template_judgement

def setup_state():
    class State(TypedDict):
        context: List[Document]
        question: Optional[str] = None
        question_history: Optional[str] = None
        history: Optional[str] = None
        total_followups: Optional[int] = None
        total_questions: Optional[int] = None
        result: Optional[str] = None
    
    workflow = StateGraph(State)
    
    return workflow

def setup_doc_loader(file_path):
    loader = PyPDFLoader(file_path)

    docs_content = loader.load()[0].page_content
    
    return docs_content

def setup_graph_nodes(llm, role, workflow, template_next_question, template_followup_question, template_judgement, max_questions, max_followups):
    def ask_next_question(resume, history):
        prompt = template_next_question.invoke({"role": role, "resume": resume, "history": history})
        
        question = llm.invoke(prompt)
        
        return question.content

    def handle_next_question(state):    
        question = ask_next_question(state["context"], state["history"])
        
        return {
            "question": question,
            "total_questions": state["total_questions"] + 1,
            "total_followups": 0,
            "history": state["history"] + "\n" + "QUESTION: " + question,
            "question_history": "QUESTION: " + question
        }

    def ask_followup_question(resume, history, question_history):
        prompt = template_followup_question.invoke({"role": role, "resume": resume, "history": history, "question_history": question_history})
        
        question = llm.invoke(prompt)
        
        return question.content

    def handle_followup_question(state):
        question = ask_followup_question(state["context"], state["history"], state["question_history"])
        
        return {
            "question": question,
            "total_followups": state["total_followups"] + 1,
            "history": state["history"] + "\n" + "QUESTION: " + question,
            "question_history": state["question_history"] + "\n" + "QUESTION: " + question
        }
        
    def human_answer_question(state):    
        answer = interrupt(state["question"])
        
        return {
            "history": state["history"] + "\n" + "ANSWER: " + answer,
            "question_history": state["question_history"] + "\n" + "ANSWER: " + answer
        }
        
    def judge_candidate(state):
        prompt = template_judgement.invoke({"role": role, "history": state["history"]})
        
        question = llm.invoke(prompt)
        
        return {"result": question.content}
    
    def check_for_followup_or_judgement(state):
        if state["total_questions"] ==  max_questions and state["total_followups"] == max_followups:
            return "judge_candidate"
        
        elif state["total_followups"] < max_followups:
            return "handle_followup_question"
        
        else:
            return "handle_next_question"
        
    workflow.add_node("handle_next_question", handle_next_question)
    workflow.add_node("handle_followup_question", handle_followup_question)
    workflow.add_node("judge_candidate", judge_candidate)
    workflow.add_node("human_answer_question", human_answer_question)

    workflow.set_entry_point("handle_next_question")

    workflow.add_edge("handle_next_question", "human_answer_question")
    workflow.add_edge("handle_followup_question", "human_answer_question")

    workflow.add_conditional_edges("human_answer_question", check_for_followup_or_judgement)    

def setup_checkpointer(workflow):
    checkpointer = MemorySaver()

    app = workflow.compile(checkpointer=checkpointer)
    
    return app
    
@click.command()
@click.option("--role", help="Role the user is applying for in the interview")
@click.option("--resume", help="Path to the resume of the user")
@click.option("--max_questions", help="Maximum number of questions to ask", default=1)
def interview(role, resume, max_questions):
    print(f"Applying for role: {role}")
    print(f"Resume: {resume}")
    
    llm = ChatOllama(model="llama3")
    
    docs_content = setup_doc_loader(resume)
    
    template_next_question, template_followup_question, template_judgement = setup_prompt_templates()
    
    workflow = setup_state()
    
    setup_graph_nodes(llm, role, workflow, template_next_question, template_followup_question, template_judgement, max_questions, 1)
    
    app = setup_checkpointer(workflow)
    
    thread_config = {
        "configurable": {
            "thread_id": uuid.uuid4()
        }
    }

    interview = app.invoke({
        "context": docs_content,
        "total_questions": 0,
        "total_followups": 0,
        "history": "",
        "question_history": "",
        "result": ""
        },
        config=thread_config
    )
    
    while not interview["result"]:
        print(interview["question"])
        
        answer = input("Answer: ")
        
        interview = app.invoke(
            Command(
                resume=answer),
                config=thread_config
        )
        
    print(interview["result"])
        

if __name__ == "__main__":
    main()
