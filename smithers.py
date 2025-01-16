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
from httpx import ConnectError
import uuid
from halo import Halo

from pydantic import BaseModel, Field

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
        Based on the candidate's answers, extract the properties mentioned in the 'Judgement' class.
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
        has_passed: Optional[bool] = None
        score: Optional[int] = None
    
    workflow = StateGraph(State)
    
    return workflow

def setup_doc_loader(file_path):
    loader = PyPDFLoader(file_path)

    loaded_docs = loader.load()

    docs_content = ""

    for doc in loaded_docs:
        docs_content += doc.page_content
    
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
        
        class Judgement(BaseModel):
            has_passed: str = Field(description="Whether the candidate is recommended for the role. The possible values are 'yes' or 'no'.")
            recommendation: str = Field(description="""
                Provide a recommendation based on the candidate's answers.
                Talk about competences such as technical knowledge, problem-solving skills, communication skills, initiative, adaptability, and teamwork.
                You don't need to mention all of them, mention the ones that are suitable for the questions asked.
            """) # https://www.reddit.com/r/LocalLLaMA/comments/1hcj0ur/structured_outputs_can_hurt_the_performance_of/
            score: int = Field(description="The score of the candidate. 0 out of 100.")
        
        question = llm.with_structured_output(Judgement).invoke(prompt)
        
        has_passed_bool = True if question.has_passed == "yes" else False
        
        return {"result": question.recommendation, "has_passed": has_passed_bool, "score": int(question.score)}
    
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
    
def introduce_interview(role):
    # https://patorjk.com/software/taag/#p=display&f=Slant&t=Smithers
    click.secho(r"""           
   _____           _ __  __                  
  / ___/____ ___  (_) /_/ /_  ___  __________
  \__ \/ __ `__ \/ / __/ __ \/ _ \/ ___/ ___/
 ___/ / / / / / / / /_/ / / /  __/ /  (__  ) 
/____/_/ /_/ /_/_/\__/_/ /_/\___/_/  /____/                            

""", fg="yellow")
    
    click.secho(f"Welcome to your interview for a {role} position!", fg="green")
    
    click.echo()
    
    click.pause("When you're ready, press any key to start the interview...")
    
    click.clear()

@click.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option("-r", "--role", help="Role the user is applying for in the interview", required=True)
@click.option("-max", "--max_questions", help="Maximum number of questions to ask", default=1)
def interview(filename, role, max_questions):
    """
    This script will run an interview with a candidate based on the provided resume FILENAME.\n
    Only PDF and DOCX files are supported.
    """
    # click.argument() does not take a help parameter. This is to follow the general convention of Unix tools of using arguments for only the most necessary things,
    # and to document them in the command help text by referring to them by name.
    
    if not (filename.endswith(".pdf") or filename.endswith(".docx")):
        click.secho("The resume file must be a PDF or DOCX file.", fg="red")
        
        return
    
    llm = ChatOllama(model="llama3.1")
    
    docs_content = setup_doc_loader(filename)
    
    template_next_question, template_followup_question, template_judgement = setup_prompt_templates()
    
    workflow = setup_state()
    
    setup_graph_nodes(llm, role, workflow, template_next_question, template_followup_question, template_judgement, max_questions, 1)
    
    app = setup_checkpointer(workflow)

    thread_config = {
        "configurable": {
            "thread_id": uuid.uuid4()
        }
    }

    try:         
        spinner = Halo(text="Loading the language model...", spinner="dots")
        
        spinner.start()
        
        interview = app.invoke({
            "context": docs_content,
            "total_questions": 0,
            "total_followups": 0,
            "history": "",
            "question_history": "",
            "result": "",
            },
            config=thread_config
        )
        
        spinner.stop()
        
    except ConnectError:
        click.secho("Failed to load the language model. Make sure Ollama is running llama3.1 before trying out this script.", fg="red")
        
        return
    
    introduce_interview(role)
        
    while not interview["result"]:
        question_index = interview["total_questions"] * 1 + interview["total_followups"]
        max_index = max_questions * 1 + max_questions
        
        click.secho(f"[{question_index}/{max_index}]: ", nl=False, fg="yellow")
        click.secho("Question: ", nl=False, fg="yellow")
        click.secho(interview["question"], fg="blue")
        
        click.echo()
        
        answer = ""
        
        while not answer:
            click.secho("Answer: ", nl=False, fg="yellow")
            answer = input()
            
            if not answer:
                click.secho("Please provide an answer.", fg="red")
        
        click.echo()
        
        spinner_message = ""
        
        if question_index != max_index:
            spinner_message = "Smithers is thinking about another question to ask..."
        else:
            spinner_message = "Smithers is evaluating your answers..."
            
        spinner = Halo(text=spinner_message, spinner="dots")
        
        spinner.start()
        
        interview = app.invoke(
            Command(resume=answer),
            config=thread_config
        )
        
        spinner.stop()
                
    click.secho(interview["result"], fg="green" if interview["has_passed"] else "red")
        
    def score_color(score, has_passed):
        if score >= 95:
            return "blue"
        elif score >= 90:
            return "yellow"
        elif has_passed:
            return "green"
        else:
            return "red"
    
    click.secho(f"SCORE: {interview["score"]}/100", fg=score_color(interview["score"], interview["has_passed"]))
       

if __name__ == "__main__":
    main()
