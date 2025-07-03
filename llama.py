from __future__ import annotations

import textwrap

from canvas import Canvas
import json
import asyncio
import aiofiles
import datetime
import logging
from dataclasses import dataclass
from typing import ClassVar
import aiohttp
from aiohttp import StreamReader
from db import DocumentEmbedding, DatabaseFactory, ConversationSummary
from sqlalchemy.ext.asyncio import create_async_engine
from env import (
    CANVAS_URL,
    CANVAS_API_TOKEN,
    EMBEDDING_URL, COMPLETION_URL, POSTGRES_URL, COMPLETION_TOKEN, EMBEDDING_TOKEN,
)

logger = logging.getLogger(__name__)


@dataclass
class LlamaResponse:
    time_taken: datetime.timedelta
    text: str
    stream: StreamReader
    similar_docs: list[DocumentEmbedding]
    similar_scores: list[float]

    def __len__(self) -> int:
        return len(self.text)


@dataclass
class LlamaMessage:
    content: str
    role: str
    name: str | None = None

    def to_dict(self) -> dict[str, str]:
        d = {
            "content": self.content,
            "role": self.role,
        }
        if self.name:
            d["name"] = self.name
        return d

    def to_str(self) -> str:
        return f"{self.name} said: \n{self.content}" if self.name else self.content


class Llama:
    session: aiohttp.ClientSession

    default_model_name: ClassVar[str] = "llama3.1-70b"
    model_endpoints: ClassVar[dict[str, str]] = {
        default_model_name: "meta/llama-3.1-70b-instruct",
        "llama3-70b": "meta/llama3-70b-instruct",
    }

    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        self.canvas = Canvas(CANVAS_URL, CANVAS_API_TOKEN, session)

    def get_model_endpoint(self, model_name: str | None) -> str:
        if model_name is None:
            model_name = self.default_model_name
        elif model_name not in self.model_endpoints:
            logger.warning(
                f"Model '{model_name}' not found in available models. Defaulting to {self.default_model_name}",
            )
            model_name = self.default_model_name
        return self.model_endpoints[model_name]

    async def generate_embeddings(self, text: str, query: bool) -> list[float]:
        payload = {
            "input": text,
            "model": "nvidia/nv-embedqa-e5-v5",
            "input_type": "query" if query else "passage",
            "encoding_format": "float",
        }
        headers = {
            "authorization": f"Bearer {EMBEDDING_TOKEN}",
            "content-type": "application/json",
            "accept": "application/json",
        }
        response = await self.session.post(EMBEDDING_URL, headers=headers, json=payload)

        if response.status != 200:
            raise Exception(f"Failed to generate embeddings: {await response.text()}")

        js = await response.json()
        return js["data"][0]["embedding"]

    def extract_content(self, data_string: str) -> str:
        final_content = []

        # Split the string into individual lines
        lines = data_string.strip().split("\n")

        # Iterate over each line and try to extract the "content" field
        for line in lines:
            try:
                # Strip "data: " and load JSON data
                json_data = json.loads(line.removeprefix("data: "))

                # Extract the "content" field if it exists
                content = json_data["choices"][0]["delta"].get("content", None)
                if content is not None:
                    final_content.append(content)

            except (json.JSONDecodeError, KeyError):
                # Skip lines that don't have valid JSON or expected structure
                continue

        # Combine all the extracted content into a final response
        return ''.join(final_content)

    async def get_summary(self, user_id: int, course_id: int, db_factory: DatabaseFactory) -> ConversationSummary:
        """Access the past conversation summary to provide context to the current LLM completion"""
        async with db_factory() as db:
            dm_summary = await db.get_summary(user_id, course_id)
            if dm_summary is None:
                dm_summary = await db.add_dm_summary(user_id, course_id)
        return dm_summary

    async def update_summary(
        self,
        summary: ConversationSummary,
        prompt_content: str,
        assistant_content: str,
        db_factory: DatabaseFactory,
    ):
        """Summarize the conversation to provide context to future LLM completions"""
        system_prompt = """\
        You must generate a summary of a conversation between a user and an AI assistant.
        You will be given an outdated summary of the conversation, a new message from a user, and a new message from the assistant.
        You will output the new summary of the conversation incorporating all three given components.
        You must ONLY respond with the new summary, do not confirm your understanding of the task or anything like 'Here is the new summary:'.
        You must respond with a reasonably concise summary that only distills the most important information from the previous summary and the new messages.
        You must omit some details from the previous summary so that your summary does not get excessively long. Limit the response to about 1024 tokens or 4000 characters. Your new summary should not be much longer or much shorter than the old summary.

        Here is an example of a good summarization.

        Old summary:
        "The user asks what a linked list is. The assistant answers that linked lists are a data structure that stores a linear sequence in non-contiguous memory, similar to a line of people where each person knows the people adjacent to themselves. The assistant mentions that Project 1 involves implementing a linked list. The assistant asks the user if they can think of a scenario where a linked list is more suitable than an array.

        User message:
        "When are they not useful?"

        Assistant message:
        "Linked lists are less suitable than comparable data structures, such as arrays, when the use case requires frequent accesses at random indices. Arrays allow for constant time random access, while linked lists require iteration through the list."

        You should respond with something like the line below, without the quotation marks.
        "The user asks what a linked list is. The assistant answers that linked lists store a linear sequence in non-contiguous memory, similar to a line of people where each person knows the people adjacent to themselves. The assistant mentions that Project 1 involves implementing a linked list. The assistant asks the user if they can think of a scenario where a linked list is more suitable than an array. The user then asks when linked lists are not suitable, and the assistant responds that linked lists are not suited when frequent random accesses by index are necessary because arrays will be faster in that case."

        Now it's your turn, here is the summary and message data:

        Old summary:
        "{old_summary}"

        User message:
        "{user_msg}"

        Assistant message
        "{assistant_msg}"
        """
        system_prompt = textwrap.dedent(system_prompt)
        system_prompt = system_prompt.format(
            old_summary=summary.content,
            user_msg=prompt_content,
            assistant_msg=assistant_content,
        )

        model_endpoint = Llama.model_endpoints[Llama.default_model_name]
        payload_msgs = [
            {
                "role": "user",
                "content": system_prompt,
            },
        ]
        payload = {
            "model": model_endpoint,
            "messages": payload_msgs,
            "temperature": 0.1,
            "top_p": 0.1,
            "max_tokens": 1024,
        }

        headers = {
            "authorization": f"Bearer {COMPLETION_TOKEN}",
            "content-type": "application/json",
            "accept": "application/json",
        }
        response = await self.session.post(
            COMPLETION_URL,
            headers=headers,
            json=payload,
        )

        if response.status != 200:
            raise Exception(
                f"Failed to generate new summary from llama: {await response.text()}",
            )

        response_json = await response.json()
        new_summary = response_json["choices"][0]["message"]["content"]
        summary.content = new_summary
        async with db_factory() as db:
            db.add(summary)
            await db.commit()


    async def generate_response(self, prompt: str, course_id: int, db_factory: DatabaseFactory, user_id: int | None) -> str:
        """
        Generate a response based on the user's input, and find relevant documents from the database.
        """
        # Create LlamaMessage from the user prompt
        user_message = LlamaMessage(content=prompt, role="user")
        # Generate embeddings for the prompt
        
        embedding = await self.generate_embeddings(prompt, query=True)
        # Search for similar documents in the database
        async with db_factory() as db:
            similar_docs = await db.find_similar_documents(course_id, embedding, limit=5)
        # If similar documents are found, include them in the response
        if similar_docs:
            document_texts = "\n".join([doc.text for doc, _ in similar_docs])
            response_text = f"Based on relevant documents:\n{document_texts}\n\n"
        else:
            response_text = "No relevant documents found in the database."

        summary = None
        if user_id:
            summary = await self.get_summary(user_id, course_id, db_factory)
        # Complete the response using the Llama model
        response = await self.complete_chat(
            user_message,
            response_text,
            summary.content if summary else "",
        )
        if user_id:
            await self.update_summary(summary, prompt, response.text, db_factory)
        
        return response.text

    async def complete_chat(self, prompt_msg: LlamaMessage, doc_text: str, summary_content: str = "") -> LlamaResponse:
        """
        Complete the chat with a system and user prompt and return the LlamaResponse.
        """
        logger.info("Completing chat with an LLM...")

        system_prompt = """\
        Your role is to assist users in finding relevant information based on their query and provided documents.
        Do not use markdown syntax like ``` for code blocks.
        Here are the relevant documents:
        {documents}
        
        {summary_part}

        Please respond to the user's query below. At the beginning of every response, please add this warning - *This chatbot is an experimental feature, kindly verify resources before proceeding.*
        """
        summary_part = (
            f"\nBelow is a summary of the past conversation."
            f"\nIf the user seems to be referring to something from the past conversation, you should use the latter parts of the summary as context, since those are the more recent messages.\n"
            f"<summary>\n"
            f"{summary_content}\n"
            f"</summary>\n"
        )

        system_prompt = system_prompt.format(
            documents=doc_text,
            summary_part=summary_part if summary_content else "",
        )
        full_msgs = [LlamaMessage(system_prompt, "system"), prompt_msg]
        payload_msgs = self.collapse_messages(full_msgs)
        payload = {
            "model": "llama-3.1-70b-instruct",
            "messages": payload_msgs,
            "temperature": 0.1,
            "top_p": 0.1,
            "max_tokens": 1024,
            "stream": True,
        }

        start = datetime.datetime.now().astimezone()

        headers = {
            "authorization": f"Bearer {COMPLETION_TOKEN}",
            "content-type": "application/json",
            "accept": "application/json",
        }
        response = await self.session.post(COMPLETION_URL, headers=headers, json=payload)

        if response.status != 200:
            raise Exception(f"Failed to get response from llama: {await response.text()}")

        return LlamaResponse(
            time_taken=datetime.datetime.now().astimezone() - start,
            text=await response.text(),
            stream=response.content,
            similar_docs=[],
            similar_scores=[],
        )

    def collapse_messages(self, messages: list[LlamaMessage]) -> list[dict[str, str]]:
        collapsed_messages = []
        cur_msg = {"role": messages[0].role, "content": messages[0].content}

        for msg in messages[1:]:
            if cur_msg["role"] == msg.role:
                cur_msg["content"] += f"\n\n{msg.content}"
            else:
                collapsed_messages.append(cur_msg)
                cur_msg = {"role": msg.role, "content": msg.content}
        collapsed_messages.append(cur_msg)

        return collapsed_messages
    
    async def get_solution_text(self, assignment_name: str, course_id: int) -> str:
        """
        Given an assignment name, returns the combined content of its solution files.

        Args:
        assignment_name (str): The name of the assignment (case-insensitive match).
        base_dir (str): Base directory where solution files/folders are stored.

        Returns:
        str: Combined content of solution files, or an empty string if nothing found.
        """

        directory_map = {
            '987654': {
                'Palindrome Number': 'FinalFiles/Easy_Question_with_solution.md',
                'Integer to Roman' : 'FinalFiles/Medium_Question_with_solution.md',
                'Text Justification' : 'FinalFiles/Hard_Question_with_solution.md'
            },
            '876543': {
                'Palindrome Number': 'FinalFiles2/Easy_Question_Direct_Solution.md',
                'Integer to Roman' : 'FinalFiles2/Medium_Question_Direct_Solution.md',
                'Text Justification' : 'FinalFiles2/Hard_Question_Direct_Solution.md'
            },
            '529762': {
                'Coding Exercises':'FinalFiles3/Coding_Exercises.md'
            }
        }
        directory = directory_map.get(str(course_id),{}).get(assignment_name,'')

        try:
            async with aiofiles.open(directory, 'r', encoding='utf-8') as f:
                content = await f.read()
                return content
        except Exception as e:
            logger.error(f"Failed to read solution file {directory}: {e}")
            return ""
    
    async def generateprompt_on_course(self, db_factory: DatabaseFactory, course_id: int| None, codio_context: dict | None, user_id:int | None):
        """
        Generate a prompt based on the course_id to decide if solution file exists or does not exist.
        """

        assignment_content=codio_context.get("guidesPage", {}).get("content", "") # type: ignore
        assignment_name = codio_context.get('assignmentData',{}).get('assignmentName','') # type: ignore
        solution_text = await self.get_solution_text(assignment_name, course_id)
        print(solution_text)
        response_text = ''
        if solution_text:
            response_text = f"Based on Solution Document:\n{solution_text}\n\n"
        else:
            response_text = "No Solution documents found in the database."

        file_text = ""
        for file in codio_context.get("files", []): # type: ignore
            if(file['path'].split('.')[-1] == 'py'):
                file_text += file["content"]

        
        doc_text = """\
            The student's name is {name}

            The following is the assignment text.
            If the student has questions about understanding the assignment, use this content:
            {assignment_content}
        """

        doc_text = doc_text.format(
            name=codio_context.get("assignmentData", {}).get("userName", ""), # type: ignore
            assignment_content=codio_context.get("guidesPage", {}).get("content", "") # type: ignore
        )

        if course_id == 987654:
            nl_description = await self.generate_natural_language(file_text)
            doc_text += "\n\nThe following is a natural language description of the student's code:\n"
            doc_text += nl_description
            
        elif course_id == 876543:
            doc_text += "\n\nThe following are files. They may contain code written by the user.\nIf the student is asking about errors or bugs in their code, reference these files:\n"
            doc_text += file_text

        if course_id in (987654, 876543):
            
            doc_text += "\n\nThe following are relevant solution documents retrieved based on semantic similarity to the assignment:\n"
            doc_text += response_text

            doc_text+="""\
                Your goals:
                    1. Determine whether the student’s logic is **fully correct**, **partially correct**, or **incorrect**, based primarily on the assignment instructions.
                    2. Identify **any missing, incorrect, or unnecessary steps** that could cause the logic to fail or behave unexpectedly.
                    3. Provide **constructive guidance** that:
                        - Respects the student’s chosen logic and structure.
                        - Suggests improvements **within their current approach**, rather than rewriting it to match the reference.
                        - Uses the reference only for subtle checks like corner cases or formatting, not to enforce code structure.
                    4. Do **not** encourage or suggest copying or exactly replicating the reference logic.
                    5. Limit the response to 50 words or fewer.
                    6. Format the response clearly for easy readability.

                Use your explanation to:
                    - Correct logical flaws.
                    - Point out better alternatives **only if necessary for correctness or clarity**.
                    - Help the student progress with their own logic style.
            """
        else:
            doc_text += "\n\nThe following are files. They may contain code written by the user.\nIf the student is asking about errors or bugs in their code, reference these files:\n"
            doc_text += file_text
            doc_text += """\

                Instructions for response:

                Do not provide full solutions.

                Break down problems into clear, logical steps.

                Encourage critical thinking when appropriate.

                Limit the response to 50 words or fewer.

                Format the response clearly for easy readability.
            """
        if codio_context.get("error", {}).get("errorState", False): # type: ignore
            doc_text += "\n\nThe following is the student's error message:\n"
            doc_text += codio_context.get("error", {}).get("text", "") # type: ignore

        summary = None
        if user_id:
            summary = await self.get_summary(user_id, course_id, db_factory)

        return doc_text, summary

    async def generate_natural_language(self, code: str):
        prompt = f"""
            You are a code analysis tool. Convert the following python code into a detailed natural language description that captures its logic, purpose, and flow. 
            Do not format it as markdown. Do not add any commentary or explanations outside the code logic itself.

            Here is the code:
            {code}
            """
        payload_msgs = [
            {"role": "system", "content": prompt},
        ]

        payload = {
            "model": "llama-3.1-70b-instruct",
            "messages": payload_msgs,
            "temperature": 0.1,
            "top_p": 0.1,
            "max_tokens": 1024,
            "stream": False,
        }
        headers = {
            "authorization": f"Bearer {COMPLETION_TOKEN}",
            "content-type": "application/json",
            "accept": "application/json",
        }
        response = await self.session.post(COMPLETION_URL, headers=headers, json=payload)
        if response.status != 200:
            raise Exception(f"Failed to generate NL from code: {await response.text()}")
        response_json = await response.json()
        return response_json["choices"][0]["message"]["content"].strip()
async def main():
    # Replace with your actual database and NVIDIA token
    # Set up database engine and aiohttp session
    engine = create_async_engine(POSTGRES_URL)
    async with aiohttp.ClientSession() as session:
        db_factory = DatabaseFactory(engine=engine)
        llama = Llama(session)
        # Simulate user input
        user_message = "Tell me about Rishabh Assignment?"
        # Generate a response based on the user message
        response = await llama.generate_response(user_message, 506795, db_factory)
        response = llama.extract_content(response)
        print(response)
if __name__ == "__main__":
    asyncio.run(main())