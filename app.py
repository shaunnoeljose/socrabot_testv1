from flask import request, jsonify
import aiohttp
from sqlalchemy.ext.asyncio import create_async_engine
from llama import Llama, LlamaMessage
from db import DatabaseFactory
from env import POSTGRES_URL
from quart import Quart, request, jsonify # type: ignore
from quart_cors import cors # type: ignore
import json
from datetime import datetime, timezone
import csv
import logging

app = Quart(__name__)
app = cors(app, allow_origin="*")

logger = logging.getLogger(__name__)

# Global variables to hold the Llama, session and DB factory instances
llama = None
document_embedder = None
session = None

# Async initialization function for Llama
async def init_llama():
    global llama, session
    session = aiohttp.ClientSession()  # Initialize aiohttp session
    llama = Llama(session)

@app.before_serving
async def startup():
    await init_llama()

@app.after_serving
async def shutdown():
    global session
    if session:
        await session.close()

@app.route('/ask-code', methods=['POST'])
async def ask_code():
    
    data = await request.get_json()

    user_input = data.get('message', '')
    codio_context = data.get('codio_context')
    print(codio_context)
    courseMap = {
        "COP 2273 - Spring 2025": "523756",
        "COP2273 - Fall 2024": "506849",
        "CAP5771 - Intro to Data Science": "529762",
        "Testing course 2": "987654",
        "Testing course": "876543",
        "COP 2273 - Summer 2025": "534534",
        "Not Exist": '000000'
    }
    course_id = courseMap[codio_context.get('assignmentData',{}).get('courseName','Not Exist')]
    user_id = data.get('user_id', '')
    user_id = int(user_id) if user_id else None
    prompt_message = LlamaMessage(content=user_input, role="user")


    engine = create_async_engine(POSTGRES_URL)
    db_factory = DatabaseFactory(engine=engine)
    
    doc_text, summary = await llama.generateprompt_on_course(db_factory, int(course_id), codio_context, user_id) # type: ignore

    response = await llama.complete_chat(prompt_message, doc_text, summary_content=summary.content if summary else '') # type: ignore
    max_code_lines = 5
    def count_code_lines(text: str):
        in_block = False
        count = 0
        for line in text.splitlines():
            if "```" in line:
                in_block = not in_block
                continue
            if in_block or line.strip().startswith(("def ", "class ", "for ", "while ", "if ", "elif ", "else:", "try:", "except", "with ", "import ", "from ", "#", "@", "return", "print(")):
                count += 1
        return count
    max_retries = 2
    retries = 0

    while True:
        if user_id:
            await llama.update_summary(summary, user_input, response.text, db_factory) # type: ignore
        code_line_count = count_code_lines(response.text)
        if code_line_count > max_code_lines:
            logger.warning(f"Response blocked due to excessive code: {code_line_count} lines.")
            if retries >= max_retries:
                logger.error("Max retries reached. Returning last response anyway.")
                break
            retry_prompt = LlamaMessage(
                content=f"Your previous response contained too many lines of code ({code_line_count}). Please reduce it to **at most {max_code_lines} lines**, and focus on reasoning or pseudocode.",
                role="system"
            )
            response = await llama.complete_chat(retry_prompt, doc_text, summary.content if summary else "") # type: ignore
            retries += 1
        else:
            break
    final_response = llama.extract_content(response.text) # type: ignore

    log_entry = f"Course ID: {course_id}, User Input: {user_input}\n"
    response_entry = f"Response: {final_response}\n"
    with open("api_requests.log", "a") as log_file:
        log_file.write(log_entry)
        log_file.write(response_entry)

    user_id = codio_context.get('assignmentData',{}).get('userId','')
    user_name = codio_context.get('assignmentData',{}).get('userName','')
    utc_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    exercise = codio_context.get('assignmentData',{}).get('assignmentName','')
    course_id = codio_context.get('assignmentData',{}).get('courseName','')
    if(len(course_id)==0):
        course_id = data.get('course_id')
    json_data = {
        'CourseID': course_id, 
        'UserInput': user_input, 
        'Response': final_response, 
        'UserID': user_id, 
        'UserName': user_name,
        'UTCTime': utc_time,
        'assignment': exercise
    }
    with open("api_requests.json",'r+') as json_file:
        try:
            file_data = json.load(json_file)
        except json.JSONDecodeError:
            file_data = []

        if not isinstance(file_data, list):
            file_data = [file_data]
        file_data.append(json_data)
        json_file.seek(0)
        json.dump(file_data, json_file, indent=4)
        json_file.truncate()
    data_to_csv = [utc_time,user_name,codio_context]
    with open('codio_context.csv','a') as log_csv:
        writer = csv.writer(log_csv)
        writer.writerow(data_to_csv)

    return jsonify({"response": final_response})

@app.route('/ask', methods=['OPTIONS'])
def handle_options():
    return jsonify({}), 200

@app.route('/ask', methods=['POST'])
async def ask():
    try:
        # Parse the course_id and user input from the request body (expects JSON with "course_id" and "message" fields)
        data = await request.get_json()
        course_id = data.get('course_id')
        user_input = data.get('message', '')
        codio_context = data.get('codio_context',{})
        user_id = data.get('user_id', '')
        user_id = int(user_id) if user_id else None
    
        if not course_id:
            return jsonify({"error": "No course ID provided"}), 400
        if not user_input:
            return jsonify({"error": "No message provided"}), 400
        
        log_entry = f"Course ID: {course_id}, User Input: {user_input}\n"
        # with open("api_requests.log", "a") as log_file:
        #     log_file.write(log_entry)
        
        # Dynamically create the engine and db_factory for the specified course ID
        engine = create_async_engine(POSTGRES_URL)
        db_factory = DatabaseFactory(engine=engine)
        # Generate a response using the Llama model
        response_text = await llama.generate_response( # type: ignore
            user_input,
            int(course_id),
            db_factory,
            user_id,
        )
        final_response = llama.extract_content(response_text) # type: ignore

        response_entry = f"Response: {final_response}\n"
        response_entry = response_entry.replace('\n', '<br>')
        with open("api_requests.log", "a") as log_file:
            log_file.write(log_entry)
            log_file.write(response_entry)

        user_id = codio_context.get('assignmentData', {}).get('userId','')
        user_name = codio_context.get('assignmentData', {}).get('userName','')
        utc_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        if not user_input:
            exercise = codio_context.get('assignmentData', {}).get('assignmentName','')
            course_id = codio_context.get('assignmentData',{}).get('courseName','')
        if(len(course_id)==0):
            course_id = data.get('course_id')
        json_data = {
            'CourseID': course_id, 
            'UserInput': user_input, 
            'Response': final_response, 
            'UserID': user_id,
            'UserName': user_name,
            'UTCTime': utc_time,
            'assignment': exercise # type: ignore
        }
        
        with open("api_requests.json",'r+') as json_file:
            try:
                file_data = json.load(json_file)
            except json.JSONDecodeError:
                file_data = []
            if not isinstance(file_data, list):
                file_data = [file_data]
            file_data.append(json_data)
            json_file.seek(0)
            json.dump(file_data, json_file, indent=4)
            json_file.truncate()
        return jsonify({"response": final_response})
    except Exception as e:
        #raise e
        return jsonify({"error": str(e)}), 500
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=20601)