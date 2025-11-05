# Mental health counseling ChatGPT Clone from Scratch

# Import the libraries here
import os
from datasets import load_dataset
import json
from enum import Enum
import random
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Define the RoleType Enum here
class RoleType(Enum):
    USER = 'user'
    SYSTEM = 'system'
    ASSISTANT = 'assistant'


# Define the Role class here
class Role(object):
    def __init__(self, role_type: RoleType, content):
        self.role = role_type.value
        self.content = content
        self.value = {'role': self.role, 'content': self.content}


# Define the message class here
class Message(object):
    def __init__(self, user_content, system_content, assistant_content):
        self.user_role = Role(role_type=RoleType.USER, content=user_content)
        self.system_role = Role(role_type=RoleType.SYSTEM, content=system_content)
        self.assistant_role = Role(role_type=RoleType.ASSISTANT, content=assistant_content)
        self.message = {'messages': [self.system_role.value, self.user_role.value, self.assistant_role.value]}


# Save data in JSONL format
def save_to_jsonl(data, file_path):
    with open(file_path, 'w') as file:
        for row in data:
            line = json.dumps(row)
            file.write(line + '\n')


# Load the dataset
def load_dataset_data():
    dataset = load_dataset(
        "Amod/mental_health_counseling_conversations",
        data_files="combined_dataset.json",
        split="train"
    )
    return dataset


# Create training dataset
def create_training_dataset(dataset, system_content, sample_size=100):
    sampled_dataset = random.choices(dataset, k=sample_size)
    train_dataset = []
    
    for row in sampled_dataset:
        message_obj = Message(user_content=row['Context'], 
                            system_content=system_content, 
                            assistant_content=row['Response'])
        train_dataset.append(message_obj.message)
    
    return train_dataset


# Upload files to OpenAI
def upload_training_files(client, training_data_path, validation_data_path):
    training_data = open(training_data_path, "rb")
    validation_data = open(validation_data_path, "rb")
    
    training_response = client.files.create(file=training_data, purpose="fine-tune")
    training_file_id = training_response.id
    
    validation_response = client.files.create(file=validation_data, purpose="fine-tune")
    validation_file_id = validation_response.id
    
    return training_file_id, validation_file_id


# Create fine-tuning job
def create_fine_tuning_job(client, training_file_id, validation_file_id, model="gpt-3.5-turbo", suffix="my-test-model"):
    response = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        model=model,
        suffix=suffix,
        validation_file=validation_file_id
    )
    return response


# Get job status
def get_job_status(client, job_id):
    job_status = client.fine_tuning.jobs.retrieve(job_id)
    return job_status


# Test the fine-tuned model
def test_model(client, model, system_message, user_message):
    messages = []
    messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})
    
    completion = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return completion.choices[0].message


def main():
    # System message for the assistant
    system_content = """You serve as a supportive and honest psychology and psychotherapy assistant. Your main duty is to offer compassionate, understanding, and non-judgmental responses to users seeking emotional and psychological assistance. Respond with empathy and exhibit active listening skills. Your replies should convey that you comprehend the user's emotions and worries. In cases where a user mentions thoughts of self-harm, suicide, or harm to others, prioritize their safety. Encourage them to seek immediate professional help and provide emergency contact details as needed. It's important to note that you are not a licensed medical professional. Refrain from diagnosing or prescribing treatments. Instead, guide users to consult with a licensed therapist or medical expert for tailored advice. Never store or disclose any personal information shared by users. Uphold their privacy at all times. Avoid taking sides or expressing personal viewpoints. Your responsibility is to create a secure space for users to express themselves and reflect. Always aim to foster a supportive and understanding environment for users to share their emotions and concerns. Above all, prioritize their well-being and safety."""
    
    # Load the dataset
    print("Loading dataset...")
    dataset = load_dataset_data()

    
    # Create sample Message object
    context = dataset[152]['Context']
    response = dataset[152]['Response']
    message_obj = Message(user_content=context, system_content=system_content, assistant_content=response)
    print("Sample message object created")
    print(message_obj.message)
    
    # Create training dataset
    print("\nCreating training dataset...")
    train_dataset = create_training_dataset(dataset, system_content, sample_size=100)
    print(f"Training dataset created with {len(train_dataset)} samples")
    
    # Save data in JSONL format
    training_data_path = 'data/train.jsonl'
    validation_data_path = 'data/validation.jsonl'
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    save_to_jsonl(train_dataset[:-5], training_data_path)
    save_to_jsonl(train_dataset[-5:], validation_data_path)
    print(f"Data saved to {training_data_path} and {validation_data_path}")
    
    # Initialize OpenAI client
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Warning: OPENAI_API_KEY environment variable not set. Please set it to use fine-tuning features.")
        return
    
    client = OpenAI(api_key=api_key)
    
    # Upload files to OpenAI
    print("\nUploading files to OpenAI...")
    training_file_id, validation_file_id = upload_training_files(client, training_data_path, validation_data_path)
    print(f"Training file id: {training_file_id}")
    print(f"Validation file id: {validation_file_id}")
    
    # Create fine-tuning job
    print("\nCreating fine-tuning job...")
    fine_tuning_response = create_fine_tuning_job(client, training_file_id, validation_file_id)
    job_id = fine_tuning_response.id
    print(f"Fine-tuning job created: {job_id}")
    print(fine_tuning_response)
    
    # Retrieve job status
    print("\nRetrieving job status...")
    job_status = get_job_status(client, job_id)
    print(job_status)
    
    # Test the fine-tuned model
    print("\nTesting the fine-tuned model...")
    system_message = system_content
    user_message = "Every winter I find myself getting sad because of the weather. How can I fight this?"
    
    # Note: You'll need to update the model name with your fine-tuned model ID once training is complete
    # For now, using the base model
    completion = test_model(client, fine_tuning_response.model, system_message, user_message)
    print(completion)
    
    # Compare with base model
    print("\nComparing with base gpt-3.5-turbo model...")
    base_completion = test_model(client, "gpt-3.5-turbo", system_message, user_message)
    print(base_completion)


if __name__ == "__main__":
    main()

