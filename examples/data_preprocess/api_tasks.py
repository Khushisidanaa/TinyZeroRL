"""
Preprocess dataset for API task completion - given a user request, generate step-by-step API calls  
"""

import os
from datasets import Dataset
from random import choice, randint, seed
from typing import List, Dict
from tqdm import tqdm
import argparse


def generate_api_tasks(num_samples: int = 1000) -> List[Dict]:
    """Generate fake API task dataset"""
    
    # Sample users
    users = ["Jenish", "Sarah", "Mike", "Anna", "David", "Lisa", "John", "Emma", "Alex", "Maria"]
    
    # Sample project names  
    projects = ["Frontend", "Backend", "Database", "API", "Mobile", "Testing", "DevOps", "Analytics"]
    
    # Sample task types
    task_types = ["bug fix", "feature", "review", "deployment", "testing", "documentation"]
    
    # API call templates
    api_templates = {
        "get_user_tasks": {
            "question_templates": [
                "Get the tasks for {user}",
                "Show me {user}'s tasks",
                "What tasks does {user} have?",
                "List all tasks assigned to {user}",
                "Fetch {user}'s current tasks"
            ],
            "answer_template": "1. Call /getUserId with username='{user}' to get user ID\n2. Call /getUserTasksById with userId=<user_id> to get all tasks for {user}"
        },
        "get_user_tasks_due_today": {
            "question_templates": [
                "Get the tasks for {user} that are due today",
                "Show me {user}'s tasks due today",
                "What tasks does {user} have due today?",
                "List {user}'s tasks that need to be completed today"
            ],
            "answer_template": "1. Call /getUserId with username='{user}' to get user ID\n2. Call /getUserTasksById with userId=<user_id> and filter=due_today to get tasks due today for {user}"
        },
        "get_project_tasks": {
            "question_templates": [
                "Get all tasks for the {project} project",
                "Show me tasks in {project} project",
                "List all {project} project tasks",
                "Fetch tasks from {project} project"
            ],
            "answer_template": "1. Call /getProjectId with projectName='{project}' to get project ID\n2. Call /getProjectTasks with projectId=<project_id> to get all tasks for {project} project"
        },
        "assign_task": {
            "question_templates": [
                "Assign the {task_type} task to {user}",
                "Give the {task_type} task to {user}",
                "Assign {user} to work on the {task_type} task"
            ],
            "answer_template": "1. Call /getUserId with username='{user}' to get user ID\n2. Call /getTaskId with taskType='{task_type}' to get task ID\n3. Call /assignTask with userId=<user_id> and taskId=<task_id> to assign the task"
        },
        "create_task": {
            "question_templates": [
                "Create a new {task_type} task for {user} in {project}",
                "Add a {task_type} task for {user} in the {project} project",
                "Make a new {task_type} task assigned to {user} for {project}"
            ],
            "answer_template": "1. Call /getUserId with username='{user}' to get user ID\n2. Call /getProjectId with projectName='{project}' to get project ID\n3. Call /createTask with userId=<user_id>, projectId=<project_id>, taskType='{task_type}' to create the task"
        },
        "get_overdue_tasks": {
            "question_templates": [
                "Get {user}'s overdue tasks",
                "Show me overdue tasks for {user}",
                "List {user}'s tasks that are overdue",
                "Fetch overdue tasks assigned to {user}"
            ],
            "answer_template": "1. Call /getUserId with username='{user}' to get user ID\n2. Call /getUserTasksById with userId=<user_id> and filter=overdue to get overdue tasks for {user}"
        }
    }
    
    samples = []
    seed(42)
    
    for i in tqdm(range(num_samples)):
        # Randomly select template and fill in variables
        template_key = choice(list(api_templates.keys()))
        template = api_templates[template_key]
        
        user = choice(users)
        project = choice(projects)
        task_type = choice(task_types)
        
        question_template = choice(template["question_templates"])
        question = question_template.format(user=user, project=project, task_type=task_type)
        
        answer = template["answer_template"].format(user=user, project=project, task_type=task_type)
        
        samples.append({
            "question": question,
            "answer": answer,
            "template_type": template_key,
            "user": user,
            "project": project,
            "task_type": task_type
        })
    
    return samples


def make_prefix(question, template_type='base'):
    """Create the prompt prefix for API tasks"""
    if template_type == 'base':
        prefix = f"""A conversation between User and Assistant. The Assistant provides step-by-step API calls to complete user requests.

User: {question}
Assistant: I'll help you with that API task. Let me break it down step by step.

<thinking>
I need to determine the correct sequence of API calls to complete this request.
</thinking>

<answer>"""
    elif template_type == 'qwen-instruct':
        prefix = f"""<|im_start|>system
You are an API assistant that provides step-by-step API calls to complete user requests. Always provide clear, sequential API calls with proper parameters.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
I'll help you with that API task. Let me break it down step by step.

<thinking>
I need to determine the correct sequence of API calls to complete this request.
</thinking>

<answer>"""
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/api_tasks')
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--train_size', type=int, default=8000)
    parser.add_argument('--test_size', type=int, default=1000)
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()

    data_source = 'api_tasks'
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    # Generate the dataset
    print("Generating API tasks dataset...")
    raw_samples = generate_api_tasks(args.num_samples)
    
    # Split into train and test
    train_samples = raw_samples[:TRAIN_SIZE]
    test_samples = raw_samples[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE]

    def process_samples(samples, split):
        processed_data = []
        for idx, sample in enumerate(samples):
            question = make_prefix(sample['question'], template_type=args.template_type)
            
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "api_planning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "correct_answer": sample['answer'],
                        "question": sample['question'],
                        "template_type": sample['template_type']
                    }
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            processed_data.append(data)
        return processed_data
    
    train_dataset = Dataset.from_list(process_samples(train_samples, 'train'))
    test_dataset = Dataset.from_list(process_samples(test_samples, 'test'))

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    print(f"Saving datasets to {local_dir}")
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    
    print(f"Generated {len(train_dataset)} training samples and {len(test_dataset)} test samples")
    print("Sample data:")
    print("Question:", train_samples[0]['question'])
    print("Answer:", train_samples[0]['answer']) 