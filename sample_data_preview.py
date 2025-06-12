#!/usr/bin/env python3
"""
Preview of the API Tasks Dataset
Shows exactly what questions and answers will be generated
"""

def show_sample_data():
    print("🚀 API TASKS DATASET PREVIEW")
    print("="*80)
    
    samples = [
        {
            "question": "Get the tasks for Jenish",
            "answer": "1. Call /getUserId with username='Jenish' to get user ID\n2. Call /getUserTasksById with userId=<user_id> to get all tasks for Jenish"
        },
        {
            "question": "Show me Sarah's tasks due today", 
            "answer": "1. Call /getUserId with username='Sarah' to get user ID\n2. Call /getUserTasksById with userId=<user_id> and filter=due_today to get tasks due today for Sarah"
        },
        {
            "question": "Create a new bug fix task for Mike in Frontend",
            "answer": "1. Call /getUserId with username='Mike' to get user ID\n2. Call /getProjectId with projectName='Frontend' to get project ID\n3. Call /createTask with userId=<user_id>, projectId=<project_id>, taskType='bug fix' to create the task"
        },
        {
            "question": "Get all tasks for the Backend project",
            "answer": "1. Call /getProjectId with projectName='Backend' to get project ID\n2. Call /getProjectTasks with projectId=<project_id> to get all tasks for Backend project"
        },
        {
            "question": "Get Mike's overdue tasks",
            "answer": "1. Call /getUserId with username='Mike' to get user ID\n2. Call /getUserTasksById with userId=<user_id> and filter=overdue to get overdue tasks for Mike"
        }
    ]
    
    for i, sample in enumerate(samples, 1):
        print(f"\n📋 SAMPLE {i}:")
        print(f"❓ Question: {sample['question']}")
        print(f"✅ Expected Answer:")
        for line in sample['answer'].split('\n'):
            print(f"   {line}")
        print("-" * 60)
    
    print(f"\n🎯 TRAINING PROCESS:")
    print("1. Model receives question like 'Get the tasks for Jenish'")
    print("2. Model generates response (initially random/poor)")
    print("3. Reward function compares with expected API sequence")
    print("4. Model gets reward score (0.0 to 1.0)")
    print("5. RL training improves model to get higher rewards")
    print("6. After training: model gives correct API calls!")
    
    print(f"\n🔍 LOGGING DURING TRAINING:")
    print("You'll see output like this:")
    print("="*50)
    print("🚀🚀🚀 TRAINING STEP 1 - SAMPLE 1")
    print("DATA SOURCE: api_tasks")
    print("REWARD SCORE: 0.850")
    print("🚀🚀🚀")
    print("")
    print("📝 QUESTION:")
    print("   Get the tasks for Jenish")
    print("")
    print("✅ EXPECTED ANSWER:")
    print("   1. Call /getUserId with username='Jenish' to get user ID")
    print("   2. Call /getUserTasksById with userId=<user_id> to get all tasks for Jenish")
    print("")
    print("🤖 MODEL RESPONSE:")
    print("   1. Call /getUserId with username='Jenish' to get user ID")
    print("   2. Call /getUserTasksById with userId=<user_id> to retrieve tasks for Jenish")
    print("")
    print("📊 EVALUATION:")
    print("   👍 GOOD! Score: 0.850")
    print("🚀🚀🚀")
    
    print(f"\n📊 REWARD SCORING:")
    print("🎉 PERFECT (1.0): Exact API sequence match")
    print("👍 GOOD (0.7-1.0): Mostly correct with minor differences")  
    print("⚠️  PARTIAL (0.1-0.7): Some correct API calls, wrong order/params")
    print("❌ POOR (0.0): No valid API calls or completely wrong")

if __name__ == "__main__":
    show_sample_data() 