#!/bin/bash

# Generate API Tasks Dataset
echo "🚀 Generating API Tasks Dataset..."

# Set variables
DATA_DIR="$HOME/data/api_tasks"
TRAIN_SIZE=8000
TEST_SIZE=1000
TEMPLATE_TYPE="base"  # or "qwen-instruct" for instruct models

# Create directory
mkdir -p $DATA_DIR

# Generate the dataset
python3 examples/data_preprocess/api_tasks.py \
    --local_dir $DATA_DIR \
    --train_size $TRAIN_SIZE \
    --test_size $TEST_SIZE \
    --template_type $TEMPLATE_TYPE

echo "✅ Dataset generated successfully!"
echo "📁 Location: $DATA_DIR"
echo "📊 Training samples: $TRAIN_SIZE"
echo "🧪 Test samples: $TEST_SIZE"

# Show sample data
echo ""
echo "📋 Sample questions from the dataset:"
echo "  - Get the tasks for Jenish"
echo "  - Show me Sarah's tasks due today"  
echo "  - Create a new bug fix task for Mike in Frontend"
echo "  - Get all tasks for the Backend project"
echo ""
echo "🎯 Now set your environment variables:"
echo "export DATA_DIR=$DATA_DIR"
echo "export BASE_MODEL={path_to_your_model}"
echo "export EXPERIMENT_NAME=api-tasks-rl" 