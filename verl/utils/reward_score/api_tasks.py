import re
import random
from difflib import SequenceMatcher


def extract_api_solution(solution_str):
    """Extract the API calls from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None
    
    # Look for answer tags
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, solution_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # If no answer tags, try to extract after "step by step" or similar phrases
    patterns = [
        r'step by step[.:]?\s*(.*?)(?:\n\n|\Z)',
        r'API calls?[.:]?\s*(.*?)(?:\n\n|\Z)', 
        r'breakdown[.:]?\s*(.*?)(?:\n\n|\Z)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, solution_str, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return None


def normalize_api_call(api_call):
    """Normalize API call format for comparison"""
    # Remove extra whitespace and standardize format
    api_call = re.sub(r'\s+', ' ', api_call.strip())
    
    # Standardize parameter format
    api_call = re.sub(r'with\s+', 'with ', api_call)
    api_call = re.sub(r'=\s*', '=', api_call)
    api_call = re.sub(r'\s*,\s*', ', ', api_call)
    
    return api_call.lower()


def extract_api_calls(text):
    """Extract individual API calls from text"""
    if not text:
        return []
    
    # Split by numbered steps or bullet points
    steps = re.split(r'(?:^|\n)\s*(?:\d+\.|\-|\•)\s*', text, flags=re.MULTILINE)
    steps = [step.strip() for step in steps if step.strip()]
    
    api_calls = []
    for step in steps:
        # Look for API endpoint patterns
        api_match = re.search(r'call\s+(/\w+(?:/\w+)*)', step, re.IGNORECASE)
        if api_match:
            api_calls.append(normalize_api_call(step))
    
    return api_calls


def compute_similarity(predicted, expected):
    """Compute similarity between predicted and expected API calls"""
    if not predicted or not expected:
        return 0.0
    
    return SequenceMatcher(None, predicted, expected).ratio()


def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.0):
    """The scoring function for API tasks.
    
    Args:
        solution_str: the solution text from the model
        ground_truth: dictionary containing correct answer and question
        method: the method to extract the solution
        format_score: the score for partially correct format
        score: the score for the correct answer
    """
    expected_answer = ground_truth['correct_answer']
    question = ground_truth.get('question', 'Unknown question')
    
    # Extract the model's response
    predicted_answer = extract_api_solution(solution_str)
    
    # Random logging (1 in 32 chance)
    do_print = random.randint(1, 32) == 1
    
    if do_print:
        print(f"\n" + "="*80)
        print(f"🔍 API TASK EVALUATION")
        print(f"Question: {question}")
        print(f"Expected: {expected_answer}")
        print(f"Predicted: {predicted_answer}")
        print(f"Full Response: {solution_str[:200]}...")
        print("="*80)
    
    if predicted_answer is None:
        if do_print:
            print("❌ No API calls found in response")
        return 0
    
    # Extract API calls from both expected and predicted
    expected_calls = extract_api_calls(expected_answer)
    predicted_calls = extract_api_calls(predicted_answer)
    
    if not predicted_calls:
        if do_print:
            print("❌ No valid API calls extracted")
        return format_score
    
    # Exact match check
    if len(expected_calls) == len(predicted_calls):
        exact_match = True
        for i, (exp, pred) in enumerate(zip(expected_calls, predicted_calls)):
            similarity = compute_similarity(pred, exp)
            if similarity < 0.8:  # Allow for minor formatting differences
                exact_match = False
                break
        
        if exact_match:
            if do_print:
                print("✅ Exact match found!")
            return score
    
    # Partial scoring based on similarity
    if len(expected_calls) > 0:
        total_similarity = 0
        for i, expected_call in enumerate(expected_calls):
            if i < len(predicted_calls):
                similarity = compute_similarity(predicted_calls[i], expected_call)
                total_similarity += similarity
            # Penalty for missing calls
        
        avg_similarity = total_similarity / len(expected_calls)
        
        # Penalty for wrong number of API calls
        length_penalty = 1.0
        if len(predicted_calls) != len(expected_calls):
            length_penalty = 0.8
        
        final_score = avg_similarity * length_penalty
        
        if do_print:
            print(f"📊 Partial match - Similarity: {avg_similarity:.2f}, Length penalty: {length_penalty:.2f}")
            print(f"Final score: {final_score:.2f}")
        
        if final_score > 0.7:
            return score * final_score
        elif final_score > 0.4:
            return format_score
        else:
            return 0
    
    if do_print:
        print("❌ No scoring criteria met")
    return 0 