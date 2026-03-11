import os
import json
import time
import requests
import argparse
import random
from typing import Tuple, Dict, Any, Optional
from multiprocessing import Pool
from tqdm import tqdm


class OpenAIClient:
    """
    OpenAI API Client with support for multiple models and custom endpoints.
    
    Supports:
    - Standard OpenAI models (gpt-4, gpt-4o, etc.)
    - GPT-5 with reasoning effort levels
    - Custom API endpoints
    """
    
    def __init__(self, 
                 model: str = "gpt-4o",
                 temperature: float = 0.0,
                 system_prompt: str = "",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        self.temperature = temperature
        self.system_prompt = system_prompt
        
        # Handle GPT-5 reasoning effort models
        if model in ["gpt-5-low", "gpt-5-medium", "gpt-5-high"]:
            self.model = "gpt-5"
            if "low" in model:
                self.reasoning_effort = "low"
            elif "medium" in model:
                self.reasoning_effort = "medium"
            else:
                self.reasoning_effort = "high"
        else:
            self.model = model
            self.reasoning_effort = None
        
        # API configuration
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.base_url = base_url or os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
        
        if not self.api_key:
            raise ValueError("API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.url = f"{self.base_url.rstrip('/')}/chat/completions"

    def chat(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        send_json = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature
        }
        
        # Add reasoning effort for GPT-5
        if self.reasoning_effort:
            send_json["reasoning_effort"] = self.reasoning_effort
        
        try:
            headers = {
                'Content-Type': 'application/json',
                "Authorization": f"Bearer {self.api_key}",
            }
            
            response = requests.post(
                self.url, 
                headers=headers, 
                json=send_json, 
                timeout=(30, 1800)  # (connect_timeout, read_timeout)
            )
            response.raise_for_status()
            
            response_data = response.json()
            content = response_data["choices"][0]["message"]["content"]
            
            return content, response_data
            
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return 'TIMEOUT', {}
        except (KeyError, IndexError) as e:
            print(f"Response parsing error: {e}")
            return 'TIMEOUT', {}
        except Exception as e:
            print(f"Unexpected error: {e}")
            return 'TIMEOUT', {}


def process_response(response: str) -> str:
    if not response:
        return ""
    
    # Replace unwanted patterns
    replacements = {
        "Updated Review": "Review",
        "Updated Assessment": "Overall Assessment",
        "Updated Score": "Score"
    }
    
    for old, new in replacements.items():
        response = response.replace(old, new)
    
    # Filter out lines containing specific phrases
    lines = response.split("\n")
    filtered_lines = []
    
    for line in lines:
        if not any(phrase in line.lower() for phrase in ["current review", "current assessment"]):
            filtered_lines.append(line)
    
    try:
        return "\n".join(filtered_lines)
    except Exception:
        print("Error processing response")
        return ""


def single_evaluate(args: Tuple[str, str, str, str, str, int, Optional[str], Optional[str]]) -> None:

    filename, model, prompts_dir, output_dir, reasoning_dir, max_retries, api_key, base_url = args
    
    try:
        # Initialize client
        client = OpenAIClient(
            model=model,
            temperature=0,
            system_prompt="",
            api_key=api_key,
            base_url=base_url
        )
        
        # Read prompt
        prompt_path = os.path.join(prompts_dir, f"{filename}.txt")
        if not os.path.exists(prompt_path):
            print(f"Prompt file not found: {prompt_path}")
            return
            
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt = f.read()
        
        # Try to get response with retries
        response = 'TIMEOUT'
        response_json = {}
        
        for attempt in range(max_retries):
            print(f"Processing {filename}, attempt {attempt + 1}/{max_retries}")
            response, response_json = client.chat(prompt)
            
            if response != 'TIMEOUT':
                break
                
            if attempt < max_retries - 1:
                sleep_time = random.uniform(1, 3)  # Random backoff
                print(f"Retrying in {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
        
        if response == 'TIMEOUT':
            print(f"Failed to process {filename} after {max_retries} attempts")
            return
        
        # Process response
        processed_response = process_response(response)
        
        # Save results
        reasoning_path = os.path.join(reasoning_dir, f"{filename}.json")
        with open(reasoning_path, 'w', encoding='utf-8') as f:
            json.dump(response_json, f, indent=2, ensure_ascii=False)
        
        output_path = os.path.join(output_dir, f"{filename}.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(processed_response)
        
        print(f"Successfully processed {filename}")
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")


def main():
    """Main function to run batch evaluation."""
    parser = argparse.ArgumentParser(
        description="Batch evaluate text files using OpenAI API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model and execution parameters
    parser.add_argument("--model", type=str, default="gpt-4o",
                       help="Model to use (e.g., gpt-4o, gpt-5-low, deepseek-reasoner)")
    parser.add_argument("--run_name", type=str, default="run_1",
                       help="Name for this evaluation run")
    
    # File paths
    parser.add_argument("--prompts_dir", type=str, required=True,
                       help="Directory containing prompt files (.txt)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Base directory for outputs")
    
    # API configuration
    parser.add_argument("--api_key", type=str,
                       help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--base_url", type=str,
                       help="Custom API base URL (or set OPENAI_BASE_URL env var)")
    
    # Execution parameters
    parser.add_argument("--processes", type=int, default=4,
                       help="Number of parallel processes")
    parser.add_argument("--max_retries", type=int, default=5,
                       help="Maximum number of retries per file")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode (single process)")
    
    args = parser.parse_args()
    
    # Setup output directories
    model_output_dir = os.path.join(args.output_dir, args.model, args.run_name)
    reasoning_dir = os.path.join(args.output_dir, args.model, "reasoning", args.run_name)
    
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(reasoning_dir, exist_ok=True)
    
    # Find files to process
    if not os.path.exists(args.prompts_dir):
        print(f"Error: Prompts directory not found: {args.prompts_dir}")
        return
    
    # Get already processed files
    processed_files = set()
    if os.path.exists(model_output_dir):
        for filename in os.listdir(model_output_dir):
            if filename.endswith('.txt'):
                processed_files.add(filename[:-4])  # Remove .txt extension
    
    print(f"Already processed: {len(processed_files)} files")
    
    # Get all prompt files
    all_files = []
    for filename in os.listdir(args.prompts_dir):
        if filename.endswith('.txt') and not filename.startswith('.'):
            file_id = filename[:-4]  # Remove .txt extension
            if file_id not in processed_files:
                all_files.append(file_id)
    
    print(f"Files to process: {len(all_files)}")
    
    if not all_files:
        print("No files to process!")
        return
    
    # Prepare arguments for multiprocessing
    task_args = [
        (
            filename, args.model, args.prompts_dir, model_output_dir, 
            reasoning_dir, args.max_retries, args.api_key, args.base_url
        )
        for filename in all_files
    ]
    
    # Process files
    if args.debug or len(all_files) == 1:
        # Single process for debugging
        print("Running in debug mode (single process)")
        for task_arg in tqdm(task_args, desc="Processing files"):
            single_evaluate(task_arg)
    else:
        # Multiprocessing
        print(f"Processing {len(all_files)} files with {args.processes} processes")
        with Pool(processes=args.processes) as pool:
            list(tqdm(pool.imap(single_evaluate, task_args), 
                     total=len(task_args), desc="Processing files"))
    
    print("Evaluation completed!")


if __name__ == '__main__':
    main()