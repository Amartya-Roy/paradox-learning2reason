#!/usr/bin/env python3
"""
Helper script to run evaluation with the correct model path
"""
import os
import subprocess
import sys

def find_latest_checkpoint():
    """Find the latest checkpoint in the output directory"""
    output_dir = "OUTPUT/RP/BERT/NEW_TOKENS"
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist")
        return None
    
    # Find all checkpoint directories
    checkpoints = []
    for item in os.listdir(output_dir):
        if item.startswith("checkpoint-"):
            try:
                checkpoint_num = int(item.split("-")[1])
                checkpoints.append((checkpoint_num, item))
            except ValueError:
                continue
    
    if not checkpoints:
        print("No checkpoints found")
        return None
    
    # Sort by checkpoint number and get the latest
    checkpoints.sort(key=lambda x: x[0])
    latest_checkpoint = checkpoints[-1][1]
    
    print(f"Found checkpoints: {[c[1] for c in checkpoints]}")
    print(f"Using latest checkpoint: {latest_checkpoint}")
    
    return os.path.join(output_dir, latest_checkpoint)

def find_model_file(checkpoint_dir):
    """Find the model file in the checkpoint directory"""
    # Check for safetensors format first (newer)
    safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.exists(safetensors_path):
        return safetensors_path
    
    # Check for pytorch_model.bin format (older)
    pytorch_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if os.path.exists(pytorch_path):
        return pytorch_path
    
    return None

def main():
    # Find the latest checkpoint
    checkpoint_dir = find_latest_checkpoint()
    if not checkpoint_dir:
        print("No valid checkpoint found")
        sys.exit(1)
    
    # Find the model file
    model_file = find_model_file(checkpoint_dir)
    if not model_file:
        print(f"No model file found in {checkpoint_dir}")
        sys.exit(1)
    
    print(f"Model file: {model_file}")
    
    # Run the evaluation
    cmd = [
        "python", "finetune_simplified.py",
        "--model_type", "bert",
        "--tokenizer_name", "bert-base-uncased",
        "--model_name_or_path", "bert-base-uncased",
        "--do_eval",
        "--do_lower_case",
        "--save_steps", "-1",
        "--per_gpu_eval_batch_size", "2",
        "--per_gpu_train_batch_size", "2",
        "--overwrite_output_dir",
        "--num_workers", "1",
        "--max_length", "1000",
        "--output_dir", "OUTPUT/eval",
        "--group_by_which_depth", "depth",
        "--limit_report_max_depth", "6",
        "--change_positional_embedding_before_loading",
        "--val_file_path", "DATA/RP/prop_examples.balanced_by_backward.max_6.json_val",
        "--custom_weight", model_file,
        "--add_special_tokens"
    ]
    
    print("Running evaluation with command:")
    print(" ".join(cmd))
    print()
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True)
        print("\nEvaluation completed successfully!")
        
        # Show results
        if os.path.exists("eval_result.txt"):
            print("\nResults from eval_result.txt:")
            with open("eval_result.txt", "r") as f:
                print(f.read())
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
