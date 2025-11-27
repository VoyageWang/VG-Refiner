import os
import json
import argparse
import numpy as np

def calculate_metrics(output_dir):
    all_results = []
    # Check if the directory exists
    if not os.path.isdir(output_dir):
        print(f"Error: Directory not found at {output_dir}")
        return

    # Aggregate results from all JSON files in the directory
    for filename in os.listdir(output_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(output_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    all_results.extend(data)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from file {filename}")
            except Exception as e:
                print(f"Warning: Error reading file {filename}: {e}")

    if not all_results:
        print("Error: No results found in the specified directory.")
        return

    # Calculate metrics
    total_samples = len(all_results)
    
    # Metrics for the final model prediction
    total_iou = sum(item.get('iou', 0) for item in all_results)
    correct_groundings = sum(item.get('is_correct', 0) for item in all_results)
    
    # Metrics for the tool's direct prediction
    total_tool_iou = sum(item.get('tool_iou', 0) for item in all_results)
    correct_tool_groundings = sum(item.get('tool_is_correct', 0) for item in all_results)

    average_iou = total_iou / total_samples if total_samples > 0 else 0
    accuracy = correct_groundings / total_samples if total_samples > 0 else 0
    
    average_tool_iou = total_tool_iou / total_samples if total_samples > 0 else 0
    tool_accuracy = correct_tool_groundings / total_samples if total_samples > 0 else 0

    # Print results
    print("\n" + "="*50)
    print("            Grounding Evaluation Results")
    print("="*50)
    print(f"  Total Samples Evaluated: {total_samples}")
    print("\n--- Final Model Performance (after using tool) ---")
    print(f"  Average IoU: {average_iou:.4f}")
    print(f"  Correct Grounding Accuracy (IoU > 0.5): {accuracy:.4f}")
    print("\n--- Tool-Only Performance  ---")
    print(f"  Average IoU: {average_tool_iou:.4f}")
    print(f"  Correct Grounding Accuracy (IoU > 0.5): {tool_accuracy:.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate grounding evaluation metrics from output files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory containing the output JSON files.")
    args = parser.parse_args()
    calculate_metrics(args.output_dir)
