import os
import json
import argparse
from glob import glob
import sys
from collections import defaultdict
import csv

def save_metrics_to_csv(all_metrics, csv_file_path):
    """Saves the calculated metrics to a CSV file."""
    if not all_metrics:
        print("No metrics to save.")
        return

    headers = [
        "Dataset", "Total Cases", "Accuracy (%)", "Tool Accuracy (%)", 
        "NSRI (%)", "NSRI When Tool Correct (%)", "NSRI When Tool Wrong (%)", "Scaled NSRI (0-1 Range)",
        "Avg. IoU Delta", 
        "Critical Correction Rate (%)", "Critical Correction Count", "Tool is Wrong Count", "Avg. Relative IoU Gain (Critical) (%)",
        "Refinement Rate (%)", "Refinement Count", "Tool is Correct Count", "Avg. Relative IoU Gain (Refinement) (%)",
        "Follow Correct Rate (%)", "Follow Correct Count",
        "Worsen Rate (%)", "Worsen Count", "Avg. Relative IoU Worsen (%)",
        "Inherited Error Rate (%)", "Inherited Error Count", "Final Wrong Count"
    ]

    try:
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            sorted_datasets = sorted(all_metrics.keys())
            for dataset_name in sorted_datasets:
                metrics = all_metrics[dataset_name]
                worsen_rate = (metrics['worsen_count'] / metrics['total_cases']) * 100 if metrics['total_cases'] > 0 else 0
                
                row = [
                    dataset_name,
                    metrics['total_cases'],
                    f"{metrics['accuracy']:.2f}",
                    f"{metrics['tool_accuracy']:.2f}",
                    f"{metrics['nsri']:.2f}",
                    f"{metrics['nsri_tool_correct']:.2f}" if metrics['tool_is_correct_count'] > 0 else "N/A",
                    f"{metrics['nsri_tool_wrong']:.2f}" if metrics['tool_is_wrong_count'] > 0 else "N/A",
                    f"{metrics['nsri_scaled_0_1']:.4f}",
                    f"{metrics['avg_iou_delta']:.4f}",
                    f"{metrics['critical_correction_rate']:.2f}",
                    metrics['critical_correction_count'],
                    metrics['tool_is_wrong_count'],
                    f"{metrics['avg_iou_gain_critical']:.2f}" if metrics['critical_correction_count'] > 0 else "N/A",
                    f"{metrics['refinement_rate']:.2f}",
                    metrics['refinement_count'],
                    metrics['tool_is_correct_count'],
                    f"{metrics['avg_iou_gain_refinement']:.2f}" if metrics['refinement_count'] > 0 else "N/A",
                    f"{metrics['follow_correct_rate']:.2f}",
                    metrics['follow_correct_count'],
                    f"{worsen_rate:.2f}",
                    metrics['worsen_count'],
                    f"{metrics['avg_iou_worsen']:.2f}" if metrics['worsen_count'] > 0 else "N/A",
                    f"{metrics['inherited_error_rate']:.2f}" if metrics['final_wrong_count'] > 0 else "N/A",
                    metrics['inherited_error_count'],
                    metrics['final_wrong_count']
                ]
                writer.writerow(row)
        print(f"\nMetrics summary successfully saved to '{csv_file_path}'")
    except IOError as e:
        print(f"Error: Could not write to CSV file '{csv_file_path}': {e}", file=sys.stderr)


def collect_metrics(results_dir, output_file, csv_file):
    """
    Collects and calculates metrics from grounding evaluation results.

    Args:
        results_dir (str): The directory containing the evaluation results.
                           This directory is expected to have subdirectories for each dataset.
        output_file (str): The path to the output JSON file to save the collected cases.
    """
    if not os.path.isdir(results_dir):
        print(f"Error: Results directory not found at '{results_dir}'", file=sys.stderr)
        return

    try:
        dataset_dirs = [d for d in os.scandir(results_dir) if d.is_dir()]
    except FileNotFoundError:
        print(f"Error: Unable to scan directory '{results_dir}'", file=sys.stderr)
        return

    print(f"Found {len(dataset_dirs)} dataset directories in '{results_dir}'\n")

    all_metrics = {}
    collected_cases = {"critical_correction_cases": [], "refinement_cases": []}

    for dataset_dir in sorted(dataset_dirs, key=lambda d: d.name):
        dataset_name = dataset_dir.name
        json_files = glob(os.path.join(dataset_dir.path, "output_*.json"))
        
        if not json_files:
            continue
        
        total_cases = 0
        is_correct_count = 0
        tool_is_correct_count = 0
        
        # Counters for new metrics
        tool_is_wrong_count = 0
        critical_correction_count = 0
        refinement_count = 0
        total_iou_delta = 0
        # New counters for detailed IoU analysis
        total_iou_gain_critical = 0
        total_iou_gain_refinement = 0
        worsen_count = 0
        total_iou_worsen = 0
        follow_correct_count = 0
        total_nsri_gain = 0
        # New counter for inherited errors
        inherited_error_count = 0
        final_wrong_count = 0
        # New counters for conditional NSRI
        total_nsri_gain_tool_correct = 0
        total_nsri_gain_tool_wrong = 0

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        total_cases += 1
                        
                        iou = item.get('iou')
                        tool_iou = item.get('tool_iou')

                        if iou is None or tool_iou is None:
                            continue

                        # Calculate g_i for NSRI
                        g_i = 0
                        iou_delta = iou - tool_iou
                        
                        if iou_delta > 1e-9:  # Improvement
                            denominator = 1.0 - tool_iou
                            if denominator > 1e-9:
                                g_i = iou_delta / denominator
                        elif iou_delta < -1e-9:  # Degradation
                            denominator = tool_iou
                            if denominator > 1e-9:
                                g_i = iou_delta / denominator
                        
                        total_nsri_gain += g_i

                        is_correct = iou > 0.5
                        tool_is_correct = tool_iou > 0.5
                        
                        # Accumulate conditional NSRI
                        if tool_is_correct:
                            total_nsri_gain_tool_correct += g_i
                        else:
                            total_nsri_gain_tool_wrong += g_i
                        
                        if is_correct:
                            is_correct_count += 1
                        else:
                            final_wrong_count += 1
                            # Check if the error was inherited from the tool without change
                            if not tool_is_correct and abs(iou - tool_iou) < 1e-9:
                                inherited_error_count += 1
                        
                        if tool_is_correct:
                            tool_is_correct_count += 1
                            # Check for refinement
                            if iou > tool_iou:
                                refinement_count += 1
                                # Convert to relative percentage gain
                                if (1.0 - tool_iou) > 1e-9:
                                    total_iou_gain_refinement += (iou - tool_iou) / (1.0 - tool_iou)
                            # Check for following correct prediction
                            elif abs(iou - tool_iou) < 1e-9:
                                follow_correct_count += 1
                        else:
                            tool_is_wrong_count += 1
                            # Check for critical correction
                            if is_correct:
                                critical_correction_count += 1
                                # Convert to relative percentage gain
                                if (1.0 - tool_iou) > 1e-9:
                                    total_iou_gain_critical += (iou - tool_iou) / (1.0 - tool_iou)

                        if iou < tool_iou:
                            worsen_count += 1
                            # Convert to relative percentage worsen
                            if tool_iou > 1e-9:
                                total_iou_worsen += (iou - tool_iou) / tool_iou
                            
                        total_iou_delta += (iou - tool_iou)
                        
                        # Prepare case data for collection
                        case_data = None
                        
                        # Check for critical correction case to collect
                        if not tool_is_correct and is_correct:
                            case_data = item.copy()
                            collected_cases["critical_correction_cases"].append(case_data)

                        # Check for refinement case to collect
                        if tool_is_correct and iou > tool_iou:
                            case_data = item.copy()
                            collected_cases["refinement_cases"].append(case_data)

                        # Add visualization path if case was collected
                        if case_data:
                            image_id = item.get('image_id')
                            ann_id = item.get('ann_id')
                            if image_id and ann_id:
                                vis_filename = f"{image_id}_{ann_id}.jpg"
                                case_data['visualization_path'] = os.path.join(dataset_dir.path, 'visualizations', vis_filename)
                            else:
                                case_data['visualization_path'] = None


            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from file '{json_file}'", file=sys.stderr)
            except Exception as e:
                print(f"An unexpected error occurred while processing {json_file}: {e}", file=sys.stderr)

        if total_cases > 0:
            accuracy = (is_correct_count / total_cases) * 100
            tool_accuracy = (tool_is_correct_count / total_cases) * 100
            critical_correction_rate = (critical_correction_count / tool_is_wrong_count) * 100 if tool_is_wrong_count > 0 else 0
            refinement_rate = (refinement_count / tool_is_correct_count) * 100 if tool_is_correct_count > 0 else 0
            avg_iou_delta = total_iou_delta / total_cases
            
            # Calculate detailed IoU metrics as percentages
            avg_iou_gain_critical = (total_iou_gain_critical / critical_correction_count) * 100 if critical_correction_count > 0 else 0
            avg_iou_gain_refinement = (total_iou_gain_refinement / refinement_count) * 100 if refinement_count > 0 else 0
            avg_iou_worsen = (total_iou_worsen / worsen_count) * 100 if worsen_count > 0 else 0
            follow_correct_rate = (follow_correct_count / tool_is_correct_count) * 100 if tool_is_correct_count > 0 else 0
            
            nsri_raw = (total_nsri_gain / total_cases) if total_cases > 0 else 0
            nsri = nsri_raw * 100
            # Scaled NSRI from [-1, 1] to [0, 1]
            nsri_scaled_0_1 = (nsri_raw + 1) / 2
            inherited_error_rate = (inherited_error_count / final_wrong_count) * 100 if final_wrong_count > 0 else 0

            # Calculate conditional NSRI
            nsri_tool_correct = (total_nsri_gain_tool_correct / tool_is_correct_count) * 100 if tool_is_correct_count > 0 else 0
            nsri_tool_wrong = (total_nsri_gain_tool_wrong / tool_is_wrong_count) * 100 if tool_is_wrong_count > 0 else 0

            all_metrics[dataset_name] = {
                'total_cases': total_cases,
                'accuracy': accuracy,
                'tool_accuracy': tool_accuracy,
                'critical_correction_rate': critical_correction_rate,
                'refinement_rate': refinement_rate,
                'avg_iou_delta': avg_iou_delta,
                'critical_correction_count': critical_correction_count,
                'tool_is_wrong_count': tool_is_wrong_count,
                'refinement_count': refinement_count,
                'tool_is_correct_count': tool_is_correct_count,
                'avg_iou_gain_critical': avg_iou_gain_critical,
                'avg_iou_gain_refinement': avg_iou_gain_refinement,
                'avg_iou_worsen': avg_iou_worsen,
                'worsen_count': worsen_count,
                'follow_correct_rate': follow_correct_rate,
                'follow_correct_count': follow_correct_count,
                'nsri': nsri,
                'nsri_scaled_0_1': nsri_scaled_0_1,
                'inherited_error_rate': inherited_error_rate,
                'inherited_error_count': inherited_error_count,
                'final_wrong_count': final_wrong_count,
                'nsri_tool_correct': nsri_tool_correct,
                'nsri_tool_wrong': nsri_tool_wrong,
            }

    if csv_file:
        csv_output_path = os.path.join(results_dir, csv_file)
        save_metrics_to_csv(all_metrics, csv_output_path)

    print("\n--- Metrics Summary ---")
    sorted_datasets = sorted(all_metrics.keys())
    for dataset_name in sorted_datasets:
        metrics = all_metrics[dataset_name]
        print(f"\nDataset: {dataset_name} ({metrics['total_cases']} cases)")
        print(f"  - Accuracy: {metrics['accuracy']:.2f}%")
        print(f"  - Normalized Signed Relative IoU (NSRI): {metrics['nsri']:.2f}%")
        if metrics['tool_is_correct_count'] > 0:
            print(f"    - When Tool Correct: {metrics['nsri_tool_correct']:.2f}% ({metrics['tool_is_correct_count']} cases)")
        if metrics['tool_is_wrong_count'] > 0:
            print(f"    - When Tool Wrong: {metrics['nsri_tool_wrong']:.2f}% ({metrics['tool_is_wrong_count']} cases)")
        print(f"  - Scaled NSRI (0-1 Range): {metrics['nsri_scaled_0_1']:.4f}")
        print(f"  - Tool Accuracy: {metrics['tool_accuracy']:.2f}%")
        print(f"  - Avg. IoU Delta: {metrics['avg_iou_delta']:.4f}")
        print(f"  - Critical Correction Rate: {metrics['critical_correction_rate']:.2f}% ({metrics['critical_correction_count']}/{metrics['tool_is_wrong_count']})")
        if metrics['critical_correction_count'] > 0:
            print(f"    - Avg. Relative IoU Gain (Critical): {metrics['avg_iou_gain_critical']:.2f}%")
        print(f"  - Refinement Rate: {metrics['refinement_rate']:.2f}% ({metrics['refinement_count']}/{metrics['tool_is_correct_count']})")
        if metrics['refinement_count'] > 0:
            print(f"    - Avg. Relative IoU Gain (Refinement): {metrics['avg_iou_gain_refinement']:.2f}%")
        
        print(f"  - Follow Correct Rate: {metrics['follow_correct_rate']:.2f}% ({metrics['follow_correct_count']}/{metrics['tool_is_correct_count']})")
        
        worsen_rate = (metrics['worsen_count'] / metrics['total_cases']) * 100 if metrics['total_cases'] > 0 else 0
        print(f"  - Worsen Rate: {worsen_rate:.2f}% ({metrics['worsen_count']}/{metrics['total_cases']})")
        if metrics['worsen_count'] > 0:
            print(f"    - Avg. Relative IoU Worsen: {metrics['avg_iou_worsen']:.2f}%")
    
        if metrics['final_wrong_count'] > 0:
            print(f"  - Inherited Error Rate: {metrics['inherited_error_rate']:.2f}% ({metrics['inherited_error_count']}/{metrics['final_wrong_count']})")

    output_path = os.path.join(results_dir, output_file)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(collected_cases, f, indent=4, ensure_ascii=False)
        print(f"\nSuccessfully collected {len(collected_cases['critical_correction_cases'])} critical correction cases and {len(collected_cases['refinement_cases'])} refinement cases.")
        print(f"Saved to '{output_path}'")
    except IOError as e:
        print(f"Error: Could not write to output file '{output_path}': {e}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect and calculate metrics from grounding evaluation results.")
    parser.add_argument(
        "--results_dir", 
        type=str, 
        required=True, 
        help="The base directory where evaluation results for different datasets are stored."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="collected_cases.json",
        help="The name of the output JSON file to store the collected cases."
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        default="metrics_summary.csv",
        help="The name of the output CSV file to store the metrics summary."
    )
    args = parser.parse_args()
    
    collect_metrics(args.results_dir, args.output_file, args.csv_file)
