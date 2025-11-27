import argparse
import torch
import json
import numpy as np
import os
from datasets import load_from_disk, load_dataset
from PIL import Image as PILImage, ImageDraw
from tqdm import tqdm
import sys
import re
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
import qwen_vl_utils
from scipy.optimize import linear_sum_assignment
import io
from PIL import Image, ImageDraw, ImageFont

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Helper functions from grounding.py ---

def extract_answer_content(predict):
    """Extract JSON content from model prediction."""
    # Debug: print what we're looking for
    if not hasattr(extract_answer_content, 'debug_count'):
        extract_answer_content.debug_count = 0
    
    if extract_answer_content.debug_count < 3:
        print(f"DEBUG: extract_answer_content - Looking for JSON in: {predict[:500]}...", file=sys.stderr, flush=True)
        extract_answer_content.debug_count += 1
    
    # Try to find JSON array in the prediction
    # Look for patterns like [{"bbox_2d": ...}] or {"bbox_2d": ...}
    try:
        # First try to parse the entire prediction as JSON
        data = json.loads(predict.strip())
        if isinstance(data, list) or isinstance(data, dict):
            return predict.strip()
    except json.JSONDecodeError:
        pass
    
    # If that fails, try to find JSON within the text
    # Look for array pattern
    array_pattern = r'\[.*?\]'
    array_matches = re.findall(array_pattern, predict, re.DOTALL)
    for match in array_matches:
        try:
            data = json.loads(match)
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and 'bbox_2d' in data[0]:
                return match
        except json.JSONDecodeError:
            continue
    
    # Look for single object pattern
    obj_pattern = r'\{[^{}]*"bbox_2d"[^{}]*\}'
    obj_matches = re.findall(obj_pattern, predict, re.DOTALL)
    for match in obj_matches:
        try:
            data = json.loads(match)
            if isinstance(data, dict) and 'bbox_2d' in data:
                return f"[{match}]"  # Wrap in array for consistency
        except json.JSONDecodeError:
            continue
    
    if extract_answer_content.debug_count < 3:
        print(f"DEBUG: extract_answer_content - No valid JSON found", file=sys.stderr, flush=True)
    
    return ''

def extract_think_content(predict):
    pattern = r"<think>(.*?)</think>"
    matches = re.findall(pattern, predict, re.DOTALL)
    if not matches:
        return ''
    return matches[-1].strip()

def batch_iou(boxes1, boxes2):
    # boxes1: (M,4), boxes2: (N,4)
    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)
    
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    
    interArea = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)
    box1Area = (x12 - x11) * (y12 - y11)
    box2Area = (x22 - x21) * (y22 - y21)
    
    unionArea = box1Area + np.transpose(box2Area) - interArea
    # Add a small epsilon to avoid division by zero
    iou = interArea / (unionArea + 1e-6)
    return iou

def extract_gt_bbox_from_GT(GT_str: str):
    """Parses a JSON string from the 'GT' field to extract the ground truth bbox."""
    if not GT_str:
        return None
    try:
        data = json.loads(GT_str)
        if isinstance(data, dict):
            data = [data]
        
        # Find the first item with a 'bbox_2d' and return it
        for item in data:
            if 'bbox_2d' in item and isinstance(item['bbox_2d'], list) and len(item['bbox_2d']) == 4:
                return item['bbox_2d']
        return None  # No valid bbox found in any item
    except (json.JSONDecodeError, TypeError):
        return None

def decode_image_field_to_pil(image_field):
    """Robustly decode dataset image field to a PIL.Image.
    Supports: PIL.Image, bytes, np.array([bytes], dtype=object), single-element list[bytes].
    """
    if isinstance(image_field, PILImage.Image):
        return image_field
    # datasets.Image may give dict with 'bytes'
    if isinstance(image_field, dict) and 'bytes' in image_field:
        return PILImage.open(io.BytesIO(image_field['bytes'])).convert("RGB")
    # raw bytes
    if isinstance(image_field, (bytes, bytearray)):
        return PILImage.open(io.BytesIO(image_field)).convert("RGB")
    # numpy object array with single bytes
    if isinstance(image_field, np.ndarray) and image_field.dtype == object:
        if image_field.size >= 1 and isinstance(image_field.flat[0], (bytes, bytearray)):
            return PILImage.open(io.BytesIO(image_field.flat[0])).convert("RGB")
    # list with single bytes
    if isinstance(image_field, list) and len(image_field) >= 1 and isinstance(image_field[0], (bytes, bytearray)):
        return PILImage.open(io.BytesIO(image_field[0])).convert("RGB")
    # Fallback: try numpy uint8 array as image
    if isinstance(image_field, np.ndarray) and image_field.dtype == np.uint8:
        try:
            return PILImage.fromarray(image_field).convert("RGB")
        except Exception:
            pass
    raise TypeError(f"Unsupported image field type: {type(image_field)}")

def calculate_iou_and_accuracy(pred_str, gt_bbox):
    """Calculates IoU and accuracy for a single prediction against a single ground truth."""
    answer_content = extract_answer_content(pred_str)
    
    # Only debug first few samples to avoid too much output
    global debug_count
    if not hasattr(calculate_iou_and_accuracy, 'debug_count'):
        calculate_iou_and_accuracy.debug_count = 0
    
    if calculate_iou_and_accuracy.debug_count < 3:
        print(f"DEBUG: Sample {calculate_iou_and_accuracy.debug_count} - Raw prediction: {pred_str[:200]}...", file=sys.stderr, flush=True)
        print(f"DEBUG: Sample {calculate_iou_and_accuracy.debug_count} - Extracted answer content: {answer_content}", file=sys.stderr, flush=True)
        print(f"DEBUG: Sample {calculate_iou_and_accuracy.debug_count} - GT bbox: {gt_bbox}", file=sys.stderr, flush=True)
        calculate_iou_and_accuracy.debug_count += 1
    
    if not answer_content.strip():
        return 0.0, 0.0, None

    try:
        data = json.loads(answer_content)
        # Ensure data is a list
        if isinstance(data, dict):
            data = [data]
        
        # Extract bbox_2d from each item
        pred_bboxes = []
        for item in data:
            if isinstance(item, dict) and 'bbox_2d' in item:
                bbox = item['bbox_2d']
                if isinstance(bbox, list) and len(bbox) == 4:
                    pred_bboxes.append(bbox)

        if not pred_bboxes:
             return 0.0, 0.0, None

        if not gt_bbox:
            # If there's a prediction but no GT, we can't calculate IoU.
            # Still, we can return the first predicted bbox for visualization.
            return 0.0, 0.0, np.array(pred_bboxes[0])

        pred_bboxes_np = np.array(pred_bboxes)
        gt_bboxes_np = np.array([gt_bbox]) # Ground truth is a single bbox

        iou_matrix = batch_iou(pred_bboxes_np, gt_bboxes_np) # (M, 1)

        # Since we compare multiple predictions to one GT, we take the max IoU
        max_iou = np.max(iou_matrix) if iou_matrix.size > 0 else 0.0
        best_pred_idx = np.argmax(iou_matrix) if iou_matrix.size > 0 else 0
        is_correct = 1.0 if max_iou > 0.5 else 0.0
        
        return max_iou, is_correct, pred_bboxes_np[best_pred_idx] if pred_bboxes else None

    except Exception as e:
        if calculate_iou_and_accuracy.debug_count < 3:
            print(f"DEBUG: Sample {calculate_iou_and_accuracy.debug_count-1} - Error calculating IoU: {e}, answer_content: {answer_content[:200]}", file=sys.stderr, flush=True)
        return 0.0, 0.0, None

def visualize_and_save_bbox(image, gt_bbox, pred_bbox, pred_iou, ref_exp, output_path):
    """
    Creates a composite visualization with two panels and saves it.
    - Left: Original image.
    - Right: Predicted BBox vs GT BBox.
    """
    if not isinstance(image, PILImage.Image):
        image = PILImage.fromarray(np.array(image))

    try:
        font = ImageFont.load_default(size=15)
    except AttributeError:
        # Fallback for older Pillow versions
        font = ImageFont.load_default()
    except IOError:
        font = None # No default font found

    # Create the two panels
    original_img = image.copy()
    pred_img = image.copy()

    # Draw on pred_img (Right panel)
    pred_draw = ImageDraw.Draw(pred_img)
    if gt_bbox:
        pred_draw.rectangle(gt_bbox, outline="green", width=3) # GT is green
    if pred_bbox is not None:
        pred_draw.rectangle(list(pred_bbox), outline="red", width=3) # Final answer is red

    # Create composite image layout
    width, height = image.size
    padding = 20
    text_area_height = 80
    
    composite_width = 2 * width + padding
    composite_height = height + text_area_height
    
    composite_img = PILImage.new('RGB', (composite_width, composite_height), 'white')
    
    # Paste images onto the composite canvas
    composite_img.paste(original_img, (0, text_area_height))
    composite_img.paste(pred_img, (width + padding, text_area_height))
    
    # Add text to the composite image
    draw = ImageDraw.Draw(composite_img)
    
    def draw_text_centered(x_center, y, text, font, fill="black"):
        if font:
            try: # Use anchor for newer Pillow versions for better centering
                draw.text((x_center, y), text, fill=fill, font=font, anchor="mt")
            except TypeError: # Fallback for older Pillow versions
                 text_width, text_height = draw.textsize(text, font)
                 draw.text((x_center - text_width / 2, y), text, fill=fill, font=font)
        else:
            draw.text((x_center, y), text, fill=fill)

    # Titles for each panel
    draw_text_centered(width // 2, 10, "Original Image", font)
    draw_text_centered(width + padding + width // 2, 10, "Final Answer vs GT (Answer: red, GT: green)", font)
    
    # IoU text below titles
    draw_text_centered(width + padding + width // 2, 30, f"Final IoU: {pred_iou:.4f}", font)
    
    # Referring expression at the bottom of the text area
    ref_exp_text = f"Referring Expression: {ref_exp}"
    draw_text_centered(composite_width // 2, 55, ref_exp_text, font)

    try:
        composite_img.save(output_path)
    except Exception as e:
        print(f"Error saving visualization to {output_path}: {e}")

# --- Model Class ---

class RefinerModel:
    def __init__(self, model_path):
        self.llm = LLM(
            model=model_path,
            limit_mm_per_prompt={"image": 10, "video": 10},
            gpu_memory_utilization=0.95,
            trust_remote_code=True,
            max_model_len=24000,
        )
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.instruction_template = """{ref_exp}

Here is the result from an external tool, which might be helpful. You should use it as a reference, but be aware that the tool can make mistakes. Carefully check the image yourself.
Tool Result:
{tool_result}

# Note #:
1. First, you should think about the user's request, look at the image, and consider the provided tool result to find the most closely matched object.
2. Finally, provide the final identified answer in JSON format:
[
    {{"bbox_2d": [x1, y1, x2, y2], "label": "obj_name/description"}},
]
"""
    
    def ground_objects_batch(self, batch_images, batch_questions, batch_tool_responses):
        llm_inputs = []
        for i, (image, question, tool_response) in enumerate(zip(batch_images, batch_questions, batch_tool_responses)):
            # DEBUG: Print length and snippet of tool_response to find the cause of long inputs
            tool_response_str = str(tool_response)
            print(f"DEBUG [Item {i}]: tool_response length = {len(tool_response_str)}", file=sys.stderr, flush=True)
            if len(tool_response_str) > 10000:
                print(f"DEBUG [Item {i}]: tool_response snippet = {tool_response_str[:500]}...", file=sys.stderr, flush=True)

            # Truncate the tool_response to avoid exceeding model's max length
            max_tool_response_len = 20000  # Leave some space for the rest of the prompt
            if len(tool_response) > max_tool_response_len:
                tool_response = tool_response[:max_tool_response_len]

            instruction = self.instruction_template.format(ref_exp=question, tool_result=tool_response)
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction},
                ]},
            ]
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            image_inputs, _, _ = qwen_vl_utils.process_vision_info(messages, return_video_kwargs=True)
            mm_data = {"image": image_inputs} if image_inputs is not None else {}
            llm_inputs.append({
                "prompt": prompt,
                "multi_modal_data": mm_data
            })

        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.05,
            max_tokens=4096,
        )
        
        outputs = self.llm.generate(llm_inputs, sampling_params=sampling_params)
        
        batch_results = []
        for i, output in enumerate(outputs):
            raw_output = output.outputs[0].text.strip()
            think_content = extract_think_content(raw_output)
            
            batch_results.append({
                "prediction_raw": raw_output,
                "think": think_content,
            })
        return batch_results

# --- Main Evaluation Logic ---

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--idx", type=int, required=True)
    parser.add_argument("--num_parts", type=int, required=True)
    parser.add_argument("--visualize", action="store_true", help="Enable visualization of bounding boxes")
    return parser.parse_args()

def main():
    args = parse_args()
    
    model = RefinerModel(model_path=args.model_path)
    
    print(f"Loading dataset from: {args.test_data_path}")
    if os.path.isdir(args.test_data_path):
        # Path is a directory. Check if it's a saved dataset or just data files.
        if (os.path.exists(os.path.join(args.test_data_path, "state.json")) or
            os.path.exists(os.path.join(args.test_data_path, "dataset_info.json"))):
            print(f"Loading saved dataset from disk: {args.test_data_path}")
            dataset = load_from_disk(args.test_data_path)
        else:
            # It's a directory of data files (e.g., Parquet)
            print(f"Loading data files from directory: {args.test_data_path}")
            dataset = load_dataset(args.test_data_path)
    else:
        # Path is not a directory, assume it's a Hugging Face Hub identifier.
        print(f"Loading dataset from Hugging Face Hub: {args.test_data_path}")
        dataset = load_dataset(args.test_data_path, split="test")

    # If we have a DatasetDict, we need to select a split.
    if isinstance(dataset, dict):
        if "test" in dataset:
            dataset = dataset["test"]
        elif "train" in dataset:
            dataset = dataset["train"]
        else:
            # Fallback: take the first available split
            split_name = list(dataset.keys())[0]
            print(f"Warning: 'test' or 'train' split not found. Using split: '{split_name}'")
            dataset = dataset[split_name]

    has_GT = 'ground_truth' in dataset.features and dataset[0]['ground_truth'] is not None
    if not has_GT:
        print("Error: The dataset does not contain the required 'ground_truth' field.")
        return
        
    
    has_tool = 'tool_results' in dataset.features and dataset[0]['tool_results'] is not None
    if not has_tool:
        print("Error: The dataset does not contain the required 'tool_results' field.")
        return

    # Parallel processing setup
    total_len = len(dataset)
    part_size = total_len // args.num_parts
    start_idx = args.idx * part_size
    end_idx = start_idx + part_size if args.idx < args.num_parts - 1 else total_len
    dataset = dataset.select(range(start_idx, end_idx))
    
    vis_output_path = None
    if args.visualize:
        vis_output_path = os.path.join(args.output_path, "visualizations")
        os.makedirs(vis_output_path, exist_ok=True)
    
    all_outputs = []
    
    for i in tqdm(range(0, len(dataset), args.batch_size), desc=f"Processing part {args.idx}/{args.num_parts}"):
        batch_data = [dataset[j] for j in range(i, min(i + args.batch_size, len(dataset)))]
        
        batch_images = [decode_image_field_to_pil(item["image"]).convert("RGB") for item in batch_data]
        batch_questions = [item["question"].lower().strip(".\"?!") for item in batch_data]
        batch_tool_responses = [item["tool_results"] for item in batch_data]
        
        id_list = []
        for item in batch_data:
            gt_bbox = extract_gt_bbox_from_GT(item["ground_truth"])
            # Debug GT bbox extraction for first few samples
            if len(id_list) < 3:
                print(f"DEBUG: Sample {len(id_list)} - ground_truth: {item['ground_truth']}", file=sys.stderr, flush=True)
                print(f"DEBUG: Sample {len(id_list)} - extracted GT bbox: {gt_bbox}", file=sys.stderr, flush=True)
            id_list.append({
                "image_id": item["image_id"],
                "ann_id": item["ann_id"],
                "bbox": gt_bbox, # The extracted GT bbox
                "original_image": decode_image_field_to_pil(item["image"]) if args.visualize else None
            })
        
        process_grounding_batch(model, batch_images, batch_questions, batch_tool_responses, id_list, all_outputs, vis_output_path, args.visualize)

    output_file = os.path.join(args.output_path, f"output_{args.idx}.json")
    with open(output_file, "w") as f:
        for item in all_outputs:
            item.pop("original_image", None) 
        json.dump(all_outputs, f, indent=2, ensure_ascii=False)
    print(f"Results for part {args.idx} saved to {output_file}")

def process_grounding_batch(model, batch_images, batch_questions, batch_tool_responses, id_list, all_outputs, vis_output_path, enable_visualize=False):
    """Process a batch of images and questions for grounding evaluation."""
    batch_predictions_data = model.ground_objects_batch(batch_images, batch_questions, batch_tool_responses)
    
    if len(batch_predictions_data) != len(id_list):
        print(
            f"警告：模型输入输出数量不匹配！\n"
            f"  输入样本数: {len(id_list)}\n"
            f"  模型输出数: {len(batch_predictions_data)}\n"
            f"  这会导致当前批次中部分样本丢失。",
            file=sys.stderr,
            flush=True
        )
    
    for j, pred_data in enumerate(batch_predictions_data):
        id_item = id_list[j]
        gt_bbox = id_item.get("bbox")
        
        # Tool-only metrics
        tool_response_str = batch_tool_responses[j]
        tool_iou, tool_is_correct, tool_bboxes = calculate_iou_and_accuracy(tool_response_str, gt_bbox)
        
        if not gt_bbox:
            continue
        
        prediction_raw = pred_data["prediction_raw"]
        think_content = pred_data["think"]
        
        # Only debug first few samples to avoid too much output
        if len(all_outputs) < 3:
            print(f"DEBUG: Sample {len(all_outputs)} - Full prediction: {prediction_raw}", file=sys.stderr, flush=True)
        
        # LLM final prediction metrics
        iou, is_correct, pred_bbox_for_vis = calculate_iou_and_accuracy(prediction_raw, gt_bbox)
        
        if enable_visualize and vis_output_path:
            vis_save_path = os.path.join(vis_output_path, f"{id_item['image_id']}_{id_item['ann_id']}.jpg")
            visualize_and_save_bbox(
                image=id_item["original_image"],
                gt_bbox=gt_bbox,
                pred_bbox=pred_bbox_for_vis,
                pred_iou=iou,
                ref_exp=batch_questions[j],
                output_path=vis_save_path,
            )
        
        all_outputs.append({
            "image_id": id_item["image_id"],
            "ann_id": id_item["ann_id"],
            "prediction_raw": prediction_raw,
            "think": think_content,
            "gt_bbox": gt_bbox,
            "pred_bbox": pred_bbox_for_vis.tolist() if pred_bbox_for_vis is not None else None,
            "iou": iou,
            "is_correct": is_correct,
            # tool-only outputs
            "tool_pred_bboxes": tool_bboxes.tolist() if tool_bboxes is not None else None,
            "tool_iou": tool_iou,
            "tool_is_correct": tool_is_correct,
        })

if __name__ == "__main__":
    main()
