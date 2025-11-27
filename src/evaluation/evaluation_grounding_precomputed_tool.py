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
    pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(pattern, predict, re.DOTALL)
    
    # Debug: print what we're looking for
    if not hasattr(extract_answer_content, 'debug_count'):
        extract_answer_content.debug_count = 0
    
    if extract_answer_content.debug_count < 3:
        print(f"DEBUG: extract_answer_content - Looking for <answer> tags in: {predict[:500]}...", file=sys.stderr, flush=True)
        print(f"DEBUG: extract_answer_content - Found {len(matches)} matches: {matches}", file=sys.stderr, flush=True)
        extract_answer_content.debug_count += 1
    
    if len(matches) == 0:
        return ''
    else:
        # Return the last match
        return matches[-1]

def extract_think_content(predict):
    pattern = r"<think>(.*?)</think>"
    matches = re.findall(pattern, predict, re.DOTALL)
    if not matches:
        return ''
    return matches[-1].strip()

def extract_rethink_content(predict):
    pattern = r"<rethink>(.*?)</rethink>"
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

def parse_tool_bboxes(tool_response_str: str):
    """Parse precomputed tool_response into a list of bbox_2d lists."""
    if tool_response_str is None:
        return []
    try:
        data = json.loads(tool_response_str)
        # tool_response may be a list[dict] or a dict or already list of boxes
        if isinstance(data, dict):
            data = [data]
        if isinstance(data, list):
            # cases: [{"bbox_2d": [...]}, ...] or directly [[...], ...]
            bboxes = []
            for item in data:
                if isinstance(item, dict) and 'bbox_2d' in item:
                    if isinstance(item['bbox_2d'], list) and len(item['bbox_2d']) == 4:
                        bboxes.append(item['bbox_2d'])
                elif isinstance(item, list) and len(item) == 4:
                    bboxes.append(item)
            return bboxes
        return []
    except Exception:
        return []

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
        if isinstance(data, dict):
            data = [data]
        pred_bboxes = [item['bbox_2d'] for item in data if 'bbox_2d' in item]

        if not pred_bboxes or not gt_bbox:
            return 0.0, 0.0, None

        pred_bboxes = np.array(pred_bboxes)
        gt_bboxes = np.array([gt_bbox]) # Ground truth is a single bbox

        iou_matrix = batch_iou(pred_bboxes, gt_bboxes) # (M, 1)

        # Since we compare multiple predictions to one GT, we take the max IoU
        max_iou = np.max(iou_matrix) if iou_matrix.size > 0 else 0.0
        is_correct = 1.0 if max_iou > 0.5 else 0.0
        
        return max_iou, is_correct, pred_bboxes[0] if pred_bboxes.size > 0 else None

    except Exception as e:
        if calculate_iou_and_accuracy.debug_count < 3:
            print(f"DEBUG: Sample {calculate_iou_and_accuracy.debug_count-1} - Error calculating IoU: {e}, answer_content: {answer_content[:200]}", file=sys.stderr, flush=True)
        return 0.0, 0.0, None

def visualize_and_save_bbox(image, gt_bbox, tool_bboxes, tool_iou, pred_bbox, pred_iou, ref_exp, output_path):
    """
    Creates a composite visualization with three panels and saves it.
    - Left: Original image.
    - Middle: Tool BBoxes vs GT BBox.
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

    # Create the three panels
    original_img = image.copy()
    tool_img = image.copy()
    pred_img = image.copy()

    # Draw on tool_img (Middle panel)
    tool_draw = ImageDraw.Draw(tool_img)
    if gt_bbox:
        tool_draw.rectangle(gt_bbox, outline="green", width=3) # GT is green
    if tool_bboxes:
        for bbox in tool_bboxes:
            tool_draw.rectangle(bbox, outline="blue", width=2) # Tool is blue

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
    
    composite_width = 3 * width + 2 * padding
    composite_height = height + text_area_height
    
    composite_img = PILImage.new('RGB', (composite_width, composite_height), 'white')
    
    # Paste images onto the composite canvas
    composite_img.paste(original_img, (0, text_area_height))
    composite_img.paste(tool_img, (width + padding, text_area_height))
    composite_img.paste(pred_img, (2 * (width + padding), text_area_height))
    
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
    draw_text_centered(width + padding + width // 2, 10, "Tool Output vs GT (Tool: blue, GT: green)", font)
    draw_text_centered(2 * (width + padding) + width // 2, 10, "Final Answer vs GT (Answer: red, GT: green)", font)
    
    # IoU text below titles
    draw_text_centered(width + padding + width // 2, 30, f"Tool IoU: {tool_iou:.4f}", font)
    draw_text_centered(2 * (width + padding) + width // 2, 30, f"Final IoU: {pred_iou:.4f}", font)
    
    # Referring expression at the bottom of the text area
    # Simple one-line for now, may need wrapping for long text
    ref_exp_text = f"Referring Expression: {ref_exp}"
    draw_text_centered(composite_width // 2, 55, ref_exp_text, font)

    try:
        composite_img.save(output_path)
    except Exception as e:
        print(f"Error saving visualization to {output_path}: {e}")

# --- Model Class ---

class VGRefinerModel:
    def __init__(self, model_path):
        self.llm = LLM(
            model=model_path,
            limit_mm_per_prompt={"image": 10, "video": 10},
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.instruction_template = """{ref_exp}
To assist you in recognition, you can call the following function to obtain the recognition results of other referring tools:
```get_extra_ref_result()```

# Note #:
1. First, you should compare the difference between objects and find the most closely matched one. Then, output the thinking process in <think> </think>;
2. You should use the provided helper functions to obtain additional referring tool results to improve recognition performance. After initial recognition, call the function strictly according to the following format: <function>\nget_extra_ref_result()\n</function> . The function response will then be placed in <tool>\nfunction response\n</tool> .
3. Review the image content by referring to the function responses of other referring tools. You should analyze your initial referring results against those of other tools, paying particular attention to any discrepancies. (Both your own and other referring tools can make mistakes, so be sure to carefully review the image.) During this stage, your analysis and reflection should occur between the <rethink> and </rethink> blocks.
4. Finally, provide the final identified answer between <answer> and </answer> i.e.`<answer>{{"bbox_2d": [45, 120, 180, 250]}}</answer>`.

"""
    
    def ground_objects_batch(self, batch_images, batch_questions, batch_tool_responses):
        # --- STAGE 1: Get initial recognition ---
        llm_inputs_stage1 = []
        for image, question in zip(batch_images, batch_questions):
            instruction = self.instruction_template.format(ref_exp=question)
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
            llm_inputs_stage1.append({
                "prompt": prompt,
                "multi_modal_data": mm_data
            })

        # --- DEBUG: Print Stage 1 inputs ---
        print(f"DEBUG: Stage 1 inputs count: {len(llm_inputs_stage1)}", file=sys.stderr, flush=True)
        for i, input_data in enumerate(llm_inputs_stage1):
            print(f"DEBUG: Stage 1 input {i} prompt length: {len(input_data['prompt'])}", file=sys.stderr, flush=True)
            print(f"DEBUG: Stage 1 input {i} prompt last 500 chars: {input_data['prompt'][-500:]}", file=sys.stderr, flush=True)

        sampling_params_stage1 = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.05,
            max_tokens=4096,
            stop=["</function>"],
        )
        
        outputs_stage1 = self.llm.generate(llm_inputs_stage1, sampling_params=sampling_params_stage1)
        
        # --- DEBUG: Print Stage 1 outputs ---
        print(f"DEBUG: Stage 1 outputs count: {len(outputs_stage1)}", file=sys.stderr, flush=True)
        for i, output in enumerate(outputs_stage1):
            print(f"DEBUG: Stage 1 output {i}: {output.outputs[0].text[:200]}...", file=sys.stderr, flush=True)
        
        # --- STAGE 2: Use pre-computed tool response from the dataset ---
        llm_inputs_stage2 = []
        for i, output in enumerate(outputs_stage1):
            think_content = output.outputs[0].text.strip()
            
            # Get the pre-computed tool response
            tool_response_str = batch_tool_responses[i]
            tool_dict = {"reftool_1": tool_response_str}
            
            tool_block = f"""<tool>
{json.dumps(tool_dict, indent=2)}
</tool>"""

            prompt_stage2 = (
                llm_inputs_stage1[i]["prompt"].strip()
                + "\n"
                + think_content
                + "\n"
                + tool_block
            )
            
            # --- DEBUG: Print Stage 2 input ---
            print(f"DEBUG: Stage 2 input {i} length: {len(prompt_stage2)}", file=sys.stderr, flush=True)
            print(f"DEBUG: Stage 2 input {i} last 500 chars: {prompt_stage2[-500:]}", file=sys.stderr, flush=True)
            
            llm_inputs_stage2.append({
                "prompt": prompt_stage2,
                "multi_modal_data": llm_inputs_stage1[i]["multi_modal_data"]
            })

        # --- STAGE 3: Get final answer ---
        sampling_params_stage2 = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.05,
            max_tokens=4096,  # Increased from 4096 to 8192
        )
        
        outputs_stage2 = self.llm.generate(llm_inputs_stage2, sampling_params=sampling_params_stage2)

        # --- DEBUG: Print Stage 2 outputs ---
        print(f"DEBUG: Stage 2 outputs count: {len(outputs_stage2)}", file=sys.stderr, flush=True)
        for i, output in enumerate(outputs_stage2):
            print(f"DEBUG: Stage 2 output {i}: {output.outputs[0].text[:200]}...", file=sys.stderr, flush=True)

        batch_results = []
        for i, output_stage2 in enumerate(outputs_stage2):
            stage1_raw = outputs_stage1[i].outputs[0].text.strip()
            stage2_raw = output_stage2.outputs[0].text.strip()
            
            think_content = extract_think_content(stage1_raw)
            rethink_content = extract_rethink_content(stage2_raw)
            
            batch_results.append({
                "stage1_raw": stage1_raw,
                "stage2_raw": stage2_raw,
                "think": think_content,
                "rethink": rethink_content,
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
    
    model = VGRefinerModel(model_path=args.model_path)
    
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
    
    for j, pred_data in enumerate(batch_predictions_data):
        id_item = id_list[j]
        gt_bbox = id_item.get("bbox")
        
        if not gt_bbox:
            continue
        
        stage1_raw = pred_data["stage1_raw"]
        stage2_raw = pred_data["stage2_raw"]
        think_content = pred_data["think"]
        rethink_content = pred_data["rethink"]
        
        # Only debug first few samples to avoid too much output
        if len(all_outputs) < 3:
            print(f"DEBUG: Sample {len(all_outputs)} - Full prediction: {stage2_raw}", file=sys.stderr, flush=True)
        
        # LLM final prediction metrics
        iou, is_correct, pred_bbox_for_vis = calculate_iou_and_accuracy(stage2_raw, gt_bbox)
        
        tool_bboxes = parse_tool_bboxes(batch_tool_responses[j])
        tool_iou = 0.0
        tool_is_correct = 0.0
        if tool_bboxes:
            tool_pred_bboxes_np = np.array(tool_bboxes)
            gt_bboxes_np = np.array([gt_bbox])
            iou_matrix = batch_iou(tool_pred_bboxes_np, gt_bboxes_np)
            tool_iou = np.max(iou_matrix) if iou_matrix.size > 0 else 0.0
            tool_is_correct = 1.0 if tool_iou > 0.5 else 0.0
        
        if enable_visualize and vis_output_path:
            vis_save_path = os.path.join(vis_output_path, f"{id_item['image_id']}_{id_item['ann_id']}.jpg")
            visualize_and_save_bbox(
                image=id_item["original_image"],
                gt_bbox=gt_bbox,
                tool_bboxes=tool_bboxes,
                tool_iou=tool_iou,
                pred_bbox=pred_bbox_for_vis,
                pred_iou=iou,
                ref_exp=batch_questions[j],
                output_path=vis_save_path,
            )
        
        all_outputs.append({
            "image_id": id_item["image_id"],
            "ann_id": id_item["ann_id"],
            "prediction_raw": stage2_raw,
            "stage1_raw": stage1_raw,
            "think": think_content,
            "rethink": rethink_content,
            "gt_bbox": gt_bbox,
            "pred_bbox": pred_bbox_for_vis.tolist() if pred_bbox_for_vis is not None else None,
            "iou": iou,
            "is_correct": is_correct,
            # tool-only outputs
            "tool_pred_bboxes": tool_bboxes,
            "tool_iou": tool_iou,
            "tool_is_correct": tool_is_correct
        })

if __name__ == "__main__":
    main()
