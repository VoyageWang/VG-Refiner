import re
import json
from scipy.optimize import linear_sum_assignment
import numpy as np
from typing import Dict, List, Union

# TODO 
# 算法细节上还有点问题，就是reward好像不是 balanceweight的
# 整体计算的reward过程是否是合理的？ 可以check一个例子
# 是否需要 mask obsevation？ 不在这个地方
# 是否是工具调用了成功了才给奖励？ 不用 直接计算奖励

def extract_answer_content(predict):
    pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(pattern, predict, re.DOTALL)
    if len(matches) == 0:
        return ''
    else:
        return matches[-1]

def format_reward(response):
    """工具调用版本的格式检查"""
    must_shown_words = [
        # "<recognition>",
        # "</recognition>",
        "<think>",
        "</think>",
        "<rethink>",
        "</rethink>",
        "<answer>",
        "</answer>",
    ]
    for word in must_shown_words:
        if word not in response:
            return 0.0
    # Extract sections from response
    return 1.0

def format_reward_answer(response):
    """非工具调用版本的格式检查"""
    must_shown_words = [
        "<answer>",
        "</answer>",
    ]
    for word in must_shown_words:
        if word not in response:
            return 0.0
    return 1.0

def vision_reasoner_format_reward(predict_str: str) -> float:
    """原有的格式奖励函数，保留用于内部计算"""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    match = re.fullmatch(pattern, predict_str, re.DOTALL)
    thinking_format_reward = 1.0 if match else 0.0 
    
    def segmentation_format(predict_str: str) -> float:
        segmentation_format_reward = 0.0
        try:
            json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
            if not json_match:
                return segmentation_format_reward
            data = json.loads(json_match.group(1))
            
            data_cnt = len(data)
            
            for item in data:
                cur_reward = 0.0

                if 'bbox_2d' in item:
                    bbox_2d = item['bbox_2d']
                    if isinstance(bbox_2d, list) and len(bbox_2d) == 4:
                        cur_reward += 1.0
                    
                if 'point_2d' in item:
                    point_2d = item['point_2d']
                    if isinstance(point_2d, list) and len(point_2d) == 2:
                        cur_reward += 1.0
                
                segmentation_format_reward += cur_reward / data_cnt
        except Exception:
            pass
        return segmentation_format_reward
        
    segmentation_format_reward = segmentation_format(predict_str)
    
    return thinking_format_reward + segmentation_format_reward

def vision_reasoner_accuracy_reward(predict_str: str, ground_truth: str) -> float:
    max_accuracy_reward = 0.0
    MAX_OBJECTS = 120  # 设置上限
    
    try:
        gt_data = json.loads(ground_truth)
        gt_bboxes = [item['bbox_2d'] for item in gt_data]
        gt_points = [item['point_2d'] for item in gt_data]
            
        #json_match = re.search(r'```json\s*(.*?)\s*```', predict_str, re.DOTALL)
        json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
            pred_bboxes = [item['bbox_2d'] for item in data]
            pred_points = [item['point_2d'] for item in data]
            
            # 只有当预测或真实值超过上限时才截断
            if len(pred_bboxes) > MAX_OBJECTS:
                pred_bboxes = pred_bboxes[:MAX_OBJECTS]
                pred_points = pred_points[:MAX_OBJECTS]
            
            if len(gt_bboxes) > MAX_OBJECTS:
                gt_bboxes = gt_bboxes[:MAX_OBJECTS]
                gt_points = gt_points[:MAX_OBJECTS]
            
            # 预处理数据为numpy数组
            pred_bboxes = np.array(pred_bboxes)  # (M,4)
            pred_points = np.array(pred_points)  # (M,2)
            gt_bboxes = np.array(gt_bboxes)    # (N,4)
            gt_points = np.array(gt_points)     # (N,2)
            
            # 并行计算所有指标
            iou_matrix = batch_iou(pred_bboxes, gt_bboxes)  # (M,N)
            l1_matrix = batch_l1_distance(pred_bboxes, gt_bboxes)  # (M,N)
            points_dist_matrix = batch_points_distance(pred_points, gt_points)  # (M,N)
            points_in_box = batch_points_in_box(pred_points, pred_bboxes)  # (M,)
            
            # 计算reward矩阵
            iou_reward = (iou_matrix > 0.5).astype(float)
            bbox_l1_reward = (l1_matrix < 10).astype(float)
            point_reward = ((points_dist_matrix < 30) & points_in_box[:,np.newaxis]).astype(float)
            
            # 构建最终的cost矩阵
            cost_matrix = 3.0 - (iou_reward + bbox_l1_reward + point_reward)
            
            # 使用匈牙利算法找最优匹配
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # 直接从cost_matrix计算总reward
            total_reward = len(row_indices) * 3.0 - cost_matrix[row_indices, col_indices].sum()
            
            # 计算平均reward
            max_length = max(len(pred_bboxes), len(gt_bboxes))
            max_accuracy_reward = total_reward / max_length
            
    except Exception:
        pass
    return max_accuracy_reward

def vision_reasoner_non_repeat_reward(predict_str: str) -> float:
    non_repeat_reward = 1.0  # 初始满分
    try:
        sentences = predict_str.split('.')
        
        # 移除空句子
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 检查重复
        seen = set()
        repeats = 0
        
        for sentence in sentences:
            if sentence in seen:
                repeats += 1
            if repeats >=2:
                non_repeat_reward = 0
                break
            seen.add(sentence)
            
    except Exception:
        pass
    
    return non_repeat_reward

def vision_reasoner_compute_score(predict_str: str, ground_truth: str) -> float:
    """原有的计算逻辑，保留用于内部使用"""
    # print(predict_str, ground_truth)
    format_reward = vision_reasoner_format_reward(predict_str)
    accuracy_reward = vision_reasoner_accuracy_reward(predict_str, ground_truth)
    non_repeat_reward = vision_reasoner_non_repeat_reward(predict_str)
    
    reward = format_reward + accuracy_reward + non_repeat_reward
    return reward

def grounding_accuracy_reward(predict: str, ground_truth: str) -> float:
    """只考虑IoU的bbox准确性奖励 + 重复检测"""
    # 计算IoU-only的bbox准确性
    iou_reward = iou_only_bbox_accuracy(predict, ground_truth)
    # # 保留重复检测
    non_repeat_reward = vision_reasoner_non_repeat_reward(predict)
    # 组合：IoU reward + 重复检测
    return iou_reward + non_repeat_reward
    

def iou_only_bbox_accuracy(predict_input: Union[str, list, dict], ground_truth: str) -> float:
    """
    只考虑IoU的bbox准确性奖励。
    这个函数现在可以接收纯净的JSON字符串，或者已经解析过的list/dict。
    """
    max_iou_reward = 0.0
    MAX_OBJECTS = 120

    try:
        # 1. Parse Ground Truth
        gt_data = json.loads(ground_truth)
        gt_bboxes = [item['bbox_2d'] for item in gt_data if 'bbox_2d' in item]

        # 2. Parse Prediction Input (handles both str and already-parsed objects)
        data = None
        if isinstance(predict_input, str):
            if predict_input.strip():
                data = json.loads(predict_input)
        elif isinstance(predict_input, (list, dict)):
            data = predict_input

        pred_bboxes = []
        if data:
            if isinstance(data, dict):
                data = [data]  # Wrap single dict in a list for consistency
            pred_bboxes = [item['bbox_2d'] for item in data if 'bbox_2d' in item]

        # 3. Handle edge cases (empty predictions or ground truth)
        if not gt_bboxes and not pred_bboxes:
            return 1.0  # Both empty is a perfect match
        if not gt_bboxes or not pred_bboxes:
            return 0.0  # One is empty but the other is not

        # 4. Truncate if necessary
        if len(pred_bboxes) > MAX_OBJECTS:
            pred_bboxes = pred_bboxes[:MAX_OBJECTS]
        if len(gt_bboxes) > MAX_OBJECTS:
            gt_bboxes = gt_bboxes[:MAX_OBJECTS]

        # 5. Calculate IoU using Hungarian matching
        pred_bboxes_np = np.array(pred_bboxes)
        gt_bboxes_np = np.array(gt_bboxes)
        
        iou_matrix = batch_iou(pred_bboxes_np, gt_bboxes_np)
        cost_matrix = 1.0 - iou_matrix
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        total_iou = iou_matrix[row_indices, col_indices].sum()
        
        max_length = max(len(pred_bboxes), len(gt_bboxes))
        if max_length == 0:
            return 1.0  # Should be caught by earlier checks, but as a safeguard
            
        max_iou_reward = total_iou / max_length

    except Exception as e:
        # For debugging, you can uncomment these lines
        # print(f"[DEBUG REWARD] Exception in iou_only_bbox_accuracy. Error: {e}")
        # print(f"[DEBUG REWARD] Failing input was: {str(predict_input)[:500]}")
        pass  # Return 0.0 on any error
        
    return max_iou_reward

def calculate_refinement_reward(predict_str: str, ground_truth: str, tool_output_str: str) -> Dict[str, float]:
    """
    Calculates a reward based on the model's ability to refine a tool's output.
    - Tool Wrong, Model Correct (Correction): +1.0
    - Tool Correct, Model Correct (Confirmation): +0.5
    - Model Wrong (Penalty): 0
    """
    IOU_THRESHOLD = 0.5
    CORRECTION_REWARD = 1  # Highest reward for the most valuable action
    CONFIRMATION_REWARD = 0.5  # Positive reward, but lower than correction
    PENALTY = 0

    # Extract model's answer
    model_answer_content = extract_answer_content(predict_str)

    # Calculate accuracies
    tool_accuracy = iou_only_bbox_accuracy(tool_output_str, ground_truth)
    model_accuracy = iou_only_bbox_accuracy(model_answer_content, ground_truth)

    # Determine correctness
    tool_is_correct = tool_accuracy >= IOU_THRESHOLD
    model_is_correct = model_accuracy >= IOU_THRESHOLD

    # Assign reward based on refinement logic
    refinement_score = 0.0
    if model_is_correct:
        if tool_is_correct:
            refinement_score = CONFIRMATION_REWARD  # Confirmation
        else:
            refinement_score = CORRECTION_REWARD  # Correction
    else:
        refinement_score = PENALTY  # Penalty for any model error

    return {
        "refinement_score": refinement_score,
        "model_iou": model_accuracy,
        "tool_iou": tool_accuracy,
    }

def compute_score_withTool(reward_inputs, format_weight: float = 0.1) -> List[Dict[str, float]]:
    """工具调用版本的计算函数 - 使用 Refinement Reward"""
    
    # --- DEBUG PRINT ---
    if reward_inputs:
        print(f"[DEBUG REWARD] Keys in first reward_input: {reward_inputs[0].keys()}")
    # --- END DEBUG PRINT ---

    scores = []
    for reward_input in reward_inputs:
        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        ground_truth = reward_input["ground_truth"]
        # Safely get tool output. Now it should be present.
        tool_output_str = reward_input.get("tool_results", "[]")

        # 1. Calculate format score
        format_score = format_reward(predict)

        # 2. Calculate the new refinement reward
        refinement_metrics = calculate_refinement_reward(predict, ground_truth, tool_output_str)
        accuracy_score = refinement_metrics["refinement_score"]

        # 3. Combine scores for the final reward
        # We give the powerful refinement_score more weight
        final_reward = (1 - format_weight) * accuracy_score + format_weight * format_score

        scores.append(
            {
                "overall": final_reward,
                "format": format_score,
                "accuracy": accuracy_score, # This is now the refinement score
                "model_iou": refinement_metrics["model_iou"],
                "tool_iou": refinement_metrics["tool_iou"],
            }
        )
    return scores

def compute_score(reward_inputs, format_weight: float = 0.1) -> List[Dict[str, float]]:
    """非工具调用版本的计算函数"""
    scores = []
    for reward_input in reward_inputs:
        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        ground_truth = reward_input["ground_truth"]
        # format score
        format_score = format_reward_answer(predict)    
        # final score
        answer_content = extract_answer_content(predict)
        accuracy_score = iou_only_bbox_accuracy(answer_content, ground_truth)
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )
    return scores

def batch_iou(boxes1, boxes2):
    # boxes1: (M,4), boxes2: (N,4)
    # 广播机制自动扩展维度
    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)  # (M,1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)  # (N,1)
    
    xA = np.maximum(x11, np.transpose(x21))  # (M,N)
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    
    interArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)
    box1Area = (x12 - x11 + 1) * (y12 - y11 + 1)  # (M,1)
    box2Area = (x22 - x21 + 1) * (y22 - y21 + 1)  # (N,1)
    
    unionArea = box1Area + np.transpose(box2Area) - interArea
    iou = interArea / unionArea  # (M,N)
    return iou

def batch_l1_distance(boxes1, boxes2):
    # boxes1: (M,4), boxes2: (N,4)
    boxes1 = boxes1[:, np.newaxis, :]  # (M,1,4)
    boxes2 = boxes2[np.newaxis, :, :]  # (1,N,4)
    return np.mean(np.abs(boxes1 - boxes2), axis=2)  # (M,N)

def batch_points_distance(points1, points2):
    # points1: (M,2), points2: (N,2)
    points1 = points1[:, np.newaxis, :]  # (M,1,2)
    points2 = points2[np.newaxis, :, :]  # (1,N,2)
    
    # 计算欧氏距离
    dist = np.sqrt(np.sum((points1 - points2)**2, axis=2))  # (M,N)
    return dist

def batch_points_in_box(points, boxes):
    """
    检查每个点是否在对应的框内
    points: (M,2) - M个点的坐标
    boxes: (M,4) - M个框的坐标 [x1,y1,x2,y2]
    返回: (M,) 布尔数组
    """
    x_check = (points[:,0] >= boxes[:,0]) & (points[:,0] <= boxes[:,2])
    y_check = (points[:,1] >= boxes[:,1]) & (points[:,1] <= boxes[:,3])
    return x_check & y_check

if __name__ == "__main__":
    # --- SIMULATION ADDED BASED ON USER LOGS ---
    predict_str = """
<answer>{"bbox_2d": [11, 259, 639, 365]}</answer>
"""
    ground_truth = """[{"bbox_2d": [368, 294, 823, 830], "point_2d": [595, 562]}]"""
    
    print("--- Simulating reward calculation with user-provided examples ---")
    
    final_score_info = compute_score_withTool([{"response": predict_str, "ground_truth": ground_truth}])
    
    print(f"\nModel Prediction String:\n{predict_str}")
    print(f"Ground Truth String:\n{ground_truth}")
    print(f"\nCalculated Score Info: {final_score_info}\n")
    # --- END SIMULATION ---
    