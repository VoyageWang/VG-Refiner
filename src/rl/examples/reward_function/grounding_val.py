import re
import json
from scipy.optimize import linear_sum_assignment
import numpy as np
from typing import Dict, List

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
    

def iou_only_bbox_accuracy(predict_json_str: str, ground_truth: str) -> float:
    """只考虑IoU的bbox准确性奖励。
    现在这个函数只接收纯净的、从<answer>标签中提取出的JSON字符串。
    """
    max_iou_reward = 0.0
    MAX_OBJECTS = 120  # 设置上限

    # 如果没有提取到<answer>内容，则进行判断
    if not predict_json_str.strip():
        try:
            gt_data = json.loads(ground_truth)
            # 如果GT也为空，则预测正确，奖励为1
            if not gt_data or not [item for item in gt_data if 'bbox_2d' in item]:
                return 1.0
        except Exception:
            pass # GT格式可能也有问题
        # 否则，预测为空但GT不为空，奖励为0
        return 0.0
    
    try:
        gt_data = json.loads(ground_truth)
        gt_bboxes = [item['bbox_2d'] for item in gt_data if 'bbox_2d' in item]
            
        data = json.loads(predict_json_str)
        
        # --- ROBUSTNESS FIX: Handle both dict and list[dict] ---
        if isinstance(data, dict):
            data = [data] # Wrap single dict in a list
        
        pred_bboxes = [item['bbox_2d'] for item in data if 'bbox_2d' in item]
        
        # --- DEBUG PRINTS ---
        print("\n" + "="*20 + " REWARD DEBUG " + "="*20)
        print(f"PRED BBOXES (from answer): {pred_bboxes}")
        print(f"GT BBOXES:                 {gt_bboxes}")
        print("="*54 + "\n")

        # Handle cases where one or both are empty after parsing
        if not gt_bboxes and not pred_bboxes:
            return 1.0
        if not gt_bboxes or not pred_bboxes:
            return 0.0
        
        # 只有当预测或真实值超过上限时才截断
        if len(pred_bboxes) > MAX_OBJECTS:
            pred_bboxes = pred_bboxes[:MAX_OBJECTS]
        
        if len(gt_bboxes) > MAX_OBJECTS:
            gt_bboxes = gt_bboxes[:MAX_OBJECTS]
        
        # 预处理数据为numpy数组
        pred_bboxes = np.array(pred_bboxes)  # (M,4)
        gt_bboxes = np.array(gt_bboxes)    # (N,4)
        
        # 只计算IoU
        iou_matrix = batch_iou(pred_bboxes, gt_bboxes)  # (M,N)
        
        # 使用匈牙利算法找最优匹配，cost是1-IoU
        cost_matrix = 1.0 - iou_matrix
        
        # 使用匈牙利算法找最优匹配
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # 计算总IoU reward
        total_iou = iou_matrix[row_indices, col_indices].sum()
        
        # 计算平均IoU reward
        max_length = max(len(pred_bboxes), len(gt_bboxes))
        if max_length == 0:
            return 1.0  # Should be caught by earlier checks, but as a safeguard
        max_iou_reward = total_iou / max_length
            
    except Exception as e:
        # --- DEBUG PRINT ADDED ---
        print(f"[DEBUG REWARD] Exception caught in iou_only_bbox_accuracy. Error: {e}")
        print(f"[DEBUG REWARD] Failing JSON string from <answer> was: {predict_json_str[:500]}")
        pass
    return max_iou_reward

def iou_metrics_at_threshold(predict_json_str: str, ground_truth: str, threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculates precision and recall for predicted bounding boxes based on an IoU threshold.
    """
    metrics = {"precision_at_0.5": 0.0, "recall_at_0.5": 0.0}
    MAX_OBJECTS = 120

    if not predict_json_str.strip():
        try:
            gt_data = json.loads(ground_truth)
            if not gt_data or not [item for item in gt_data if 'bbox_2d' in item]:
                # Both pred and gt are empty, perfect score.
                return {"precision_at_0.5": 1.0, "recall_at_0.5": 1.0}
        except Exception:
            pass
        # Pred is empty but GT is not.
        return metrics

    try:
        gt_data = json.loads(ground_truth)
        gt_bboxes = [item['bbox_2d'] for item in gt_data if 'bbox_2d' in item]

        data = json.loads(predict_json_str)
        if isinstance(data, dict):
            data = [data]
        pred_bboxes = [item['bbox_2d'] for item in data if 'bbox_2d' in item]

        if not gt_bboxes and not pred_bboxes:
            return {"precision_at_0.5": 1.0, "recall_at_0.5": 1.0}
        
        num_pred = len(pred_bboxes)
        num_gt = len(gt_bboxes)

        if num_pred == 0:
            return {"precision_at_0.5": 1.0 if num_gt == 0 else 0.0, "recall_at_0.5": 1.0 if num_gt == 0 else 0.0}
        if num_gt == 0:
            return {"precision_at_0.5": 0.0, "recall_at_0.5": 1.0}
            
        if len(pred_bboxes) > MAX_OBJECTS:
            pred_bboxes = pred_bboxes[:MAX_OBJECTS]
        if len(gt_bboxes) > MAX_OBJECTS:
            gt_bboxes = gt_bboxes[:MAX_OBJECTS]

        pred_bboxes_np = np.array(pred_bboxes)
        gt_bboxes_np = np.array(gt_bboxes)

        iou_matrix = batch_iou(pred_bboxes_np, gt_bboxes_np)
        cost_matrix = 1.0 - iou_matrix
        
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        matched_ious = iou_matrix[row_indices, col_indices]
        num_correct = np.sum(matched_ious > threshold)

        metrics["precision_at_0.5"] = num_correct / num_pred
        metrics["recall_at_0.5"] = num_correct / num_gt

    except Exception:
        pass
    
    return metrics

def compute_score_withTool(reward_inputs, format_weight: float = 0.1) -> List[Dict[str, float]]:
    """工具调用版本的计算函数"""
    scores = []
    
    # Variables for correction_ratio
    tool_was_wrong_count = 0
    model_corrected_count = 0

    for reward_input in reward_inputs:
        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        ground_truth = reward_input["ground_truth"]
        
        # --- Correction Ratio Calculation ---
        tool_output_str = reward_input.get("tool_results", "[]") # Default to empty list if not present
        
        # 1. Check if tool output was wrong
        tool_accuracy = iou_only_bbox_accuracy(tool_output_str, ground_truth)
        tool_is_wrong = tool_accuracy < 0.5

        # 2. Check if model output is correct
        answer_content = extract_answer_content(predict)
        model_accuracy = iou_only_bbox_accuracy(answer_content, ground_truth)
        model_is_correct = model_accuracy >= 0.5

        # 3. Update counters
        is_corrected = 0
        if tool_is_wrong:
            tool_was_wrong_count += 1
            if model_is_correct:
                model_corrected_count += 1
                is_corrected = 1 # Mark this specific sample as a correction
        # --- End Correction Ratio Calculation ---


        # format score, a part of the original logic
        format_score = format_reward(predict)    
        
        # Calculate custom metrics (precision/recall)
        custom_metrics = iou_metrics_at_threshold(answer_content, ground_truth, threshold=0.5)

        scores.append(
            {
                "overall": (1 - format_weight) * model_accuracy + format_weight * format_score,
                "format": format_score,
                "accuracy": model_accuracy, # Use model_accuracy here
                "is_corrected": float(is_corrected), # 0.0 or 1.0 for this sample
                **custom_metrics,
            }
        )

    # After the loop, calculate the final correction ratio and add it to every sample's metrics
    # Wandb will average this across all batches to get the epoch-level metric.
    correction_ratio = (model_corrected_count / tool_was_wrong_count) if tool_was_wrong_count > 0 else 0.0
    for s in scores:
        s["correction_ratio"] = correction_ratio

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
    