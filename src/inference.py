from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import json
import re
from PIL import Image, ImageDraw

import os

def extract_answer_content(predict):
    pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(pattern, predict, re.DOTALL)
    if len(matches) == 0:
        return ''
    else:
        # Return the last match
        return matches[-1]

def extract_bbox(answer_content):
    if not answer_content:
        return None
    try:
        data = json.loads(answer_content)
        return data.get("bbox_2d")
    except (json.JSONDecodeError, TypeError):
        return None

# 模型路径
model_path = "./model_weights/VG-Refiner"
# 图片路径
image_path = "./assests/pizza.jpg"


image = Image.open(image_path)
draw = ImageDraw.Draw(image)

ref_exp = "a piece of pizza under a big piece of pizza in a vessel"

instruction = f"""{ref_exp}
To assist you in recognition, you can call the following function to obtain the recognition results of other referring tools:
```get_extra_ref_result()```

# Note #:
1. First, you should compare the difference between objects and find the most closely matched one. Then, output the thinking process in <think> </think>;
2. You should use the provided helper functions to obtain additional referring tool results to improve recognition performance. After initial recognition, call the function strictly according to the following format: <function>\nget_extra_ref_result()\n</function> . The function response will then be placed in <tool>\nfunction response\n</tool> .
3. Review the image content by referring to the function responses of other referring tools. You should analyze your initial referring results against those of other tools, paying particular attention to any discrepancies. (Both your own and other referring tools can make mistakes, so be sure to carefully review the image.) During this stage, your analysis and reflection should occur between the <rethink> and </rethink> blocks.
4. Finally, provide the final identified answer between <answer> and </answer> i.e.`<answer>{{"bbox_2d": [45, 120, 180, 250]}}</answer>`.

"""

tool1 = "[{\"bbox_2d\": [63,238,272,445]}]"

tool1_data = json.loads(tool1)
if tool1_data and 'bbox_2d' in tool1_data[0]:
    tool_bbox = tool1_data[0]['bbox_2d']
    draw.rectangle(tool_bbox, outline="blue", width=3)

tools = """<tool>
{{
    "reftool_1": "{tool1}"
}}
</tool>
"""

llm = LLM(
    model=model_path,
    limit_mm_per_prompt={"image": 10, "video": 10},
    gpu_memory_utilization=0.4,
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": instruction},
        ],
    },
]

prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
image_inputs, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)

mm_data = {}
if image_inputs is not None:
    mm_data["image"] = image_inputs

sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    repetition_penalty=1.05,
    max_tokens=4096,
    stop=["<tool>"],
)

llm_inputs = [
    {
        "prompt": prompt,
        "multi_modal_data": mm_data
    }
]

outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
think_content = outputs[0].outputs[0].text.strip()
print("#" * 20 + " think " + "#" * 20)
print(think_content)

llm_inputs[0]["prompt"] = (
    llm_inputs[0]["prompt"].strip()
    + "\n"
    + think_content
    + "\n"
    + tools.format(tool1=tool1)
)
sampling_params = SamplingParams(
    temperature=0.0, top_p=1.0, repetition_penalty=1.05, max_tokens=4096
)
outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
rethink_content = outputs[0].outputs[0].text.strip()
print("#" * 20 + " rethink " + "#" * 20)
print(rethink_content)

answer_content = extract_answer_content(rethink_content)
final_bbox = extract_bbox(answer_content)

if final_bbox:
    # Assuming final bbox is [xmin, ymin, xmax, ymax]
    draw.rectangle(final_bbox, outline="red", width=3)

image.save("inference_visualization.jpg")
print("Visualization saved to inference_visualization.jpg")
