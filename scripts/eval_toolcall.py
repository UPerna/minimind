import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
import math
import json
import time
import random
import argparse
import warnings
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from openai import OpenAI
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.language_examples import TRANSLATION_PAIRS
from trainer.trainer_utils import setup_seed, get_model_params, safe_math_eval
warnings.filterwarnings('ignore')

TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", "description": "Evaluate a mathematical expression with arithmetic, powers, and square roots", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "A mathematical expression, such as 123+456, 2**10, or sqrt(144)"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "Get the current date and time in a specified time zone", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "description": "Time zone name, such as Asia/Shanghai or America/New_York", "default": "Asia/Shanghai"}}, "required": []}}},
    {"type": "function", "function": {"name": "random_number", "description": "Generate a random number within the specified range", "parameters": {"type": "object", "properties": {"min": {"type": "integer", "description": "Minimum value", "default": 0}, "max": {"type": "integer", "description": "Maximum value", "default": 100}}, "required": []}}},
    {"type": "function", "function": {"name": "text_length", "description": "Count the characters and words in a text", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "Text to count"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "unit_converter", "description": "Convert units of length, weight, or temperature", "parameters": {"type": "object", "properties": {"value": {"type": "number", "description": "Numeric value to convert"}, "from_unit": {"type": "string", "description": "Source unit, such as km, miles, kg, pounds, celsius, or fahrenheit"}, "to_unit": {"type": "string", "description": "Target unit"}}, "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "get_current_weather", "description": "Get a city weather report, including temperature, humidity, and conditions", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City name, such as Beijing, Shanghai, or New York"}, "unit": {"type": "string", "description": "Temperature unit: celsius or fahrenheit", "enum": ["celsius", "fahrenheit"], "default": "celsius"}}, "required": ["location"]}}},
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "Look up the current exchange rate between two currencies", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string", "description": "Source currency code, such as USD, CNY, or EUR"}, "to_currency": {"type": "string", "description": "Target currency code, such as USD, CNY, or EUR"}}, "required": ["from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "Translate text into the target language", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "Text to translate"}, "target_language": {"type": "string", "description": "Target language, such as english, chinese, japanese, or french"}}, "required": ["text", "target_language"]}}},
]

MOCK_RESULTS = {
    "calculate_math": lambda args: {"result": str(safe_math_eval(args.get("expression", "0")))},
    "get_current_time": lambda args: {"datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "timezone": args.get("timezone", "Asia/Shanghai")},
    "random_number": lambda args: {"result": random.randint(int(args.get("min", 0)), int(args.get("max", 100)))},
    "text_length": lambda args: {"characters": len(args.get("text", "")), "words": len(args.get("text", "").split())},
    "unit_converter": lambda args: {"result": round(float(args.get("value", 0)) * 0.621371, 2), "from": f"{args.get('value', 0)} {args.get('from_unit', '')}", "to": args.get("to_unit", "")},
    "get_current_weather": lambda args: {"city": args.get("location"), "temperature": "22°C", "humidity": "65%", "condition": "sunny"},
    "get_exchange_rate": lambda args: {"from": args.get("from_currency", ""), "to": args.get("to_currency", ""), "rate": 7.15},
    "translate_text": lambda args: {"translated": "hello world"},
}

TOOL_MAP = {t["function"]["name"]: t for t in TOOLS}

def get_tools(names):
    return [TOOL_MAP[n] for n in names]

TEST_CASES = [
    {"prompt": "What is 256 times 37?", "tools": ["calculate_math", "get_current_time"]},
    {"prompt": "What time is it?", "tools": ["get_current_time", "random_number"]},
    {"prompt": "Convert 100 kilometers to miles.", "tools": ["unit_converter", "calculate_math"]},
    {"prompt": "Generate a random number from 1 to 1000, then calculate its square.", "tools": ["random_number", "calculate_math", "text_length"]},
    {"prompt": "What is the weather in Beijing today?", "tools": ["get_current_weather", "get_current_time"]},
    {"prompt": "Look up the US dollar to Chinese yuan exchange rate.", "tools": ["get_exchange_rate", "get_current_time"]},
    {"prompt": f"Translate '{TRANSLATION_PAIRS[0][0]}' into English.", "tools": ["translate_text", "text_length"]},
    {"prompt": "What is the weather in Tokyo? Also convert 30 celsius to fahrenheit.", "tools": ["get_current_weather", "unit_converter", "get_current_time"]},
]


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if os.path.basename(os.path.normpath(args.load_from)) == 'model':
        model = MiniMindForCausalLM(MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe)))
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    get_model_params(model, model.config)
    return model.half().eval().to(args.device), tokenizer


def parse_tool_calls(text):
    matches = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    calls = []
    for m in matches:
        try:
            calls.append(json.loads(m.strip()))
        except Exception:
            pass
    return calls


def parse_tool_call_from_text(content):
    pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    matches = re.findall(pattern, content, re.DOTALL)
    if not matches:
        return None
    tool_calls = []
    for i, match in enumerate(matches):
        try:
            data = json.loads(match)
            tool_calls.append({
                "id": f"call_{i}",
                "function": {"name": data.get("name", ""), "arguments": json.dumps(data.get("arguments", {}), ensure_ascii=False)}
            })
        except Exception:
            pass
    return tool_calls if tool_calls else None


def execute_tool(call, arguments=None):
    name = call.get("name", "") if isinstance(call, dict) else call
    try:
        raw_args = call.get("arguments", {}) if isinstance(call, dict) else arguments
        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
    except Exception:
        args = {}
    fn = MOCK_RESULTS.get(name)
    if not fn:
        return {"error": f"Unknown tool: {name}"}
    try:
        return fn(args)
    except Exception as e:
        return {"error": f"Tool execution failed: {str(e)[:80]}"}


def generate(model, tokenizer, messages, tools, args):
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools, open_thinking=False)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(args.device)
    st = time.time()
    print('🧠: ', end='')
    generated_ids = model.generate(
        inputs["input_ids"], attention_mask=inputs["attention_mask"],
        max_new_tokens=args.max_new_tokens, do_sample=args.temperature > 0, streamer=streamer,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        top_p=args.top_p, temperature=args.temperature
    )
    response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
    print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s') if args.show_speed else print()
    return response


def chat_api(client, messages, tools, args, stream=True):
    response = client.chat.completions.create(
        model=args.api_model, messages=messages, tools=tools,
        stream=stream, temperature=args.temperature,
        max_tokens=8192, top_p=args.top_p
    )
    if not stream:
        choice = response.choices[0]
        content = choice.message.content or ""
        tool_calls = choice.message.tool_calls
        if not tool_calls:
            tool_calls = parse_tool_call_from_text(content)
        print(f'🧠: {content}')
        return content, tool_calls
    print('🧠: ', end='', flush=True)
    content, tool_calls = "", None
    for chunk in response:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)
            content += delta.content
        if delta.tool_calls:
            if tool_calls is None:
                tool_calls = []
            for tc_chunk in delta.tool_calls:
                idx = tc_chunk.index if tc_chunk.index is not None else len(tool_calls)
                while len(tool_calls) <= idx:
                    tool_calls.append({
                        "id": "",
                        "function": {"name": "", "arguments": ""}
                    })
                if tc_chunk.id:
                    tool_calls[idx]["id"] += tc_chunk.id
                if tc_chunk.function:
                    if tc_chunk.function.name:
                        tool_calls[idx]["function"]["name"] += tc_chunk.function.name
                    if tc_chunk.function.arguments:
                        tool_calls[idx]["function"]["arguments"] += tc_chunk.function.arguments
    print()
    if not tool_calls:
        tool_calls = parse_tool_call_from_text(content)
    return content, tool_calls


def run_case(prompt, tools, args, model=None, tokenizer=None, client=None):
    messages = [{"role": "user", "content": prompt}]
    while True:
        if args.backend == 'local':
            content = generate(model, tokenizer, messages, tools, args)
            tool_calls = parse_tool_calls(content)
        else:
            content, tool_calls = chat_api(client, messages, tools, args, stream=bool(args.stream))
        if not tool_calls:
            break
        tool_calls = [{
            "id": tc.id if hasattr(tc, 'id') else tc.get("id", ""),
            "name": tc.function.name if hasattr(tc, 'function') else tc["function"]["name"],
            "arguments": tc.function.arguments if hasattr(tc, 'function') else tc["function"]["arguments"]
        } for tc in tool_calls] if args.backend == 'api' else tool_calls
        messages.append({"role": "assistant", "content": content} if args.backend == 'local' else {"role": "assistant", "content": content, "tool_calls": [{"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}} for tc in tool_calls]})
        for tc in tool_calls:
            name = tc["name"]
            arguments = tc["arguments"]
            print(f'📞 [Tool Calling]: {name} | args={arguments}')
            result = execute_tool(tc if args.backend == 'local' else name, arguments)
            print(f'✅ [Tool Called]: {json.dumps(result, ensure_ascii=False)}')
            messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)} if args.backend == 'local' else {"role": "tool", "content": json.dumps(result, ensure_ascii=False), "tool_call_id": tc["id"]})


def main():
    parser = argparse.ArgumentParser(description="Evaluate MiniMind tool calling")
    parser.add_argument('--backend', default='local', choices=['local', 'api'], type=str, help="Inference backend (local = local model, api = OpenAI-compatible endpoint)")
    parser.add_argument('--load_from', default='../model', type=str, help="Model path: use model for native PyTorch weights or a directory in Transformers format")
    parser.add_argument('--save_dir', default='../out', type=str, help="Directory containing model weights")
    parser.add_argument('--weight', default='full_sft', type=str, help="Weight filename prefix (pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo)")
    parser.add_argument('--hidden_size', default=768, type=int, help="Hidden size")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="Number of hidden layers")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="Use the MoE architecture (0 = no, 1 = yes)")
    parser.add_argument('--max_new_tokens', default=512, type=int, help="Maximum number of new tokens")
    parser.add_argument('--temperature', default=0.9, type=float, help="Sampling temperature (0 selects greedy decoding)")
    parser.add_argument('--top_p', default=0.9, type=float, help="Nucleus sampling threshold (0 to 1)")
    parser.add_argument('--show_speed', default=0, type=int, help="Show decoding speed in tokens per second")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="Device for inference")
    parser.add_argument('--api_base_url', default="http://localhost:11434/v1", type=str, help="Base URL for the OpenAI-compatible endpoint")
    parser.add_argument('--api_key', default='sk-123', type=str, help="API key for the OpenAI-compatible endpoint")
    parser.add_argument('--api_model', default='jingyaogong/minimind-3:latest', type=str, help="Model name to send in API requests")
    parser.add_argument('--stream', default=1, type=int, help="Stream responses in API mode (0 = no, 1 = yes)")
    args = parser.parse_args()
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("--temperature must be finite and nonnegative")

    model = tokenizer = client = None
    if args.backend == 'local': model, tokenizer = init_model(args)
    else: client = OpenAI(api_key=args.api_key, base_url=args.api_base_url)

    input_mode = int(input('[0] Run the example prompts\n[1] Enter prompts manually\n'))

    cases = [{"prompt": case["prompt"], "tools": get_tools(case["tools"]), "tool_names": case["tools"]} for case in TEST_CASES] if input_mode == 0 else iter(lambda: {"prompt": input('💬: '), "tools": TOOLS, "tool_names": [t["function"]["name"] for t in TOOLS]}, {"prompt": "", "tools": TOOLS, "tool_names": []})
    for case in cases:
        if not case["prompt"]: break
        setup_seed(random.randint(0, 31415926))
        if input_mode == 0:
            print(f'📦 Available tools: {case["tool_names"]}\n')
            print(f'💬: {case["prompt"]}')
        run_case(case["prompt"], case["tools"], args, model=model, tokenizer=tokenizer, client=client)
        print('\n' + '-' * 50 + '\n')


if __name__ == "__main__":
    main()
