import numpy as np
from PIL import Image
from transformers import AutoProcessor
import torch
import argparse
import os
import random
from tqdm import tqdm
from unipath.constants import *
from unipath.conversation import conv_templates
from unipath.model.builder import load_pretrained_model
from unipath.utils import disable_torch_init
from unipath.conversation import conv_templates
from unipath.retrieval import MultiModalRetriever


# ==================== 参数解析 ====================
parser = argparse.ArgumentParser(description='BLIP3o Image Generation Inference')
parser.add_argument('--model_path', type=str, 
                    default='/home/hmh/project/UniPath_Github/src/checkpoints',
                    help='Path to the trained model checkpoint')
parser.add_argument('--use_rag', type=bool, default=True,
                    help='Whether to use RAG retrieval for prototype features')
parser.add_argument('--rag_root_dir', type=str,
                    default='/home/hmh/nas/UniPath/UniPath-68k/RAG_8K',
                    help='Path to RAG root directory (expects fixed names for vocab/index/h5/images)')
parser.add_argument('--output_dir', type=str,
                    default='./generated_images',
                    help='Directory to save generated images')
parser.add_argument('--num_seeds', type=int, default=5,
                    help='Number of different seeds to generate for each sample')
args = parser.parse_args()


def resolve_rag_paths(rag_root_dir):
    rag_root_dir = os.path.abspath(os.path.expanduser(rag_root_dir))
    if not os.path.isdir(rag_root_dir):
        raise NotADirectoryError(f"RAG root directory does not exist: {rag_root_dir}")

    rag_vocab_path = os.path.join(rag_root_dir, 'llm_filtered_vocab_gemini_pro.txt')
    rag_index_path = os.path.join(rag_root_dir, 'keyword_inverted_index.json')
    rag_h5_file = os.path.join(rag_root_dir, 'selected_8k.h5')
    rag_image_dir = os.path.join(rag_root_dir, 'images')

    missing_paths = []
    if not os.path.isfile(rag_vocab_path):
        missing_paths.append(rag_vocab_path)
    if not os.path.isfile(rag_index_path):
        missing_paths.append(rag_index_path)
    if not os.path.isfile(rag_h5_file):
        missing_paths.append(rag_h5_file)
    if not os.path.isdir(rag_image_dir):
        missing_paths.append(rag_image_dir)

    if missing_paths:
        raise FileNotFoundError("Missing required RAG path(s):\n  - " + "\n  - ".join(missing_paths))

    return rag_vocab_path, rag_index_path, rag_h5_file, rag_image_dir

model_path = args.model_path
processor = AutoProcessor.from_pretrained("/home/hmh/nas/model_weight/Qwen2.5-VL-7B-Instruct")

disable_torch_init()
model_path = os.path.expanduser(model_path)
tokenizer, multi_model, context_len = load_pretrained_model(model_path)

retriever = None
print("Initializing RAG retriever...")

rag_vocab_path, rag_index_path, rag_h5_file, rag_image_dir = resolve_rag_paths(args.rag_root_dir)

retriever = MultiModalRetriever(
    h5_file=rag_h5_file,
    vocab_file=rag_vocab_path,
    inverted_index_file=rag_index_path,
    image_dir=rag_image_dir,
    device="cuda",
    load_conch=True
)

def add_template(prompt):
   conv = conv_templates['qwen'].copy()
   conv.append_message(conv.roles[0], prompt[0])
   conv.append_message(conv.roles[1], None)
   prompt = conv.get_prompt()
   return [prompt]

def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

os.makedirs(args.output_dir, exist_ok=True)

prompts = ["The tumor is arranged in solid sheets of large epithelioid cells with abundant eosinophilic cytoplasm. The nuclei are large and demonstrate marked pleomorphism, with irregular contours, coarse chromatin, and prominent nucleoli. Extensive hemorrhage is present, dissecting through the neoplastic cell population. The stroma is scant and fibrous."]
sample_keys = [f"sample_{i}" for i in range(len(prompts))]

total_images = len(prompts) * args.num_seeds
print(f"Will generate {len(prompts)} samples, {args.num_seeds} seeds per sample, total {total_images} images")

for idx, (prompt, sample_key) in enumerate(tqdm(zip(prompts, sample_keys), total=len(prompts), desc="Generating images")):
    print(f"\n[{idx+1}/{len(prompts)}] Processing sample: {sample_key}")
    print(f"Prompt: {prompt[:100]}...")
    
    for seed_idx in range(args.num_seeds):
        seed = seed_idx
        set_global_seed(seed)
        processed_prompt = add_template([f"Analysis of my description: {prompt}"])
        gen_img = multi_model.generate_image(
            text=processed_prompt,
            user_text=[prompt],
            tokenizer=tokenizer,
            retriever=retriever,
        )
        
        if args.num_seeds > 1:
            output_filename = f"{sample_key}_seed{seed}.png"
        else:
            output_filename = f"{sample_key}.png"
        
        output_path = os.path.join(args.output_dir, output_filename)
        gen_img[0].save(output_path)
        
        print(f"  Image Generated -> {output_filename} (seed={seed})")

print(f"\n✅ All images generated successfully!")
