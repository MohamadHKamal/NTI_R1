import argparse
import os
import cv2
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

# Load Face Embedding Model
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

def get_face_embedding(pil_img):
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = embedding_model(tensor).cpu().numpy()
    return embedding[0]

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Diffusion Model Wrapper
def load_pipeline(model_path, device):
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        use_safetensors=True
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    return pipe

def deidentify_face(input_path, base_output_dir, seeds, min_threshold, max_threshold, max_retries):
    MODEL_PATH = "SG161222/RealVisXL_V4.0"
    PROMPT = "Ultra-realistic portrait of a different identity, photorealistic skin texture, natural expression, different face feature"
    NEGATIVE_PROMPT = "same person, cartoon, anime, blurry, lowres, distorted face"

    # Create the "generatedFace" subfolder under the specified output directory
    output_dir = os.path.join(base_output_dir, "generatedFace")
    os.makedirs(output_dir, exist_ok=True)

    pipe = load_pipeline(MODEL_PATH, device)

    # Load original image
    original_img = Image.open(input_path).convert("RGB")
    cv_img = cv2.imread(input_path)
    h_img, w_img = cv_img.shape[:2]

    # Crop face area (reduce 5% from each side)
    reduce_ratio = 0.05
    reduce_side = int(w_img * reduce_ratio)
    x_new, y_new = reduce_side, 0
    w_new, h_new = w_img - 2 * reduce_side, h_img

    face_crop = cv_img[y_new:y_new+h_new, x_new:x_new+w_new]
    face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)).resize((768, 768))

    # Compute original embedding
    orig_embedding = get_face_embedding(face_pil)

    # Extract input image name (without extension)
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    for seed in seeds:
        generator = torch.Generator(device).manual_seed(seed)
        strength = 0.35
        retries = 0

        best_result = None
        best_distance = float("inf")

        while retries <= max_retries:
            # Generate synthetic face
            gen_face = pipe(
                prompt=PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                image=face_pil,
                strength=strength,
                guidance_scale=7,
                num_inference_steps=40,
                generator=generator
            ).images[0]

            gen_face_resized = gen_face.resize((w_new, h_new))
            output_img = original_img.copy()
            output_img.paste(gen_face_resized, (x_new, y_new))

            gen_embedding = get_face_embedding(gen_face_resized)
            sim = cosine_similarity(orig_embedding, gen_embedding)

            print(f"Seed {seed}, Strength {strength:.2f}, Similarity {sim:.4f}")

            if min_threshold <= sim <= max_threshold:
                final_path = os.path.join(output_dir, f"{base_name}_{seed}.jpg")
                output_img.save(final_path)
                print(f"Saved de-identified face (within range): {final_path}")
                break
            else:
                if sim < min_threshold:
                    distance = min_threshold - sim
                elif sim > max_threshold:
                    distance = sim - max_threshold
                else:
                    distance = 0

                if distance < best_distance:
                    best_distance = distance
                    best_result = (output_img.copy(), sim)

                if sim < min_threshold:
                    strength = max(0.25, strength - 0.05)
                    print("Too different, decreasing strength.")
                elif sim > max_threshold:
                    strength = min(0.75, strength + 0.05)
                    print("Too similar, increasing strength.")

                retries += 1

        if retries > max_retries and best_result is not None:
            final_path = os.path.join(output_dir, f"{base_name}_{seed}_closest.jpg")
            best_result[0].save(final_path)
            print(f"No result in range. Saved closest similarity ({best_result[1]:.4f}) as {final_path}")

# CLI Entrypoint
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face de-identification with Stable Diffusion XL")
    parser.add_argument("--input", type=str, required=True, help="Path to input face image (e.g., JPG/PNG)")
    parser.add_argument("--output", type=str, required=True, help="Base directory where 'generatedFace' folder will be created")
    parser.add_argument("--seeds", nargs="+", type=int, default=[12345], help="List of random seeds")
    parser.add_argument("--min_threshold", type=float, default=0.5, help="Minimum cosine similarity")
    parser.add_argument("--max_threshold", type=float, default=0.65, help="Maximum cosine similarity")
    parser.add_argument("--max_retries", type=int, default=3, help="Max number of retries per seed")
    args = parser.parse_args()

    deidentify_face(args.input, args.output, args.seeds, args.min_threshold, args.max_threshold, args.max_retries)
