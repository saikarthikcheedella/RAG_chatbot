import os
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import whoami, login
from dotenv import load_dotenv


# load_dotenv()
#print(os.getenv('HUGGING_FACE_API_KEY'))
#login(token=os.getenv('HUGGING_FACE_API_KEY'))

#print(os.getcwd())

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
CUSTOM_MODEL_PATH = "./llm/models/mistralai"



#tokenizer = BloomTokenizerFast.from_pretrained(model_name)
# model = BloomForCausalLM.from_pretrained(model_name)
# print(model)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CUSTOM_MODEL_PATH, low_cpu_mem_usage=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CUSTOM_MODEL_PATH, low_cpu_mem_usage=True)

tokenizer.pad_token = tokenizer.eos_token

def generate_text(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs["attention_mask"] = inputs["input_ids"].ne(tokenizer.pad_token_id)  # Ensure attention mask

    output = model.generate(
        **inputs,
        max_length=max_length,
        pad_token_id=tokenizer.pad_token_id  # Explicitly set PAD token
    )
    print("inside : ", output[0])
    return tokenizer.decode(output[0], skip_special_tokens=False)

response = generate_text("Tell me about AI models.")
print("response: ", response)


#inputs = tokenizer.encode("Hello How are you? .", return_tensors="pt")
outputs = generate_text("Welcome! How are you ?")
print("outputs: ", outputs)
#print(tokenizer.decode(outputs[0]))