import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
base_model_name = "nferruz/ProtGPT2"
adapter_path = "./amp_genius_lora_final"

print("Загрузка токенизатора")
tokenizer = AutoTokenizer.from_pretrained(adapter_path)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

# Расширяем эмбеддинги под наши спец-теги
base_model.resize_token_embeddings(len(tokenizer))

#Подключаем обученный адаптер
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# Выбираем ускорение Apple Silicon (MPS) или CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
print(f"Модель готова к генерации на устройстве: {device}")

#управляющие теги 
# Запрашиваем пептид двойного действия [AMP][ACP], с высоким зарядом и короткой длиной
prompt_tags = "[AMP][ACP][CHARGE_HIGH_POS][LEN_SHORT]"
prompt = f"<|startoftext|>\n{prompt_tags}\n"

#Токенизация
inputs = tokenizer(prompt, return_tensors="pt").to(device)

print(f"\nЗапрос на генерацию под условия: {prompt_tags}")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=80,
        do_sample=True,
        top_k=50,
        top_p=0.92,
        temperature=0.7,
        num_return_sequences=3,
        pad_token_id=tokenizer.eos_token_id
    )

print("\nСгенерированные пептидные последовательности:")
for i, out in enumerate(outputs):
    decoded_text = tokenizer.decode(out, skip_special_tokens=False)
    print(f"\n--- Вариант {i+1} ---")
    print(decoded_text)